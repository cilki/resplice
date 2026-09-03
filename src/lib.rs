pub use anyhow::Result;

use anyhow::{anyhow, Context};
use std::fs;
use std::path::Path;

mod arch;
mod link;
pub use arch::Endian;
pub use link::{link_rlib, Applied};

/// Page size used for laying out the injected segment.
const PAGE: u64 = 0x1000;

/// Round `x` up to the next multiple of `align` (a power of two).
fn align_up(x: u64, align: u64) -> u64 {
    (x + align - 1) & !(align - 1)
}

/// A single replacement extracted from an rlib.
///
/// `code` is the machine code emitted for the spliced function; `begin` and
/// `end` are the target address range it should overwrite in the original
/// binary.
#[derive(Debug, Clone)]
pub struct Splice {
    pub begin: u64,
    pub end: u64,
    pub code: Vec<u8>,
}

/// Represents a binary file that can be patched
pub struct Binary {
    data: Vec<u8>,
    format: BinaryFormat,
    arch: Architecture,
    is_64: bool,
    endian: Endian,
}

/// Supported binary formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryFormat {
    Elf,
    Pe,
    MachO,
}

/// Supported CPU architectures
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    X86,
    X86_64,
    Arm,
    Arm64,
    Mips,
    Mips64,
}

impl Architecture {
    /// Whether this architecture uses 64-bit pointers.
    fn is_64(self) -> bool {
        matches!(
            self,
            Architecture::X86_64 | Architecture::Arm64 | Architecture::Mips64
        )
    }
}

impl Binary {
    /// Load a binary file from disk
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let data = fs::read(path)?;
        let (format, arch, is_64, endian) = Self::detect_format_and_arch(&data)?;

        Ok(Binary {
            data,
            format,
            arch,
            is_64,
            endian,
        })
    }

    /// Detect the binary format, architecture, pointer width, and byte order.
    fn detect_format_and_arch(
        data: &[u8],
    ) -> Result<(BinaryFormat, Architecture, bool, Endian)> {
        match goblin::Object::parse(data)? {
            goblin::Object::Elf(elf) => {
                let format = BinaryFormat::Elf;
                let arch = match elf.header.e_machine {
                    goblin::elf::header::EM_386 => Architecture::X86,
                    goblin::elf::header::EM_X86_64 => Architecture::X86_64,
                    goblin::elf::header::EM_ARM => Architecture::Arm,
                    goblin::elf::header::EM_AARCH64 => Architecture::Arm64,
                    goblin::elf::header::EM_MIPS => {
                        // Determine if 32-bit or 64-bit MIPS
                        if elf.is_64 {
                            Architecture::Mips64
                        } else {
                            Architecture::Mips
                        }
                    }
                    _ => return Err(anyhow!("Unsupported binary format")),
                };
                let endian = if elf.little_endian {
                    Endian::Little
                } else {
                    Endian::Big
                };
                Ok((format, arch, elf.is_64, endian))
            }
            goblin::Object::PE(pe) => {
                let format = BinaryFormat::Pe;
                let arch = match pe.header.coff_header.machine {
                    goblin::pe::header::COFF_MACHINE_X86 => Architecture::X86,
                    goblin::pe::header::COFF_MACHINE_X86_64 => Architecture::X86_64,
                    goblin::pe::header::COFF_MACHINE_ARM => Architecture::Arm,
                    goblin::pe::header::COFF_MACHINE_ARM64 => Architecture::Arm64,
                    _ => return Err(anyhow!("Unsupported binary format")),
                };
                Ok((format, arch, arch.is_64(), Endian::Little))
            }
            goblin::Object::Mach(mach) => {
                use goblin::mach::Mach;
                let format = BinaryFormat::MachO;
                let arch = match mach {
                    Mach::Binary(macho) => match macho.header.cputype {
                        goblin::mach::cputype::CPU_TYPE_X86 => Architecture::X86,
                        goblin::mach::cputype::CPU_TYPE_X86_64 => Architecture::X86_64,
                        goblin::mach::cputype::CPU_TYPE_ARM => Architecture::Arm,
                        goblin::mach::cputype::CPU_TYPE_ARM64 => Architecture::Arm64,
                        _ => return Err(anyhow!("Unsupported binary format")),
                    },
                    Mach::Fat(_) => {
                        // For fat binaries, default to x86_64 for now
                        Architecture::X86_64
                    }
                };
                Ok((format, arch, arch.is_64(), Endian::Little))
            }
            _ => Err(anyhow!("Unsupported binary format")),
        }
    }

    /// Detect just the binary format.
    #[allow(dead_code)]
    fn detect_format(data: &[u8]) -> Result<BinaryFormat> {
        let (format, ..) = Self::detect_format_and_arch(data)?;
        Ok(format)
    }

    /// Apply a splice by directly patching the binary.
    ///
    /// `begin`/`end` are **file offsets** into the raw binary image; this is the
    /// low-level primitive. Callers working in virtual-address space translate
    /// via [`Binary::va_to_offset`] first (see [`link`]).
    pub fn apply_direct_patch(&mut self, begin: u64, end: u64, code: &[u8]) -> Result<()> {
        let size = (end - begin) as usize;
        if code.len() > size {
            return Err(anyhow!("Invalid address range: {:#x} to {:#x}", begin, end));
        }
        self.patch_bytes(begin as usize, size, code)
    }

    /// Write `code` at file offset `offset`, filling any trailing space up to
    /// `region_len` with NOP instructions. `code` must not be longer than
    /// `region_len`.
    pub fn patch_bytes(&mut self, offset: usize, region_len: usize, code: &[u8]) -> Result<()> {
        if code.len() > region_len {
            return Err(anyhow!(
                "replacement ({} bytes) larger than region ({} bytes)",
                code.len(),
                region_len
            ));
        }

        let end_pos = offset + code.len();
        if offset + region_len > self.data.len() {
            return Err(anyhow!(
                "patch region {:#x}..{:#x} is outside the binary",
                offset,
                offset + region_len
            ));
        }

        self.data[offset..end_pos].copy_from_slice(code);

        // If the new code is smaller than the region, fill the rest with NOPs.
        if code.len() < region_len {
            let nop_insn = self.get_nop_instruction();
            let remaining = offset + region_len - end_pos;
            let mut written = 0;
            while written < remaining {
                let n = std::cmp::min(nop_insn.len(), remaining - written);
                self.data[end_pos + written..end_pos + written + n].copy_from_slice(&nop_insn[..n]);
                written += n;
            }
        }

        Ok(())
    }

    /// Get the NOP instruction bytes for the current architecture, encoded in
    /// the target's byte order.
    fn get_nop_instruction(&self) -> Vec<u8> {
        match self.arch {
            Architecture::X86 | Architecture::X86_64 => vec![0x90],
            Architecture::Arm => self.encode_insn(0xe1a0_0000), // mov r0, r0
            Architecture::Arm64 => self.encode_insn(0xd503_201f), // nop
            Architecture::Mips | Architecture::Mips64 => vec![0, 0, 0, 0],
        }
    }

    /// Encode a 32-bit instruction word in the target's byte order.
    fn encode_insn(&self, insn: u32) -> Vec<u8> {
        let mut b = vec![0u8; 4];
        self.endian.write_uint(&mut b, 0, 4, insn as u64);
        b
    }

    /// Machine code for an unconditional jump from `from_va` to `to_va`, for the
    /// current architecture. Used to build trampolines for oversized splices.
    pub fn jump_bytes(&self, from_va: u64, to_va: u64) -> Result<Vec<u8>> {
        self.generate_jump_instruction(from_va, to_va)
    }

    /// Apply a splice using an unconditional jump
    pub fn apply_jump_patch(&mut self, begin: u64, _end: u64, target: u64) -> Result<()> {
        let jump_code = self.generate_jump_instruction(begin, target)?;
        let start = begin as usize;

        if start + jump_code.len() > self.data.len() {
            return Err(anyhow!(
                "Invalid address range: {:#x} to {:#x}",
                begin,
                begin + jump_code.len() as u64
            ));
        }

        self.data[start..start + jump_code.len()].copy_from_slice(&jump_code);

        Ok(())
    }

    /// Generate an unconditional jump instruction for the current architecture
    fn generate_jump_instruction(&self, from: u64, to: u64) -> Result<Vec<u8>> {
        match self.arch {
            Architecture::X86 | Architecture::X86_64 => {
                // JMP rel32 (E9 XX XX XX XX)
                let offset = (to as i64 - (from as i64 + 5)) as i32;
                Ok(vec![
                    0xE9,
                    (offset & 0xFF) as u8,
                    ((offset >> 8) & 0xFF) as u8,
                    ((offset >> 16) & 0xFF) as u8,
                    ((offset >> 24) & 0xFF) as u8,
                ])
            }
            Architecture::Arm => {
                // ARM branch instruction: B <offset>
                // Encoding: 0xEA000000 | ((offset >> 2) & 0x00FFFFFF)
                // Offset is calculated as (target - pc - 8) / 4
                let pc = from + 8; // ARM PC is 2 instructions ahead
                let offset = ((to as i64 - pc as i64) / 4) as i32;

                if !(-0x800000..=0x7FFFFF).contains(&offset) {
                    return Err(anyhow!("Invalid address range: {:#x} to {:#x}", from, to));
                }

                let insn = 0xEA000000u32 | ((offset as u32) & 0x00FFFFFF);
                Ok(self.encode_insn(insn))
            }
            Architecture::Arm64 => {
                // ARM64 branch instruction: B <offset>
                // Encoding: 0x14000000 | ((offset >> 2) & 0x03FFFFFF)
                let offset = ((to as i64 - from as i64) / 4) as i32;

                if !(-0x2000000..=0x1FFFFFF).contains(&offset) {
                    return Err(anyhow!("Invalid address range: {:#x} to {:#x}", from, to));
                }

                let insn = 0x14000000u32 | ((offset as u32) & 0x03FFFFFF);
                Ok(self.encode_insn(insn))
            }
            Architecture::Mips | Architecture::Mips64 => {
                // MIPS J instruction: J <address>
                // Encoding: 0x08000000 | ((address >> 2) & 0x03FFFFFF)
                // Note: the target must lie in the same 256MB region as the delay
                // slot; encoded in the target's byte order (BE for mips, LE for
                // mipsel).
                let addr_bits = ((to >> 2) & 0x03FFFFFF) as u32;
                let insn = 0x08000000u32 | addr_bits;
                Ok(self.encode_insn(insn))
            }
        }
    }

    /// Save the patched binary to disk
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        fs::write(path, &self.data)?;
        Ok(())
    }

    /// Get a reference to the binary data
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Get the binary format
    pub fn format(&self) -> BinaryFormat {
        self.format
    }

    /// Get the architecture
    pub fn architecture(&self) -> Architecture {
        self.arch
    }

    /// Whether the target uses 64-bit pointers.
    pub fn is_64(&self) -> bool {
        self.is_64
    }

    /// The target's byte order.
    pub fn endian(&self) -> Endian {
        self.endian
    }

    /// Parse this binary as an ELF image (errors for non-ELF targets).
    fn elf(&self) -> Result<goblin::elf::Elf<'_>> {
        match goblin::Object::parse(&self.data)? {
            goblin::Object::Elf(elf) => Ok(elf),
            _ => Err(anyhow!(
                "relocation-aware splicing is only supported for ELF targets"
            )),
        }
    }

    /// Translate a virtual address to a file offset using the program headers.
    pub fn va_to_offset(&self, va: u64) -> Result<usize> {
        use goblin::elf::program_header::PT_LOAD;
        let elf = self.elf()?;
        for ph in &elf.program_headers {
            if ph.p_type == PT_LOAD && va >= ph.p_vaddr && va < ph.p_vaddr + ph.p_filesz {
                return Ok((ph.p_offset + (va - ph.p_vaddr)) as usize);
            }
        }
        Err(anyhow!("virtual address {va:#x} is not mapped by any PT_LOAD segment"))
    }

    /// The highest virtual address occupied by any loadable segment.
    fn max_vaddr(&self) -> Result<u64> {
        use goblin::elf::program_header::PT_LOAD;
        let elf = self.elf()?;
        Ok(elf
            .program_headers
            .iter()
            .filter(|ph| ph.p_type == PT_LOAD)
            .map(|ph| ph.p_vaddr + ph.p_memsz)
            .max()
            .unwrap_or(0))
    }

    /// The base virtual address the injected segment will be mapped at:
    /// one page above the target's current image, page-aligned.
    pub fn injected_base(&self) -> Result<u64> {
        Ok(align_up(self.max_vaddr()?, PAGE) + PAGE)
    }

    /// Resolve a symbol name against the target binary's own symbols.
    ///
    /// Tries, in order: a defined symbol in `.symtab`, a defined symbol in
    /// `.dynsym`, then a function the target imports through its PLT (the stub's
    /// address). Returns `None` if the target provides no such symbol.
    pub fn resolve_target_symbol(&self, name: &str) -> Result<Option<u64>> {
        use goblin::elf::section_header::SHN_UNDEF;
        let elf = self.elf()?;

        // 1 & 2: a symbol the target defines itself.
        for (syms, strtab) in [(&elf.syms, &elf.strtab), (&elf.dynsyms, &elf.dynstrtab)] {
            for sym in syms.iter() {
                if sym.st_shndx as u32 == SHN_UNDEF || sym.st_value == 0 {
                    continue;
                }
                if strtab.get_at(sym.st_name) == Some(name) {
                    return Ok(Some(sym.st_value));
                }
            }
        }

        // 3: a function the target imports through the PLT. The i-th `.rela.plt`
        // entry maps to the `.plt` stub at `plt_base + header + i * entry`, whose
        // sizes are architecture-specific. (MIPS classically uses `.MIPS.stubs`
        // rather than a `.plt` of this shape; its import binding is best-effort
        // and target-*defined* symbols above are the reliable path there.)
        let (plt_header, plt_entry) = match self.arch {
            Architecture::X86 | Architecture::X86_64 => (16, 16),
            Architecture::Arm64 => (32, 16),
            Architecture::Arm => (20, 12),
            Architecture::Mips | Architecture::Mips64 => (32, 16),
        };
        let plt_addr = elf
            .section_headers
            .iter()
            .find(|sh| elf.shdr_strtab.get_at(sh.sh_name) == Some(".plt"))
            .map(|sh| sh.sh_addr);
        if let Some(plt_addr) = plt_addr {
            for (i, reloc) in elf.pltrelocs.iter().enumerate() {
                if let Some(sym) = elf.dynsyms.get(reloc.r_sym) {
                    if elf.dynstrtab.get_at(sym.st_name) == Some(name) {
                        return Ok(Some(plt_addr + plt_header + i as u64 * plt_entry));
                    }
                }
            }
        }

        Ok(None)
    }

    /// Inject `blob` as a new read+execute segment mapped at virtual address
    /// `base`, by converting an existing `PT_NOTE` program header into a
    /// `PT_LOAD` (so the program-header table itself need not be relocated).
    ///
    /// `base` must be page-aligned; the file is padded to a page boundary before
    /// the blob is appended so the loader's `p_vaddr ≡ p_offset (mod p_align)`
    /// requirement holds.
    pub fn inject_segment(&mut self, base: u64, blob: &[u8]) -> Result<()> {
        use goblin::elf::program_header::{PF_R, PF_X, PT_LOAD, PT_NOTE};

        if base % PAGE != 0 {
            return Err(anyhow!("injected base {base:#x} is not page-aligned"));
        }
        let expected_class = if self.is_64 { 2 } else { 1 };
        let expected_data = match self.endian {
            Endian::Little => 1,
            Endian::Big => 2,
        };
        if self.format != BinaryFormat::Elf
            || self.data.len() < 64
            || self.data[..4] != [0x7f, b'E', b'L', b'F']
            || self.data[4] != expected_class
            || self.data[5] != expected_data
        {
            return Err(anyhow!("segment injection requires an ELF target"));
        }
        let endian = self.endian;

        // Program-header table location and entry size, at class-dependent
        // offsets in the ELF header.
        let (phoff_off, phentsize_off, phnum_off, phentsize) = if self.is_64 {
            (0x20, 0x36, 0x38, 56usize)
        } else {
            (0x1c, 0x2a, 0x2c, 32usize)
        };
        let word = if self.is_64 { 8 } else { 4 };
        let phoff = endian.read_uint(&self.data, phoff_off, word) as usize;
        let got_phentsize = endian.read_uint(&self.data, phentsize_off, 2) as usize;
        let phnum = endian.read_uint(&self.data, phnum_off, 2) as usize;
        if got_phentsize != phentsize {
            return Err(anyhow!("unexpected program-header entry size {got_phentsize}"));
        }

        // Find a PT_NOTE entry to repurpose (p_type is the first word of every
        // program header, in both ELF classes).
        let mut note_off = None;
        for i in 0..phnum {
            let off = phoff + i * phentsize;
            if endian.read_uint(&self.data, off, 4) as u32 == PT_NOTE {
                note_off = Some(off);
                break;
            }
        }

        // Many statically-linked targets carry no PT_NOTE at all,
        // just REGINFO + a couple of PT_LOADs. Rather than give up, grow the
        // program-header table by one entry into the padding gap between the
        // table itself and whatever comes first after it in the file (the
        // start of file content any existing segment or the section-header
        // table claims) — the same trick linkers use to leave slack for
        // PT_NOTE in the first place. This never touches bytes any parser
        // needs, since it strictly stays inside currently-unclaimed padding.
        let note_off = match note_off {
            Some(off) => off,
            None => {
                let (poff_field, poff_w) = if self.is_64 { (8, 8) } else { (4, 4) };
                let mut first_claimed = u64::MAX;
                for i in 0..phnum {
                    let off = phoff + i * phentsize;
                    let p_off = endian.read_uint(&self.data, off + poff_field, poff_w);
                    if p_off > 0 {
                        first_claimed = first_claimed.min(p_off);
                    }
                }
                let (eshoff_off, eshoff_w, eshentsize_off, eshnum_off) = if self.is_64 {
                    (0x28, 8, 0x3a, 0x3c)
                } else {
                    (0x20, 4, 0x2e, 0x30)
                };
                let e_shoff = endian.read_uint(&self.data, eshoff_off, eshoff_w);
                if e_shoff > 0 {
                    first_claimed = first_claimed.min(e_shoff);
                }
                let _ = (eshentsize_off, eshnum_off); // only the start of the shdr table matters here

                let table_end = (phoff + phnum * phentsize) as u64;
                if first_claimed == u64::MAX || first_claimed < table_end {
                    return Err(anyhow!(
                        "no PT_NOTE segment to convert, and no room to grow the \
                         program-header table (next file content starts at \
                         {first_claimed:#x}, table ends at {table_end:#x})"
                    ));
                }
                let gap = first_claimed - table_end;
                if gap < phentsize as u64 {
                    return Err(anyhow!(
                        "no PT_NOTE segment to convert, and insufficient padding to add \
                         a new program header ({gap} bytes available, {phentsize} needed)"
                    ));
                }

                let new_phnum = phnum + 1;
                if new_phnum > u16::MAX as usize {
                    return Err(anyhow!("program header count overflow"));
                }
                endian.write_uint(&mut self.data, phnum_off, 2, new_phnum as u64);
                table_end as usize
            }
        };

        // Append the blob at a page-aligned file offset.
        let file_off = align_up(self.data.len() as u64, PAGE);
        self.data.resize(file_off as usize, 0);
        self.data.extend_from_slice(blob);
        let len = blob.len() as u64;

        // Rewrite the note phdr in place as a PT_LOAD covering the appended blob.
        // The 32- and 64-bit program-header layouts differ in field order and
        // width (notably `p_flags` moves from offset 4 to offset 24 in ELF32).
        let flags = (PF_R | PF_X) as u64;
        let d = &mut self.data;
        endian.write_uint(d, note_off, 4, PT_LOAD as u64);
        if self.is_64 {
            endian.write_uint(d, note_off + 4, 4, flags);
            endian.write_uint(d, note_off + 8, 8, file_off);
            endian.write_uint(d, note_off + 16, 8, base);
            endian.write_uint(d, note_off + 24, 8, base);
            endian.write_uint(d, note_off + 32, 8, len);
            endian.write_uint(d, note_off + 40, 8, len);
            endian.write_uint(d, note_off + 48, 8, PAGE);
        } else {
            endian.write_uint(d, note_off + 4, 4, file_off);
            endian.write_uint(d, note_off + 8, 4, base);
            endian.write_uint(d, note_off + 12, 4, base);
            endian.write_uint(d, note_off + 16, 4, len);
            endian.write_uint(d, note_off + 20, 4, len);
            endian.write_uint(d, note_off + 24, 4, flags);
            endian.write_uint(d, note_off + 28, 4, PAGE);
        }

        Ok(())
    }
}

/// Parse a `.rspl.<begin>.<end>` section name into its address range.
///
/// Returns `None` for any section that is not a splice section.
fn parse_splice_section(name: &str) -> Option<Result<(u64, u64)>> {
    let rest = name.strip_prefix(".rspl.")?;
    Some((|| {
        let (begin, end) = rest
            .split_once('.')
            .ok_or_else(|| anyhow!("malformed splice section name: {name:?}"))?;
        let begin = u64::from_str_radix(begin, 16)
            .with_context(|| format!("invalid begin address in section {name:?}"))?;
        let end = u64::from_str_radix(end, 16)
            .with_context(|| format!("invalid end address in section {name:?}"))?;
        Ok((begin, end))
    })())
}

/// Read all splices from an rlib.
///
/// An rlib is an `ar` archive of relocatable object files. Each `#[Splice]`
/// function lives in its own `.rspl.<begin>.<end>` section, so we walk every
/// object member's sections, decode the range from the section name, and take
/// the section bytes as the replacement code.
pub fn read_splices_from_rlib<P: AsRef<Path>>(path: P) -> Result<Vec<Splice>> {
    use object::read::archive::ArchiveFile;
    use object::{Object, ObjectSection};

    let data = fs::read(&path)
        .with_context(|| format!("failed to read rlib {:?}", path.as_ref()))?;

    let archive = ArchiveFile::parse(&*data).context("failed to parse rlib archive")?;

    let mut splices = Vec::new();
    for member in archive.members() {
        let member = member.context("failed to read archive member")?;
        let member_data = member.data(&*data).context("failed to read member data")?;

        // Members may be non-object files (e.g. the rmeta blob); skip anything
        // that does not parse as an object file.
        let obj = match object::File::parse(member_data) {
            Ok(obj) => obj,
            Err(_) => continue,
        };

        for section in obj.sections() {
            let name = match section.name() {
                Ok(name) => name,
                Err(_) => continue,
            };
            let Some(parsed) = parse_splice_section(name) else {
                continue;
            };
            let (begin, end) = parsed?;
            let code = section
                .data()
                .with_context(|| format!("failed to read data of section {name:?}"))?
                .to_vec();
            splices.push(Splice { begin, end, code });
        }
    }

    Ok(splices)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to create a minimal valid ELF header
    fn create_minimal_elf() -> Vec<u8> {
        let mut data = vec![0; 64]; // Minimal ELF header is 64 bytes for 64-bit
        data[0] = 0x7f; // ELF magic
        data[1] = 0x45;
        data[2] = 0x4c;
        data[3] = 0x46;
        data[4] = 2; // 64-bit
        data[5] = 1; // Little endian
        data[6] = 1; // ELF version
        data[18] = 0x3e; // e_machine = EM_X86_64 (little endian)
        data
    }

    // Helper function to create a minimal PE header
    fn create_minimal_pe() -> Vec<u8> {
        let mut data = vec![0; 512];
        data[0] = 0x4d; // MZ magic
        data[1] = 0x5a;
        // PE signature offset at 0x3c
        data[0x3c] = 0x80;
        // PE signature at offset 0x80
        data[0x80] = b'P';
        data[0x81] = b'E';
        data[0x82] = 0;
        data[0x83] = 0;
        // COFF header machine field = IMAGE_FILE_MACHINE_AMD64 (0x8664)
        data[0x84] = 0x64;
        data[0x85] = 0x86;
        data
    }

    #[test]
    fn test_binary_format_detection_incomplete() {
        let elf_header = vec![0x7f, 0x45, 0x4c, 0x46]; // Too short
        assert!(Binary::detect_format(&elf_header).is_err());
    }

    #[test]
    fn test_binary_format_detection_elf() {
        let elf = create_minimal_elf();
        let format = Binary::detect_format(&elf).unwrap();
        assert_eq!(format, BinaryFormat::Elf);
    }

    #[test]
    fn test_binary_format_detection_pe() {
        let pe = create_minimal_pe();
        let format = Binary::detect_format(&pe).unwrap();
        assert_eq!(format, BinaryFormat::Pe);
    }

    #[test]
    fn test_binary_format_detection_invalid() {
        let invalid = vec![0; 100];
        assert!(Binary::detect_format(&invalid).is_err());
    }

    #[test]
    fn test_direct_patch_basic() {
        let mut binary = Binary {
            data: vec![0; 100],
            format: BinaryFormat::Elf,
            arch: Architecture::X86_64,
            is_64: true,
            endian: Endian::Little,
        };

        let code = vec![0x90, 0x90, 0x90];
        binary.apply_direct_patch(10, 20, &code).unwrap();

        assert_eq!(binary.data[10], 0x90);
        assert_eq!(binary.data[11], 0x90);
        assert_eq!(binary.data[12], 0x90);
    }

    #[test]
    fn test_direct_patch_with_nop_padding() {
        let mut binary = Binary {
            data: vec![0; 100],
            format: BinaryFormat::Elf,
            arch: Architecture::X86_64,
            is_64: true,
            endian: Endian::Little,
        };

        let code = vec![0xAA, 0xBB];
        binary.apply_direct_patch(10, 20, &code).unwrap();

        // Check code is written
        assert_eq!(binary.data[10], 0xAA);
        assert_eq!(binary.data[11], 0xBB);
        // Check NOPs are filled
        assert_eq!(binary.data[12], 0x90); // NOP for x86
        assert_eq!(binary.data[19], 0x90);
    }

    #[test]
    fn test_direct_patch_exact_fit() {
        let mut binary = Binary {
            data: vec![0; 100],
            format: BinaryFormat::Elf,
            arch: Architecture::X86_64,
            is_64: true,
            endian: Endian::Little,
        };

        let code = vec![0xAA; 10]; // Exactly 10 bytes
        binary.apply_direct_patch(10, 20, &code).unwrap();

        assert_eq!(binary.data[10], 0xAA);
        assert_eq!(binary.data[19], 0xAA);
        assert_eq!(binary.data[20], 0x00); // Untouched
    }

    #[test]
    fn test_direct_patch_code_too_large() {
        let mut binary = Binary {
            data: vec![0; 100],
            format: BinaryFormat::Elf,
            arch: Architecture::X86_64,
            is_64: true,
            endian: Endian::Little,
        };

        let code = vec![0x90; 15]; // Too large for range 10-20
        let result = binary.apply_direct_patch(10, 20, &code);

        assert!(result.is_err());
        // With anyhow, we just check that it's an error
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Invalid address range"));
    }

    #[test]
    fn test_direct_patch_out_of_bounds() {
        let mut binary = Binary {
            data: vec![0; 100],
            format: BinaryFormat::Elf,
            arch: Architecture::X86_64,
            is_64: true,
            endian: Endian::Little,
        };

        let code = vec![0x90; 5];
        let result = binary.apply_direct_patch(96, 110, &code);

        assert!(result.is_err());
    }

    #[test]
    fn test_direct_patch_at_start() {
        let mut binary = Binary {
            data: vec![0; 100],
            format: BinaryFormat::Elf,
            arch: Architecture::X86_64,
            is_64: true,
            endian: Endian::Little,
        };

        let code = vec![0xFF, 0xEE];
        binary.apply_direct_patch(0, 10, &code).unwrap();

        assert_eq!(binary.data[0], 0xFF);
        assert_eq!(binary.data[1], 0xEE);
    }

    #[test]
    fn test_direct_patch_at_end() {
        let mut binary = Binary {
            data: vec![0; 100],
            format: BinaryFormat::Elf,
            arch: Architecture::X86_64,
            is_64: true,
            endian: Endian::Little,
        };

        let code = vec![0xFF, 0xEE];
        binary.apply_direct_patch(98, 100, &code).unwrap();

        assert_eq!(binary.data[98], 0xFF);
        assert_eq!(binary.data[99], 0xEE);
    }

    #[test]
    fn test_jump_patch_forward() {
        let mut binary = Binary {
            data: vec![0; 1000],
            format: BinaryFormat::Elf,
            arch: Architecture::X86_64,
            is_64: true,
            endian: Endian::Little,
        };

        // Jump from 0x100 to 0x200
        binary.apply_jump_patch(0x100, 0x110, 0x200).unwrap();

        // Check JMP instruction (E9)
        assert_eq!(binary.data[0x100], 0xE9);

        // Calculate expected offset: target - (begin + 5)
        // 0x200 - (0x100 + 5) = 0xFB
        let offset = 0x200 - (0x100 + 5);
        assert_eq!(binary.data[0x101], (offset & 0xFF) as u8);
        assert_eq!(binary.data[0x102], ((offset >> 8) & 0xFF) as u8);
    }

    #[test]
    fn test_jump_patch_backward() {
        let mut binary = Binary {
            data: vec![0; 1000],
            format: BinaryFormat::Elf,
            arch: Architecture::X86_64,
            is_64: true,
            endian: Endian::Little,
        };

        // Jump from 0x200 to 0x100
        binary.apply_jump_patch(0x200, 0x210, 0x100).unwrap();

        // Check JMP instruction
        assert_eq!(binary.data[0x200], 0xE9);

        // Offset will be negative
        let offset = (0x100_i64 - (0x200 + 5) as i64) as i32;
        assert_eq!(binary.data[0x201], (offset & 0xFF) as u8);
    }

    #[test]
    fn test_jump_patch_out_of_bounds() {
        let mut binary = Binary {
            data: vec![0; 100],
            format: BinaryFormat::Elf,
            arch: Architecture::X86_64,
            is_64: true,
            endian: Endian::Little,
        };

        let result = binary.apply_jump_patch(98, 100, 0x200);
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_patches() {
        let mut binary = Binary {
            data: vec![0; 100],
            format: BinaryFormat::Elf,
            arch: Architecture::X86_64,
            is_64: true,
            endian: Endian::Little,
        };

        // Apply first patch
        binary.apply_direct_patch(10, 15, &[0xAA, 0xBB]).unwrap();
        // Apply second patch
        binary.apply_direct_patch(20, 25, &[0xCC, 0xDD]).unwrap();

        assert_eq!(binary.data[10], 0xAA);
        assert_eq!(binary.data[11], 0xBB);
        assert_eq!(binary.data[20], 0xCC);
        assert_eq!(binary.data[21], 0xDD);
    }

    #[test]
    fn test_patch_with_empty_code() {
        let mut binary = Binary {
            data: vec![0; 100],
            format: BinaryFormat::Elf,
            arch: Architecture::X86_64,
            is_64: true,
            endian: Endian::Little,
        };

        let code = vec![];
        binary.apply_direct_patch(10, 20, &code).unwrap();

        // Should fill with NOPs
        assert_eq!(binary.data[10], 0x90);
        assert_eq!(binary.data[19], 0x90);
    }

    #[test]
    fn test_large_jump_offset() {
        let mut binary = Binary {
            data: vec![0; 100000],
            format: BinaryFormat::Elf,
            arch: Architecture::X86_64,
            is_64: true,
            endian: Endian::Little,
        };

        // Large forward jump
        binary.apply_jump_patch(0x1000, 0x1010, 0x10000).unwrap();
        assert_eq!(binary.data[0x1000], 0xE9);
    }

    #[test]
    fn test_direct_patch_overlapping_ranges() {
        let mut binary = Binary {
            data: vec![0; 100],
            format: BinaryFormat::Elf,
            arch: Architecture::X86_64,
            is_64: true,
            endian: Endian::Little,
        };

        // First patch
        binary.apply_direct_patch(10, 20, &[0xAA; 5]).unwrap();
        // Overlapping patch - this should succeed and overwrite
        binary.apply_direct_patch(15, 25, &[0xBB; 5]).unwrap();

        assert_eq!(binary.data[14], 0xAA); // last code byte of first patch, untouched
        assert_eq!(binary.data[15], 0xBB); // Overwritten by second patch
        assert_eq!(binary.data[19], 0xBB);
    }

    #[test]
    fn test_parse_splice_section_valid() {
        let (begin, end) = parse_splice_section(".rspl.1670.1680").unwrap().unwrap();
        assert_eq!(begin, 0x1670);
        assert_eq!(end, 0x1680);
    }

    #[test]
    fn test_parse_splice_section_not_a_splice() {
        assert!(parse_splice_section(".text").is_none());
        assert!(parse_splice_section(".rodata").is_none());
    }

    #[test]
    fn test_parse_splice_section_malformed() {
        // Missing the second address.
        assert!(parse_splice_section(".rspl.1670").unwrap().is_err());
        // Non-hex address.
        assert!(parse_splice_section(".rspl.zz.10").unwrap().is_err());
    }

    // Wrap object bytes in a minimal GNU `ar` archive so we can exercise the
    // rlib reader without invoking a compiler.
    fn ar_wrap(member_name: &str, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"!<arch>\n");

        let mut header = format!("{:<16}", format!("{member_name}/"));
        header.push_str(&format!("{:<12}", 0)); // mtime
        header.push_str(&format!("{:<6}", 0)); // owner
        header.push_str(&format!("{:<6}", 0)); // group
        header.push_str(&format!("{:<8}", "644")); // mode
        header.push_str(&format!("{:<10}", data.len())); // size
        header.push_str("`\n");
        assert_eq!(header.len(), 60);

        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(data);
        if data.len() % 2 == 1 {
            out.push(b'\n');
        }
        out
    }

    /// Write a program header at `po` in the class-appropriate layout.
    #[allow(clippy::too_many_arguments)]
    fn write_phdr(
        d: &mut [u8],
        po: usize,
        is_64: bool,
        endian: Endian,
        p_type: u64,
        p_off: u64,
        p_vaddr: u64,
        p_filesz: u64,
        p_flags: u64,
    ) {
        endian.write_uint(d, po, 4, p_type);
        if is_64 {
            endian.write_uint(d, po + 4, 4, p_flags);
            endian.write_uint(d, po + 8, 8, p_off);
            endian.write_uint(d, po + 16, 8, p_vaddr);
            endian.write_uint(d, po + 24, 8, p_vaddr);
            endian.write_uint(d, po + 32, 8, p_filesz);
            endian.write_uint(d, po + 40, 8, p_filesz);
            endian.write_uint(d, po + 48, 8, 0x1000);
        } else {
            endian.write_uint(d, po + 4, 4, p_off);
            endian.write_uint(d, po + 8, 4, p_vaddr);
            endian.write_uint(d, po + 12, 4, p_vaddr);
            endian.write_uint(d, po + 16, 4, p_filesz);
            endian.write_uint(d, po + 20, 4, p_filesz);
            endian.write_uint(d, po + 24, 4, p_flags);
            endian.write_uint(d, po + 28, 4, 0x1000);
        }
    }

    /// A `Binary` whose bytes carry an ELF header of the given class/endianness
    /// plus a PT_LOAD and a PT_NOTE, ready for `inject_segment`.
    fn craft_injectable(is_64: bool, endian: Endian, arch: Architecture) -> Binary {
        let mut d = vec![0u8; 0x400];
        d[..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
        d[4] = if is_64 { 2 } else { 1 };
        d[5] = match endian {
            Endian::Little => 1,
            Endian::Big => 2,
        };
        d[6] = 1;
        let (phoff_off, phentsize_off, phnum_off, phentsize, phoff) = if is_64 {
            (0x20usize, 0x36, 0x38, 56usize, 0x40u64)
        } else {
            (0x1c, 0x2a, 0x2c, 32, 0x34)
        };
        let word = if is_64 { 8 } else { 4 };
        endian.write_uint(&mut d, phoff_off, word, phoff);
        endian.write_uint(&mut d, phentsize_off, 2, phentsize as u64);
        endian.write_uint(&mut d, phnum_off, 2, 2);
        let po = phoff as usize;
        write_phdr(&mut d, po, is_64, endian, 1, 0, 0x400000, 0x400, 5);
        write_phdr(&mut d, po + phentsize, is_64, endian, 4, 0x100, 0x400100, 0x20, 4);
        Binary {
            data: d,
            format: BinaryFormat::Elf,
            arch,
            is_64,
            endian,
        }
    }

    #[test]
    fn test_inject_segment_all_classes_and_endians() {
        let cases = [
            (true, Endian::Little, Architecture::X86_64),
            (true, Endian::Big, Architecture::Mips64),
            (false, Endian::Little, Architecture::Arm),
            (false, Endian::Big, Architecture::Mips),
        ];
        for (is_64, endian, arch) in cases {
            let mut bin = craft_injectable(is_64, endian, arch);
            let base = 0x600000u64;
            let payload = [0xde, 0xad, 0xbe, 0xef, 0x01, 0x02, 0x03];
            bin.inject_segment(base, &payload).unwrap();

            let (phoff_off, phentsize, phnum_off) = if is_64 {
                (0x20usize, 56usize, 0x38usize)
            } else {
                (0x1c, 32, 0x2c)
            };
            let word = if is_64 { 8 } else { 4 };
            let (voff, ooff, szoff, floff) = if is_64 {
                (16usize, 8usize, 32usize, 4usize)
            } else {
                (8, 4, 16, 24)
            };
            let phoff = endian.read_uint(bin.data(), phoff_off, word) as usize;
            let phnum = endian.read_uint(bin.data(), phnum_off, 2) as usize;

            let mut found = false;
            let mut has_note = false;
            for i in 0..phnum {
                let po = phoff + i * phentsize;
                let ptype = endian.read_uint(bin.data(), po, 4);
                if ptype == 4 {
                    has_note = true;
                }
                let pvaddr = endian.read_uint(bin.data(), po + voff, word);
                if ptype == 1 && pvaddr == base {
                    found = true;
                    let poff = endian.read_uint(bin.data(), po + ooff, word) as usize;
                    let filesz = endian.read_uint(bin.data(), po + szoff, word) as usize;
                    let flags = endian.read_uint(bin.data(), po + floff, 4);
                    assert_eq!(filesz, payload.len(), "{arch:?}");
                    assert_eq!(flags, 5, "R|X flags for {arch:?}");
                    assert_eq!(&bin.data()[poff..poff + payload.len()], &payload, "{arch:?}");
                }
            }
            assert!(found, "no injected PT_LOAD for {arch:?}");
            assert!(!has_note, "PT_NOTE should be consumed for {arch:?}");
        }
    }

    /// A `Binary` like `craft_injectable`, but with no PT_NOTE — the shape of a
    /// ELF whose linker never emitted one. There is deliberately a large gap between the end of the
    /// program-header table and the first PT_LOAD's file offset, mirroring
    /// the padding real toolchains leave before the first page-aligned
    /// segment.
    fn craft_injectable_no_note(is_64: bool, endian: Endian, arch: Architecture) -> Binary {
        let mut d = vec![0u8; 0x2000];
        d[..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
        d[4] = if is_64 { 2 } else { 1 };
        d[5] = match endian {
            Endian::Little => 1,
            Endian::Big => 2,
        };
        d[6] = 1;
        let (phoff_off, phentsize_off, phnum_off, phentsize, phoff) = if is_64 {
            (0x20usize, 0x36, 0x38, 56usize, 0x40u64)
        } else {
            (0x1c, 0x2a, 0x2c, 32, 0x34)
        };
        let word = if is_64 { 8 } else { 4 };
        endian.write_uint(&mut d, phoff_off, word, phoff);
        endian.write_uint(&mut d, phentsize_off, 2, phentsize as u64);
        endian.write_uint(&mut d, phnum_off, 2, 2);
        let po = phoff as usize;
        // Two PT_LOADs, first content starting well past the phdr table —
        // plenty of unclaimed padding to grow into.
        write_phdr(&mut d, po, is_64, endian, 1, 0x1000, 0x400000, 0x400, 5);
        write_phdr(&mut d, po + phentsize, is_64, endian, 1, 0x1400, 0x500000, 0x400, 6);
        Binary {
            data: d,
            format: BinaryFormat::Elf,
            arch,
            is_64,
            endian,
        }
    }

    #[test]
    fn test_inject_segment_grows_phdr_table_without_note() {
        // Regression test: targets with no PT_NOTE must still be injectable
        // by growing the program-header table into its own padding, rather
        // than failing outright.
        let cases = [
            (true, Endian::Little, Architecture::X86_64),
            (false, Endian::Big, Architecture::Mips),
        ];
        for (is_64, endian, arch) in cases {
            let mut bin = craft_injectable_no_note(is_64, endian, arch);
            let base = 0x600000u64;
            let payload = [0xde, 0xad, 0xbe, 0xef, 0x01, 0x02, 0x03];
            bin.inject_segment(base, &payload).unwrap();

            let (phoff_off, phentsize, phnum_off) = if is_64 {
                (0x20usize, 56usize, 0x38usize)
            } else {
                (0x1c, 32, 0x2c)
            };
            let word = if is_64 { 8 } else { 4 };
            let (voff, ooff, szoff, floff) = if is_64 {
                (16usize, 8usize, 32usize, 4usize)
            } else {
                (8, 4, 16, 24)
            };
            let phoff = endian.read_uint(bin.data(), phoff_off, word) as usize;
            let phnum = endian.read_uint(bin.data(), phnum_off, 2) as usize;
            assert_eq!(phnum, 3, "table should have grown by one entry for {arch:?}");

            let mut found = false;
            for i in 0..phnum {
                let po = phoff + i * phentsize;
                let ptype = endian.read_uint(bin.data(), po, 4);
                let pvaddr = endian.read_uint(bin.data(), po + voff, word);
                if ptype == 1 && pvaddr == base {
                    found = true;
                    let poff = endian.read_uint(bin.data(), po + ooff, word) as usize;
                    let filesz = endian.read_uint(bin.data(), po + szoff, word) as usize;
                    let flags = endian.read_uint(bin.data(), po + floff, 4);
                    assert_eq!(filesz, payload.len(), "{arch:?}");
                    assert_eq!(flags, 5, "R|X flags for {arch:?}");
                    assert_eq!(&bin.data()[poff..poff + payload.len()], &payload, "{arch:?}");
                }
            }
            assert!(found, "no injected PT_LOAD for {arch:?}");
        }
    }

    #[test]
    fn test_inject_segment_no_note_no_room_errors() {
        // When there's no PT_NOTE *and* no padding to grow into, injection
        // must fail with a clear error instead of corrupting the file.
        let endian = Endian::Little;
        let mut d = vec![0u8; 0x100];
        d[..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
        d[4] = 2;
        d[5] = 1;
        d[6] = 1;
        let (phoff_off, phentsize_off, phnum_off, phentsize, phoff) =
            (0x20usize, 0x36, 0x38, 56usize, 0x40u64);
        endian.write_uint(&mut d, phoff_off, 8, phoff);
        endian.write_uint(&mut d, phentsize_off, 2, phentsize as u64);
        endian.write_uint(&mut d, phnum_off, 2, 1);
        // A single PT_LOAD whose file offset sits immediately after the
        // table's one entry — zero padding available.
        let po = phoff as usize;
        write_phdr(&mut d, po, true, endian, 1, po as u64 + phentsize as u64, 0x400000, 0x40, 5);
        let mut bin = Binary {
            data: d,
            format: BinaryFormat::Elf,
            arch: Architecture::X86_64,
            is_64: true,
            endian,
        };
        let err = bin.inject_segment(0x600000, &[0xde, 0xad]).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("no PT_NOTE") && msg.contains("padding"),
            "{msg}"
        );
    }

    #[test]
    fn test_read_splices_from_rlib() {
        use object::write::{Object, SectionKind};
        use object::{Architecture, BinaryFormat, Endianness};

        let mut obj = Object::new(BinaryFormat::Elf, Architecture::X86_64, Endianness::Little);
        let code = [0xb8, 0x2a, 0x00, 0x00, 0x00, 0xc3]; // mov eax, 42; ret
        let sec = obj.add_section(Vec::new(), b".rspl.a.b".to_vec(), SectionKind::Text);
        obj.append_section_data(sec, &code, 1);
        let obj_bytes = obj.write().unwrap();

        let archive = ar_wrap("splice.o", &obj_bytes);
        let path = std::env::temp_dir().join(format!("resplice_rlib_test_{}.a", std::process::id()));
        fs::write(&path, &archive).unwrap();

        let splices = read_splices_from_rlib(&path).unwrap();
        fs::remove_file(&path).ok();

        assert_eq!(splices.len(), 1);
        assert_eq!(splices[0].begin, 0xa);
        assert_eq!(splices[0].end, 0xb);
        assert_eq!(splices[0].code, code);
    }
}
