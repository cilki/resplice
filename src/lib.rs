pub use anyhow::Result;

use anyhow::{anyhow, Context};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

mod arch;
mod link;
pub use arch::Endian;
pub use link::{link_rlib, Applied};

/// Fallback granularity for laying out the injected segment, used only when the
/// target declares nothing better. See [`Binary::load_align`].
const PAGE: u64 = 0x1000;

/// Round `x` up to the next multiple of `align` (a power of two), saturating at
/// the largest representable multiple rather than wrapping back to zero.
fn align_up(x: u64, align: u64) -> u64 {
    x.saturating_add(align - 1) & !(align - 1)
}

/// Represents an ELF binary that can be patched
pub struct Binary {
    data: Vec<u8>,
    arch: Architecture,
    endian: Endian,
    /// Operator-supplied virtual address for the injected segment, overriding
    /// [`Binary::injected_base`]'s default. See [`Binary::set_injected_base`].
    inject_base: Option<u64>,
    /// Every name the target can bind an external reference to, built on first
    /// use by [`Binary::build_symbol_map`] and dropped whenever `data` changes.
    ///
    /// A splice typically calls several functions the target already defines, so
    /// this is consulted once per relocation. Scanning `.symtab`/`.dynsym` for
    /// each of those is quadratic in a real target's symbol count, which is
    /// where this tool spent nearly all of its time before the map existed.
    symbols: RefCell<Option<HashMap<String, u64>>>,
}

/// One entry of the target's program-header table.
#[derive(Debug)]
struct Phdr {
    /// File offset of this entry, for rewriting it in place.
    entry_off: usize,
    p_type: u32,
    p_offset: u64,
    p_vaddr: u64,
    p_filesz: u64,
    p_memsz: u64,
    p_align: u64,
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
        let (arch, endian) = Self::detect_arch(&data)?;
        Ok(Self::new(data, arch, endian))
    }

    /// Wrap an image already held in memory.
    fn new(data: Vec<u8>, arch: Architecture, endian: Endian) -> Self {
        Binary {
            data,
            arch,
            endian,
            inject_base: None,
            symbols: RefCell::new(None),
        }
    }

    /// Drop everything derived from `data`, after `data` has been changed.
    fn invalidate_caches(&mut self) {
        *self.symbols.get_mut() = None;
    }

    /// Pin the injected segment to `base` instead of the default.
    ///
    /// The default places the segment one page past the end of the image, which
    /// is only safe when nothing else claims that address. It often is not: a
    /// game console binary typically has its allocator hand out memory starting
    /// at the end of `.bss`, so the default lands one page into the heap and the
    /// injected code and data are live only until the first big allocation.
    /// There is no portable way to detect that, so the operator supplies a known
    /// free address instead. `base` must be page-aligned.
    pub fn set_injected_base(&mut self, base: Option<u64>) {
        self.inject_base = base;
    }

    /// Detect the architecture and byte order of an ELF target.
    ///
    /// Every operation this crate performs on a target — translating virtual
    /// addresses, reading its symbol table, injecting a segment — is ELF-only,
    /// so non-ELF inputs are rejected here rather than later.
    fn detect_arch(data: &[u8]) -> Result<(Architecture, Endian)> {
        let elf = goblin::elf::Elf::parse(data).context("target is not an ELF binary")?;
        let arch = match elf.header.e_machine {
            goblin::elf::header::EM_386 => Architecture::X86,
            goblin::elf::header::EM_X86_64 => Architecture::X86_64,
            goblin::elf::header::EM_ARM => Architecture::Arm,
            goblin::elf::header::EM_AARCH64 => Architecture::Arm64,
            goblin::elf::header::EM_MIPS if elf.is_64 => Architecture::Mips64,
            goblin::elf::header::EM_MIPS => Architecture::Mips,
            other => return Err(anyhow!("unsupported ELF machine type {other}")),
        };
        // The machine type implies the pointer width everywhere except MIPS,
        // which is resolved above; a header that disagrees with itself would
        // make every subsequent header field be read at the wrong width.
        if arch.is_64() != elf.is_64 {
            return Err(anyhow!(
                "ELF class contradicts machine type {}",
                elf.header.e_machine
            ));
        }
        let endian = if elf.little_endian {
            Endian::Little
        } else {
            Endian::Big
        };
        Ok((arch, endian))
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

        // Checked, not `offset + region_len`: both come from addresses in a
        // splice section's name, and a wrapping sum here would pass for in
        // bounds and leave the NOP fill below running off the end of the buffer.
        let region_end = offset
            .checked_add(region_len)
            .filter(|&end| end <= self.data.len())
            .ok_or_else(|| {
                anyhow!("patch region {offset:#x}..+{region_len:#x} is outside the binary")
            })?;

        let end_pos = offset + code.len();
        self.data[offset..end_pos].copy_from_slice(code);

        // If the new code is smaller than the region, fill the rest with NOPs.
        if end_pos < region_end {
            let nop_insn = self.get_nop_instruction();
            let mut pos = end_pos;
            while pos < region_end {
                let n = std::cmp::min(nop_insn.len(), region_end - pos);
                self.data[pos..pos + n].copy_from_slice(&nop_insn[..n]);
                pos += n;
            }
        }

        self.invalidate_caches();
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

    /// Machine code for an unconditional jump from `from` to `to`, for the
    /// current architecture. Used to build trampolines for oversized splices.
    pub fn jump_bytes(&self, from: u64, to: u64) -> Result<Vec<u8>> {
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
                //
                // `j` has a branch delay slot: the instruction *after* it
                // executes before the jump takes effect. A bare `j` would
                // therefore leave whatever the replaced function had at
                // `from + 4` to run as the delay slot, so the trampoline must
                // carry its own `nop`.
                //
                // `j` also only encodes the low 28 bits of the target; the top
                // four come from the delay slot's own PC, so the two must share
                // a 256MB region. Truncating silently would jump into hyperspace.
                if to & 3 != 0 {
                    return Err(anyhow!("mips jump target {to:#x} is not 4-byte aligned"));
                }
                let delay_pc = from + 4;
                if (delay_pc & !0x0fff_ffff) != (to & !0x0fff_ffff) {
                    return Err(anyhow!(
                        "mips jump from {from:#x} to {to:#x} crosses a 256MB region \
                         boundary and cannot be encoded as `j`"
                    ));
                }
                let addr_bits = ((to >> 2) & 0x03FFFFFF) as u32;
                let mut out = self.encode_insn(0x08000000u32 | addr_bits);
                out.extend_from_slice(&self.encode_insn(0)); // delay slot: nop
                Ok(out)
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

    /// Get the architecture
    pub fn architecture(&self) -> Architecture {
        self.arch
    }

    /// Whether the target uses 64-bit pointers.
    pub fn is_64(&self) -> bool {
        self.arch.is_64()
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

    /// Read the program-header table straight out of the image.
    ///
    /// This is deliberately not a full [`elf`](Self::elf) parse: the table is a
    /// dozen fixed-layout entries, while parsing the whole image walks its
    /// section headers, symbol tables and string tables. Address translation
    /// needs nothing but this table and is called once per splice, so paying
    /// for the rest of the parse each time dominated the tool's runtime.
    ///
    /// Reading the bytes directly also keeps the result current: the table is
    /// rewritten by [`inject_segment`](Self::inject_segment), and every caller
    /// here must see that edit.
    fn program_headers(&self) -> Result<Vec<Phdr>> {
        let endian = self.endian;
        let is_64 = self.is_64();
        // Header fields holding the table's location, and the entry layout of
        // this ELF class. ELF32 has no `p_flags` between `p_type` and
        // `p_offset`, so every field after the first sits elsewhere.
        let (phoff_off, phentsize_off, phnum_off, phentsize, ehsize) = if is_64 {
            (0x20, 0x36, 0x38, 56usize, 64usize)
        } else {
            (0x1c, 0x2a, 0x2c, 32usize, 52usize)
        };
        let (o_offset, o_vaddr, o_filesz, o_memsz, o_align) = if is_64 {
            (8, 16, 32, 40, 48)
        } else {
            (4, 8, 16, 20, 28)
        };
        let word = if is_64 { 8 } else { 4 };

        if self.data.len() < ehsize {
            return Err(anyhow!(
                "image is {} bytes, too short for an ELF header",
                self.data.len()
            ));
        }
        let phoff = endian.read_uint(&self.data, phoff_off, word) as usize;
        let got_phentsize = endian.read_uint(&self.data, phentsize_off, 2) as usize;
        let phnum = endian.read_uint(&self.data, phnum_off, 2) as usize;
        if got_phentsize != phentsize {
            return Err(anyhow!("unexpected program-header entry size {got_phentsize}"));
        }
        // Checked, so a table the header claims lies past the end of the file is
        // refused here instead of panicking on the reads below. (`phnum` is a
        // 16-bit field and `phentsize` is fixed, so the product cannot wrap.)
        if phoff
            .checked_add(phnum * phentsize)
            .is_none_or(|end| end > self.data.len())
        {
            return Err(anyhow!(
                "program-header table at {phoff:#x} ({phnum} entries of {phentsize} \
                 bytes) runs past the end of the {}-byte image",
                self.data.len()
            ));
        }

        Ok((0..phnum)
            .map(|i| {
                let off = phoff + i * phentsize;
                Phdr {
                    entry_off: off,
                    p_type: endian.read_uint(&self.data, off, 4) as u32,
                    p_offset: endian.read_uint(&self.data, off + o_offset, word),
                    p_vaddr: endian.read_uint(&self.data, off + o_vaddr, word),
                    p_filesz: endian.read_uint(&self.data, off + o_filesz, word),
                    p_memsz: endian.read_uint(&self.data, off + o_memsz, word),
                    p_align: endian.read_uint(&self.data, off + o_align, word),
                }
            })
            .collect())
    }

    /// Translate a virtual address to a file offset using the program headers.
    pub fn va_to_offset(&self, va: u64) -> Result<usize> {
        use goblin::elf::program_header::PT_LOAD;
        for ph in self.program_headers()? {
            if ph.p_type == PT_LOAD && va >= ph.p_vaddr && va < ph.p_vaddr + ph.p_filesz {
                return Ok((ph.p_offset + (va - ph.p_vaddr)) as usize);
            }
        }
        Err(anyhow!("virtual address {va:#x} is not mapped by any PT_LOAD segment"))
    }

    /// The highest virtual address occupied by any loadable segment.
    fn max_vaddr(&self) -> Result<u64> {
        use goblin::elf::program_header::PT_LOAD;
        Ok(self
            .program_headers()?
            .iter()
            .filter(|ph| ph.p_type == PT_LOAD)
            .map(|ph| ph.p_vaddr + ph.p_memsz)
            .max()
            .unwrap_or(0))
    }

    /// The granularity the injected segment must be laid out at: the largest
    /// `p_align` of any `PT_LOAD` in the target (never below [`PAGE`]).
    ///
    /// This is the target's *maximum page size* — the granularity the linker
    /// aligned its segments to so the image loads under any page size the
    /// architecture permits. It matters for two reasons, and assuming 4K for
    /// both is wrong on every architecture with larger pages (aarch64 images are
    /// linked for 64K, and real aarch64 kernels run 16K or 64K pages):
    ///
    /// - The kernel maps a `PT_LOAD` from `p_offset` rounded *down* to a page
    ///   boundary, so a segment placed less than a page past the image shares a
    ///   page with it and silently replaces that mapping. Landing on `.got` that
    ///   way crashes the dynamic loader before `main` even runs.
    /// - `p_vaddr ≡ p_offset (mod page size)` must hold, or the loader rejects
    ///   the image outright.
    fn load_align(&self) -> Result<u64> {
        use goblin::elf::program_header::PT_LOAD;
        let declared = self
            .program_headers()?
            .iter()
            .filter(|ph| ph.p_type == PT_LOAD && ph.p_align.is_power_of_two())
            .map(|ph| ph.p_align)
            .max()
            .unwrap_or(PAGE);
        Ok(declared.max(PAGE))
    }

    /// The base virtual address the injected segment will be mapped at: one
    /// [`load_align`](Self::load_align) granule above the target's image.
    pub fn injected_base(&self) -> Result<u64> {
        let align = self.load_align()?;
        if let Some(base) = self.inject_base {
            if base % align != 0 {
                return Err(anyhow!(
                    "injected base {base:#x} is not aligned to the target's page \
                     size ({align:#x})"
                ));
            }
            return Ok(base);
        }
        Ok(align_up(self.max_vaddr()?, align) + align)
    }

    /// Resolve a symbol name against the target binary's own symbols.
    ///
    /// Tries, in order: a defined symbol in `.symtab`, a defined symbol in
    /// `.dynsym`, then a function the target imports through its PLT (the stub's
    /// address). Returns `None` if the target provides no such symbol.
    pub fn resolve_target_symbol(&self, name: &str) -> Result<Option<u64>> {
        let mut cached = self.symbols.borrow_mut();
        if cached.is_none() {
            *cached = Some(self.build_symbol_map()?);
        }
        Ok(cached.as_ref().and_then(|map| map.get(name).copied()))
    }

    /// Collect every name [`resolve_target_symbol`](Self::resolve_target_symbol)
    /// can bind, with the address it resolves to.
    ///
    /// Earlier insertions win, which is what gives that function its documented
    /// precedence: `.symtab` over `.dynsym` over the PLT, and within one table
    /// the first matching entry.
    fn build_symbol_map(&self) -> Result<HashMap<String, u64>> {
        use goblin::elf::section_header::SHN_UNDEF;
        let elf = self.elf()?;
        let mut map: HashMap<String, u64> = HashMap::new();

        // 1 & 2: symbols the target defines itself.
        for (syms, strtab) in [(&elf.syms, &elf.strtab), (&elf.dynsyms, &elf.dynstrtab)] {
            for sym in syms.iter() {
                if sym.st_shndx as u32 == SHN_UNDEF || sym.st_value == 0 {
                    continue;
                }
                if let Some(name) = strtab.get_at(sym.st_name) {
                    map.entry(name.to_owned()).or_insert(sym.st_value);
                }
            }
        }

        // 3: functions the target imports through the PLT. The i-th `.rela.plt`
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
                    if let Some(name) = elf.dynstrtab.get_at(sym.st_name) {
                        map.entry(name.to_owned())
                            .or_insert(plt_addr + plt_header + i as u64 * plt_entry);
                    }
                }
            }
        }

        Ok(map)
    }

    /// Verify that `[base, base + len)` does not collide with the target's own
    /// image, so the injected segment adds a mapping instead of replacing one.
    ///
    /// Alignment used to be the only thing checked about an operator-supplied
    /// base, which let an address that simply is not free through: a mistyped
    /// `--inject-base` (the image's own base, or an address a digit short of
    /// the intended one) produced a `PT_LOAD` mapping the injected blob over
    /// the target's code. `resplice` reported success, and the result either
    /// crashed or was refused by the loader.
    ///
    /// Pages *shared* with an existing `PT_LOAD` count as a collision: the
    /// kernel maps whole pages, and the later mapping replaces the earlier one
    /// rather than being merged into it. Both ranges are therefore compared
    /// after rounding out to `align`, the target's page size.
    fn check_base_is_free(&self, base: u64, len: u64, align: u64) -> Result<()> {
        use goblin::elf::program_header::PT_LOAD;

        let end = base.checked_add(len).ok_or_else(|| {
            anyhow!("injected segment at {base:#x} runs past the end of the address space")
        })?;
        let claimed_end = align_up(end, align);

        for ph in self.program_headers()? {
            if ph.p_type != PT_LOAD {
                continue;
            }
            let seg_end = ph.p_vaddr.saturating_add(ph.p_memsz);
            let (lo, hi) = (ph.p_vaddr & !(align - 1), align_up(seg_end, align));
            if base < hi && lo < claimed_end {
                return Err(anyhow!(
                    "injected segment {base:#x}..{end:#x} would be mapped over the \
                     target's own PT_LOAD at {:#x}..{seg_end:#x} (both round out to \
                     pages of {align:#x}, so {base:#x}..{claimed_end:#x} overlaps \
                     {lo:#x}..{hi:#x}); choose an --inject-base that is free in the \
                     target's address space",
                    ph.p_vaddr
                ));
            }
        }
        Ok(())
    }

    /// Inject `blob` as a new read+execute segment mapped at virtual address
    /// `base`, by converting an existing `PT_NOTE` program header into a
    /// `PT_LOAD` (so the program-header table itself need not be relocated).
    ///
    /// `base` must be aligned to the target's [`load_align`](Self::load_align);
    /// the file is padded to the same granularity before the blob is appended so
    /// the loader's `p_vaddr ≡ p_offset (mod p_align)` requirement holds.
    pub fn inject_segment(&mut self, base: u64, blob: &[u8]) -> Result<()> {
        use goblin::elf::program_header::{PF_R, PF_X, PT_LOAD, PT_NOTE};

        let align = self.load_align()?;
        if base % align != 0 {
            return Err(anyhow!(
                "injected base {base:#x} is not aligned to the target's page size \
                 ({align:#x})"
            ));
        }
        self.check_base_is_free(base, blob.len() as u64, align)?;
        let endian = self.endian;
        let is_64 = self.is_64();

        // Where the program-header table lives and how big its entries are, at
        // class-dependent offsets in the ELF header. The no-PT_NOTE path below
        // appends an entry to the table and bumps the count.
        let (phoff_off, phnum_off, phentsize) = if is_64 {
            (0x20, 0x38, 56usize)
        } else {
            (0x1c, 0x2c, 32usize)
        };
        let word = if is_64 { 8 } else { 4 };
        let phoff = endian.read_uint(&self.data, phoff_off, word) as usize;
        let phdrs = self.program_headers()?;
        let phnum = phdrs.len();

        // Find a PT_NOTE entry to repurpose.
        let note_off = phdrs
            .iter()
            .find(|ph| ph.p_type == PT_NOTE)
            .map(|ph| ph.entry_off);

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
                let mut first_claimed = phdrs
                    .iter()
                    .map(|ph| ph.p_offset)
                    .filter(|&off| off > 0)
                    .min()
                    .unwrap_or(u64::MAX);
                // Only the start of the section-header table matters here;
                // its entry size and count are irrelevant to the padding gap.
                let (eshoff_off, eshoff_w) = if is_64 { (0x28, 8) } else { (0x20, 4) };
                let e_shoff = endian.read_uint(&self.data, eshoff_off, eshoff_w);
                if e_shoff > 0 {
                    first_claimed = first_claimed.min(e_shoff);
                }

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

        // Append the blob at a file offset congruent to `base` modulo `align`.
        let file_off = align_up(self.data.len() as u64, align);
        self.data.resize(file_off as usize, 0);
        self.data.extend_from_slice(blob);
        let len = blob.len() as u64;

        // Rewrite the note phdr in place as a PT_LOAD covering the appended blob.
        // The 32- and 64-bit program-header layouts differ in field order and
        // width (notably `p_flags` moves from offset 4 to offset 24 in ELF32).
        let flags = (PF_R | PF_X) as u64;
        let d = &mut self.data;
        endian.write_uint(d, note_off, 4, PT_LOAD as u64);
        if is_64 {
            endian.write_uint(d, note_off + 4, 4, flags);
            endian.write_uint(d, note_off + 8, 8, file_off);
            endian.write_uint(d, note_off + 16, 8, base);
            endian.write_uint(d, note_off + 24, 8, base);
            endian.write_uint(d, note_off + 32, 8, len);
            endian.write_uint(d, note_off + 40, 8, len);
            endian.write_uint(d, note_off + 48, 8, align);
        } else {
            endian.write_uint(d, note_off + 4, 4, file_off);
            endian.write_uint(d, note_off + 8, 4, base);
            endian.write_uint(d, note_off + 12, 4, base);
            endian.write_uint(d, note_off + 16, 4, len);
            endian.write_uint(d, note_off + 20, 4, len);
            endian.write_uint(d, note_off + 24, 4, flags);
            endian.write_uint(d, note_off + 28, 4, align);
        }

        self.invalidate_caches();
        Ok(())
    }
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
    fn detect_arch_reads_machine_and_byte_order() {
        let (arch, endian) = Binary::detect_arch(&create_minimal_elf()).unwrap();
        assert_eq!(arch, Architecture::X86_64);
        assert_eq!(endian, Endian::Little);
    }

    /// Everything downstream (address translation, symbol lookup, segment
    /// injection) is ELF-only, so anything else must be refused at load time.
    #[test]
    fn detect_arch_rejects_non_elf() {
        for data in [create_minimal_pe(), vec![0x7f, 0x45, 0x4c, 0x46], vec![0; 100]] {
            assert!(Binary::detect_arch(&data).is_err());
        }
    }

    /// The machine type drives the pointer width used to read every later header
    /// field, so a header whose ELF class contradicts it must not be accepted.
    #[test]
    fn detect_arch_rejects_class_machine_mismatch() {
        let mut data = create_minimal_elf();
        data[4] = 1; // ELFCLASS32 alongside e_machine = EM_X86_64
        let err = Binary::detect_arch(&data).unwrap_err();
        assert!(err.to_string().contains("contradicts"), "{err}");
    }

    #[test]
    fn test_patch_bytes_writes_code_and_nop_fills_the_rest() {
        let mut binary = Binary::new(vec![0; 100], Architecture::X86_64, Endian::Little);

        binary.patch_bytes(10, 10, &[0xAA, 0xBB]).unwrap();

        assert_eq!(binary.data[10], 0xAA);
        assert_eq!(binary.data[11], 0xBB);
        // The remainder of the region is filled with x86 NOPs.
        assert_eq!(binary.data[12], 0x90);
        assert_eq!(binary.data[19], 0x90);
        assert_eq!(binary.data[20], 0x00); // just past the region, untouched
    }

    #[test]
    fn test_patch_bytes_exact_fit_leaves_no_padding() {
        let mut binary = Binary::new(vec![0; 100], Architecture::X86_64, Endian::Little);

        binary.patch_bytes(10, 10, &[0xAA; 10]).unwrap();

        assert_eq!(binary.data[10], 0xAA);
        assert_eq!(binary.data[19], 0xAA);
        assert_eq!(binary.data[20], 0x00); // untouched
    }

    #[test]
    fn test_patch_bytes_empty_code_fills_region_with_nops() {
        let mut binary = Binary::new(vec![0; 100], Architecture::X86_64, Endian::Little);

        binary.patch_bytes(10, 10, &[]).unwrap();

        assert_eq!(binary.data[10], 0x90);
        assert_eq!(binary.data[19], 0x90);
    }

    #[test]
    fn test_patch_bytes_code_too_large_errors() {
        let mut binary = Binary::new(vec![0; 100], Architecture::X86_64, Endian::Little);

        let err = binary.patch_bytes(10, 10, &[0x90; 15]).unwrap_err();
        assert!(err.to_string().contains("larger than region"));
    }

    #[test]
    fn test_patch_bytes_region_out_of_bounds_errors() {
        let mut binary = Binary::new(vec![0; 100], Architecture::X86_64, Endian::Little);

        assert!(binary.patch_bytes(96, 14, &[0x90; 5]).is_err());
    }

    /// `region_len` comes from the addresses in a splice section's name. A
    /// region so long that `offset + region_len` wraps must be refused rather
    /// than wrapping into something that looks in bounds, which would leave the
    /// NOP fill writing past the end of the image.
    #[test]
    fn test_patch_bytes_region_length_cannot_wrap_past_the_bounds_check() {
        let mut binary = Binary::new(vec![0; 100], Architecture::X86_64, Endian::Little);

        let err = binary.patch_bytes(10, usize::MAX - 5, &[0xAA]).unwrap_err();
        assert!(err.to_string().contains("outside the binary"));
        assert!(binary.data.iter().all(|&b| b == 0), "binary was modified");
    }

    #[test]
    fn test_jump_bytes_x86_encodes_relative_offset() {
        let binary = Binary::new(vec![0; 1000], Architecture::X86_64, Endian::Little);

        // Forward jump: E9 followed by rel32 = target - (from + 5).
        let jump = binary.jump_bytes(0x100, 0x200).unwrap();
        assert_eq!(jump[0], 0xE9);
        let fwd = (0x200_i64 - (0x100 + 5)) as i32;
        assert_eq!(&jump[1..5], &fwd.to_le_bytes());

        // Backward jump encodes a negative rel32.
        let jump = binary.jump_bytes(0x200, 0x100).unwrap();
        let back = (0x100_i64 - (0x200 + 5)) as i32;
        assert_eq!(&jump[1..5], &back.to_le_bytes());
    }

    /// The injected base must be overridable: the default (one page past the
    /// image) lands in the heap on a console binary that allocates from the end
    /// of `.bss`, which silently corrupts the injected code during play.
    #[test]
    fn injected_base_can_be_pinned() {
        use goblin::elf::program_header::PT_LOAD;
        let phdrs: [Phdr; 1] = [(PT_LOAD as u64, 0, 0x400000, 0x400, 5, 0x1000)];
        let mut binary = craft_injectable(Endian::Little, Architecture::Mips, 0x400, &phdrs);

        binary.set_injected_base(Some(0x001c_0000));
        assert_eq!(binary.injected_base().unwrap(), 0x001c_0000);

        binary.set_injected_base(Some(0x001c_0001));
        assert!(binary.injected_base().unwrap_err().to_string().contains("aligned"));
    }

    /// A MIPS trampoline must carry its own delay-slot `nop`: `j` executes the
    /// following instruction before branching, and at a splice site that
    /// instruction is leftover code from the function being replaced.
    #[test]
    fn mips_trampoline_fills_its_delay_slot() {
        let binary = Binary::new(vec![0; 0x1000], Architecture::Mips, Endian::Little);

        let jump = binary.jump_bytes(0x0022_9fd8, 0x0034_0000).unwrap();
        assert_eq!(jump.len(), 8, "trampoline must be j + nop");
        assert_eq!(&jump[..4], &(0x0800_0000u32 | (0x0034_0000 >> 2)).to_le_bytes());
        assert_eq!(&jump[4..], &0u32.to_le_bytes(), "delay slot must be a nop");
    }

    /// `j` only encodes the low 28 bits, so a target in another 256MB region is
    /// unencodable and must be reported rather than silently truncated.
    #[test]
    fn mips_trampoline_rejects_region_crossing_jump() {
        let binary = Binary::new(vec![0; 0x1000], Architecture::Mips, Endian::Little);

        let err = binary.jump_bytes(0x0022_9fd8, 0x1234_5678).unwrap_err();
        assert!(err.to_string().contains("256MB"), "got: {err}");
        assert!(binary.jump_bytes(0x0022_9fd8, 0x0034_0002).is_err());
    }

    /// One program header, as
    /// `(p_type, p_offset, p_vaddr, p_filesz, p_flags, p_align)`.
    type Phdr = (u64, u64, u64, u64, u64, u64);

    /// Offsets and widths of the header fields the injection tests read and
    /// write, for the given ELF class.
    struct Layout {
        phoff: u64,
        phoff_off: usize,
        phentsize_off: usize,
        phnum_off: usize,
        phentsize: usize,
        word: usize,
        /// Program-header field offsets. The field order differs between the
        /// two classes (notably `p_flags`, which moves from 4 to 24 in ELF32).
        p_offset: usize,
        p_vaddr: usize,
        p_filesz: usize,
        p_flags: usize,
        p_align: usize,
    }

    fn layout(is_64: bool) -> Layout {
        if is_64 {
            Layout {
                phoff: 0x40,
                phoff_off: 0x20,
                phentsize_off: 0x36,
                phnum_off: 0x38,
                phentsize: 56,
                word: 8,
                p_offset: 8,
                p_vaddr: 16,
                p_filesz: 32,
                p_flags: 4,
                p_align: 48,
            }
        } else {
            Layout {
                phoff: 0x34,
                phoff_off: 0x1c,
                phentsize_off: 0x2a,
                phnum_off: 0x2c,
                phentsize: 32,
                word: 4,
                p_offset: 4,
                p_vaddr: 8,
                p_filesz: 16,
                p_flags: 24,
                p_align: 28,
            }
        }
    }

    /// A `Binary` of `len` bytes carrying an ELF header of the class implied by
    /// `arch` plus `phdrs`, ready for `inject_segment`.
    fn craft_injectable(endian: Endian, arch: Architecture, len: usize, phdrs: &[Phdr]) -> Binary {
        let is_64 = arch.is_64();
        let l = layout(is_64);
        let mut d = vec![0u8; len];
        d[..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
        d[4] = if is_64 { 2 } else { 1 };
        d[5] = match endian {
            Endian::Little => 1,
            Endian::Big => 2,
        };
        d[6] = 1;
        endian.write_uint(&mut d, l.phoff_off, l.word, l.phoff);
        endian.write_uint(&mut d, l.phentsize_off, 2, l.phentsize as u64);
        endian.write_uint(&mut d, l.phnum_off, 2, phdrs.len() as u64);
        for (i, &(p_type, p_off, p_vaddr, p_filesz, p_flags, p_align)) in phdrs.iter().enumerate() {
            let po = l.phoff as usize + i * l.phentsize;
            endian.write_uint(&mut d, po, 4, p_type);
            endian.write_uint(&mut d, po + l.p_offset, l.word, p_off);
            endian.write_uint(&mut d, po + l.p_vaddr, l.word, p_vaddr);
            endian.write_uint(&mut d, po + l.p_vaddr + l.word, l.word, p_vaddr); // p_paddr
            endian.write_uint(&mut d, po + l.p_filesz, l.word, p_filesz);
            endian.write_uint(&mut d, po + l.p_filesz + l.word, l.word, p_filesz); // p_memsz
            endian.write_uint(&mut d, po + l.p_flags, 4, p_flags);
            endian.write_uint(&mut d, po + l.p_align, l.word, p_align);
        }
        Binary::new(d, arch, endian)
    }

    /// The file offset of every program header in `bin`.
    fn phdr_offsets(bin: &Binary) -> Vec<usize> {
        let l = layout(bin.is_64());
        let phoff = bin.endian().read_uint(bin.data(), l.phoff_off, l.word) as usize;
        let phnum = bin.endian().read_uint(bin.data(), l.phnum_off, 2) as usize;
        (0..phnum).map(|i| phoff + i * l.phentsize).collect()
    }

    /// The `p_type` of every program header in `bin`.
    fn phdr_types(bin: &Binary) -> Vec<u64> {
        phdr_offsets(bin)
            .into_iter()
            .map(|po| bin.endian().read_uint(bin.data(), po, 4))
            .collect()
    }

    /// Assert that exactly one R+X `PT_LOAD` maps `payload` at `base`, and
    /// return the file offset of its program header.
    fn injected_phdr(bin: &Binary, base: u64, payload: &[u8]) -> usize {
        use goblin::elf::program_header::{PF_R, PF_X, PT_LOAD};
        let endian = bin.endian();
        let l = layout(bin.is_64());

        let mut found = None;
        for po in phdr_offsets(bin) {
            if endian.read_uint(bin.data(), po, 4) != PT_LOAD as u64
                || endian.read_uint(bin.data(), po + l.p_vaddr, l.word) != base
            {
                continue;
            }
            assert!(found.is_none(), "more than one PT_LOAD at {base:#x}");
            found = Some(po);
            let p_off = endian.read_uint(bin.data(), po + l.p_offset, l.word) as usize;
            assert_eq!(
                endian.read_uint(bin.data(), po + l.p_filesz, l.word),
                payload.len() as u64
            );
            assert_eq!(
                endian.read_uint(bin.data(), po + l.p_flags, 4),
                (PF_R | PF_X) as u64
            );
            assert_eq!(&bin.data()[p_off..p_off + payload.len()], payload);
        }
        found.unwrap_or_else(|| panic!("expected an injected PT_LOAD at {base:#x}"))
    }

    const PAYLOAD: [u8; 7] = [0xde, 0xad, 0xbe, 0xef, 0x01, 0x02, 0x03];
    const INJECT_AT: u64 = 0x600000;

    #[test]
    fn test_inject_segment_all_classes_and_endians() {
        use goblin::elf::program_header::{PT_LOAD, PT_NOTE};
        let cases = [
            (Endian::Little, Architecture::X86_64),
            (Endian::Big, Architecture::Mips64),
            (Endian::Little, Architecture::Arm),
            (Endian::Big, Architecture::Mips),
        ];
        for (endian, arch) in cases {
            // One PT_LOAD plus a PT_NOTE for `inject_segment` to convert.
            let phdrs: [Phdr; 2] = [
                (PT_LOAD as u64, 0, 0x400000, 0x400, 5, 0x1000),
                (PT_NOTE as u64, 0x100, 0x400100, 0x20, 4, 4),
            ];
            let mut bin = craft_injectable(endian, arch, 0x400, &phdrs);
            bin.inject_segment(INJECT_AT, &PAYLOAD).unwrap();

            injected_phdr(&bin, INJECT_AT, &PAYLOAD);
            let types = phdr_types(&bin);
            assert_eq!(types.len(), 2, "table should not have grown for {arch:?}");
            assert!(
                !types.contains(&(PT_NOTE as u64)),
                "PT_NOTE should be consumed for {arch:?}"
            );
        }
    }

    /// The injected segment must be laid out at the target's *own* maximum page
    /// size, not an assumed 4K. A target linked for 64K pages (every aarch64
    /// image is) whose segment is only 4K-clear of the image shares a page with
    /// it under a 16K or 64K kernel, and the kernel's mapping of the injected
    /// segment silently replaces the image's — typically clobbering `.got`, so
    /// the dynamic loader segfaults before `main` runs.
    #[test]
    fn injected_segment_uses_the_targets_page_size() {
        use goblin::elf::program_header::{PT_LOAD, PT_NOTE};
        let align = 0x10000u64;
        let phdrs: [Phdr; 2] = [
            (PT_LOAD as u64, 0, 0x400000, 0x400, 5, align),
            (PT_NOTE as u64, 0x100, 0x400100, 0x20, 4, 4),
        ];
        let mut bin = craft_injectable(Endian::Little, Architecture::Arm64, 0x400, &phdrs);

        // The default base clears the image by a full granule of the target's
        // page size, so no page can be shared with it.
        let base = bin.injected_base().unwrap();
        assert_eq!(base % align, 0, "base {base:#x} not aligned to {align:#x}");
        assert!(base >= 0x400400 + align, "base {base:#x} too close to the image");

        // A 4K-aligned base is not good enough for this target and must be
        // refused rather than silently producing a broken binary.
        bin.set_injected_base(Some(0x50_1000));
        let err = bin.injected_base().unwrap_err().to_string();
        assert!(err.contains("page size"), "{err}");
        bin.set_injected_base(None);

        bin.inject_segment(base, &PAYLOAD).unwrap();

        // The loader's constraints have to hold modulo the target's page size,
        // not 4K.
        let l = layout(true);
        let po = injected_phdr(&bin, base, &PAYLOAD);
        let p_offset = bin.endian().read_uint(bin.data(), po + l.p_offset, l.word);
        assert_eq!(
            bin.endian().read_uint(bin.data(), po + l.p_align, l.word),
            align,
            "p_align must be the target's page size"
        );
        assert_eq!(
            p_offset % align,
            base % align,
            "p_vaddr {base:#x} and p_offset {p_offset:#x} must be congruent mod {align:#x}"
        );
    }

    /// Address translation and segment layout read the program-header table
    /// directly instead of parsing the whole image, so that reader has to agree
    /// with a real ELF parse -- in both classes and both byte orders, whose
    /// program-header layouts differ.
    #[test]
    fn program_headers_match_a_full_elf_parse() {
        use goblin::elf::program_header::{PT_LOAD, PT_NOTE};
        for (endian, arch) in [
            (Endian::Little, Architecture::X86_64),
            (Endian::Big, Architecture::Mips64),
            (Endian::Little, Architecture::Arm),
            (Endian::Big, Architecture::Mips),
        ] {
            let phdrs: [Phdr; 3] = [
                (PT_LOAD as u64, 0x1000, 0x400000, 0x400, 5, 0x10000),
                (PT_LOAD as u64, 0x1400, 0x500000, 0x123, 6, 0x1000),
                (PT_NOTE as u64, 0x100, 0x400100, 0x20, 4, 4),
            ];
            let bin = craft_injectable(endian, arch, 0x2000, &phdrs);
            let elf = goblin::elf::Elf::parse(bin.data()).unwrap();
            let ours = bin.program_headers().unwrap();

            assert_eq!(ours.len(), elf.program_headers.len(), "{arch:?}");
            for (i, (o, g)) in ours.iter().zip(&elf.program_headers).enumerate() {
                let layout = layout(arch.is_64());
                assert_eq!(o.entry_off, layout.phoff as usize + i * layout.phentsize);
                assert_eq!(
                    (o.p_type, o.p_offset, o.p_vaddr, o.p_filesz, o.p_memsz, o.p_align),
                    (
                        g.p_type,
                        g.p_offset,
                        g.p_vaddr,
                        g.p_filesz,
                        g.p_memsz,
                        g.p_align
                    ),
                    "{arch:?} program header {i}"
                );
            }
        }
    }

    /// A header claiming more program headers than the file holds must be
    /// refused. The reader indexes the image directly, so an unchecked count
    /// would panic rather than report a malformed target.
    #[test]
    fn program_headers_refuse_a_table_outside_the_image() {
        use goblin::elf::program_header::PT_LOAD;
        let phdrs: [Phdr; 1] = [(PT_LOAD as u64, 0, 0x400000, 0x400, 5, 0x1000)];
        let mut bin = craft_injectable(Endian::Little, Architecture::X86_64, 0x400, &phdrs);
        let l = layout(true);
        bin.endian.write_uint(&mut bin.data, l.phnum_off, 2, 1000);

        let err = bin.program_headers().unwrap_err().to_string();
        assert!(err.contains("runs past the end"), "{err}");
        // Every entry point that reads the table has to report it, not panic.
        assert!(bin.va_to_offset(0x400000).is_err());
        assert!(bin.injected_base().is_err());
        assert!(bin.inject_segment(INJECT_AT, &PAYLOAD).is_err());
    }

    /// Address translation must see a segment that was just injected: nothing
    /// derived from the program headers may outlive the edit that rewrote them.
    #[test]
    fn va_to_offset_sees_a_freshly_injected_segment() {
        use goblin::elf::program_header::{PT_LOAD, PT_NOTE};
        let phdrs: [Phdr; 2] = [
            (PT_LOAD as u64, 0, 0x400000, 0x400, 5, 0x1000),
            (PT_NOTE as u64, 0x100, 0x400100, 0x20, 4, 4),
        ];
        let mut bin = craft_injectable(Endian::Little, Architecture::X86_64, 0x400, &phdrs);

        assert!(bin.va_to_offset(INJECT_AT).is_err(), "nothing is mapped there yet");
        bin.inject_segment(INJECT_AT, &PAYLOAD).unwrap();

        let off = bin.va_to_offset(INJECT_AT).unwrap();
        assert_eq!(&bin.data()[off..off + PAYLOAD.len()], &PAYLOAD);
    }

    #[test]
    fn test_inject_segment_grows_phdr_table_without_note() {
        // Regression test: targets with no PT_NOTE must still be injectable by
        // growing the program-header table into its own padding, rather than
        // failing outright. The gap between the end of the table and the first
        // PT_LOAD's file offset mirrors the padding real toolchains leave
        // before the first page-aligned segment.
        use goblin::elf::program_header::PT_LOAD;
        for (endian, arch) in [
            (Endian::Little, Architecture::X86_64),
            (Endian::Big, Architecture::Mips),
        ] {
            let phdrs: [Phdr; 2] = [
                (PT_LOAD as u64, 0x1000, 0x400000, 0x400, 5, 0x1000),
                (PT_LOAD as u64, 0x1400, 0x500000, 0x400, 6, 0x1000),
            ];
            let mut bin = craft_injectable(endian, arch, 0x2000, &phdrs);
            bin.inject_segment(INJECT_AT, &PAYLOAD).unwrap();

            injected_phdr(&bin, INJECT_AT, &PAYLOAD);
            assert_eq!(phdr_types(&bin).len(), 3, "table should have grown for {arch:?}");
        }
    }

    /// A base that is aligned but not *free* has to be refused. Alignment used
    /// to be the only check, so an `--inject-base` pointing into the target's
    /// own image produced a `PT_LOAD` mapping the blob over the target's code,
    /// reported as a successful splice.
    #[test]
    fn test_inject_segment_rejects_base_overlapping_the_image() {
        use goblin::elf::program_header::{PT_LOAD, PT_NOTE};
        // Image occupies 0x400000..0x400400, so the first free page is
        // 0x401000 and everything below it is taken.
        let phdrs: [Phdr; 2] = [
            (PT_LOAD as u64, 0, 0x400000, 0x400, 5, 0x1000),
            (PT_NOTE as u64, 0x100, 0x400100, 0x20, 4, 4),
        ];
        let make = || craft_injectable(Endian::Little, Architecture::X86_64, 0x400, &phdrs);

        // The image's own base.
        let err = make().inject_segment(0x400000, &PAYLOAD).unwrap_err().to_string();
        assert!(err.contains("over the target's own PT_LOAD"), "{err}");
        assert!(err.contains("0x400000..0x400400"), "{err}");

        // A base below the image whose blob grows up into it.
        let err = make()
            .inject_segment(0x3ff000, &[0u8; 0x1001])
            .unwrap_err()
            .to_string();
        assert!(err.contains("over the target's own PT_LOAD"), "{err}");

        // The first page past the image is free, and still accepted.
        let mut bin = make();
        bin.inject_segment(0x401000, &PAYLOAD).unwrap();
        injected_phdr(&bin, 0x401000, &PAYLOAD);
        assert!(
            !phdr_types(&bin).contains(&(PT_NOTE as u64)),
            "PT_NOTE should have been consumed"
        );
    }

    #[test]
    fn test_inject_segment_no_note_no_room_errors() {
        // When there's no PT_NOTE *and* no padding to grow into, injection
        // must fail with a clear error instead of corrupting the file. The
        // single PT_LOAD's content starts immediately after the table's one
        // entry, so zero padding is available.
        use goblin::elf::program_header::PT_LOAD;
        let phdrs: [Phdr; 1] = [(PT_LOAD as u64, 0x40 + 56, 0x400000, 0x40, 5, 0x1000)];
        let mut bin = craft_injectable(Endian::Little, Architecture::X86_64, 0x100, &phdrs);
        let err = bin.inject_segment(INJECT_AT, &[0xde, 0xad]).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("no PT_NOTE") && msg.contains("padding"),
            "{msg}"
        );
    }
}
