pub use anyhow::Result;

use anyhow::{anyhow, Context};
use std::fs;
use std::path::Path;

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

impl Binary {
    /// Load a binary file from disk
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let data = fs::read(path)?;
        let (format, arch) = Self::detect_format_and_arch(&data)?;

        Ok(Binary { data, format, arch })
    }

    /// Detect the binary format and architecture
    fn detect_format_and_arch(data: &[u8]) -> Result<(BinaryFormat, Architecture)> {
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
                Ok((format, arch))
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
                Ok((format, arch))
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
                Ok((format, arch))
            }
            _ => Err(anyhow!("Unsupported binary format")),
        }
    }

    /// Detect just the binary format.
    #[allow(dead_code)]
    fn detect_format(data: &[u8]) -> Result<BinaryFormat> {
        let (format, _) = Self::detect_format_and_arch(data)?;
        Ok(format)
    }

    /// Apply a splice by directly patching the binary
    pub fn apply_direct_patch(&mut self, begin: u64, end: u64, code: &[u8]) -> Result<()> {
        let size = (end - begin) as usize;

        if code.len() > size {
            return Err(anyhow!("Invalid address range: {:#x} to {:#x}", begin, end));
        }

        // Direct substitution: replace bytes at the target address
        let start = begin as usize;
        let end_pos = start + code.len();

        if end_pos > self.data.len() {
            return Err(anyhow!("Invalid address range: {:#x} to {:#x}", begin, end));
        }

        self.data[start..end_pos].copy_from_slice(code);

        // If the new code is smaller, fill the rest with NOPs
        if code.len() < size {
            let nop_insn = self.get_nop_instruction();
            let remaining = start + size - end_pos;
            let mut offset = 0;

            while offset < remaining {
                let bytes_to_copy = std::cmp::min(nop_insn.len(), remaining - offset);
                self.data[end_pos + offset..end_pos + offset + bytes_to_copy]
                    .copy_from_slice(&nop_insn[..bytes_to_copy]);
                offset += bytes_to_copy;
            }
        }

        Ok(())
    }

    /// Get the NOP instruction bytes for the current architecture
    fn get_nop_instruction(&self) -> &'static [u8] {
        match self.arch {
            Architecture::X86 | Architecture::X86_64 => {
                &[0x90] // NOP
            }
            Architecture::Arm => {
                // MOV r0, r0 (little-endian)
                &[0x00, 0x00, 0xa0, 0xe1]
            }
            Architecture::Arm64 => {
                // NOP instruction for ARM64
                &[0x1f, 0x20, 0x03, 0xd5]
            }
            Architecture::Mips | Architecture::Mips64 => {
                // NOP instruction for MIPS
                &[0x00, 0x00, 0x00, 0x00]
            }
        }
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

                if offset < -0x800000 || offset > 0x7FFFFF {
                    return Err(anyhow!("Invalid address range: {:#x} to {:#x}", from, to));
                }

                let insn = 0xEA000000u32 | ((offset as u32) & 0x00FFFFFF);
                Ok(insn.to_le_bytes().to_vec())
            }
            Architecture::Arm64 => {
                // ARM64 branch instruction: B <offset>
                // Encoding: 0x14000000 | ((offset >> 2) & 0x03FFFFFF)
                let offset = ((to as i64 - from as i64) / 4) as i32;

                if offset < -0x2000000 || offset > 0x1FFFFFF {
                    return Err(anyhow!("Invalid address range: {:#x} to {:#x}", from, to));
                }

                let insn = 0x14000000u32 | ((offset as u32) & 0x03FFFFFF);
                Ok(insn.to_le_bytes().to_vec())
            }
            Architecture::Mips => {
                // MIPS J instruction: J <address>
                // Encoding: 0x08000000 | ((address >> 2) & 0x03FFFFFF)
                // Note: Target address must be in same 256MB region
                let addr_bits = ((to >> 2) & 0x03FFFFFF) as u32;
                let insn = 0x08000000u32 | addr_bits;
                Ok(insn.to_be_bytes().to_vec()) // MIPS is typically big-endian
            }
            Architecture::Mips64 => {
                // MIPS64 uses same J instruction format as MIPS32
                let addr_bits = ((to >> 2) & 0x03FFFFFF) as u32;
                let insn = 0x08000000u32 | addr_bits;
                Ok(insn.to_be_bytes().to_vec())
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

/// Apply every splice to a loaded binary via direct patching.
pub fn apply_splices(binary: &mut Binary, splices: &[Splice]) -> Result<()> {
    for splice in splices {
        let range = (splice.end - splice.begin) as usize;
        if splice.code.len() > range {
            return Err(anyhow!(
                "replacement code ({} bytes) is larger than target range \
                 {:#x}..{:#x} ({} bytes); trampolines are not yet implemented",
                splice.code.len(),
                splice.begin,
                splice.end,
                range,
            ));
        }
        binary.apply_direct_patch(splice.begin, splice.end, &splice.code)?;
    }
    Ok(())
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
        assert!(matches!(
            Binary::detect_format(&elf_header),
            Err(_) // Will fail to parse incomplete header
        ));
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
        };

        // Jump from 0x200 to 0x100
        binary.apply_jump_patch(0x200, 0x210, 0x100).unwrap();

        // Check JMP instruction
        assert_eq!(binary.data[0x200], 0xE9);

        // Offset will be negative
        let offset = (0x100 as i64 - (0x200 + 5) as i64) as i32;
        assert_eq!(binary.data[0x201], (offset & 0xFF) as u8);
    }

    #[test]
    fn test_jump_patch_out_of_bounds() {
        let mut binary = Binary {
            data: vec![0; 100],
            format: BinaryFormat::Elf,
            arch: Architecture::X86_64,
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
