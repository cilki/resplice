pub use anyhow::Result;

use anyhow::{anyhow, bail};
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

/// The instruction-unit displacement an ARM/AArch64 branch at `from` needs to
/// reach `to`, where the architecture resolves the branch against `pc` (`from`
/// on AArch64, `from + 8` on 32-bit ARM) and encodes the displacement in a
/// signed immediate holding `[-limit, limit)` instruction units.
///
/// Both conditions are checked on the full-width displacement, before it is
/// narrowed to what the immediate can hold. Computing `(to - from) / 4 as i32`
/// first and range-checking the result afterwards -- which is what this used to
/// do -- lets a displacement far too large to encode wrap into the permitted
/// range and pass: with an `--inject-base` 16GB from the splice, an AArch64
/// trampoline whose displacement is `0x1_0000_0000` instruction units narrowed
/// to `0`, and `resplice` reported a successful "trampoline to injected
/// segment" for a `b` that branched to itself. The 4-byte alignment matters for
/// the same reason: the immediate counts instructions, so a `to` that is not
/// 4-byte aligned silently rounds towards `from` and lands somewhere else.
fn branch_offset(from: u64, to: u64, pc: u64, limit: i64, insn: &str) -> Result<i64> {
    let disp = (to as i64).wrapping_sub(pc as i64);
    if disp % 4 != 0 {
        return Err(anyhow!(
            "trampoline at {from:#x} cannot reach {to:#x}: {insn} counts whole \
             instructions, so it can only branch to a 4-byte aligned address"
        ));
    }
    let offset = disp / 4;
    if !(-limit..limit).contains(&offset) {
        let reach = limit * 4;
        let away = (to as i64).wrapping_sub(from as i64);
        return Err(anyhow!(
            "trampoline at {from:#x} is {away:#x} bytes from its target {to:#x}, \
             which is out of reach of {insn}; choose an --inject-base within \
             {reach:#x} bytes of the splice"
        ));
    }
    Ok(offset)
}

/// Represents an ELF binary that can be patched
pub struct Binary {
    data: Vec<u8>,
    arch: Architecture,
    endian: Endian,
    /// Operator-supplied virtual address for the injected segment, overriding
    /// [`Binary::injected_base`]'s default. See [`Binary::set_injected_base`].
    inject_base: Option<u64>,
}

/// What a name the target defines resolves to.
///
/// Not every defined symbol carries an address a reference can be pointed at:
/// `st_value` means something else entirely for two symbol types a real target
/// genuinely has, and binding it as an address produces a patched binary that
/// is wrong with nothing to say so. Both are recorded here and refused by
/// [`TargetSymbols::resolve`] instead.
#[derive(Debug, Clone, Copy)]
enum Binding {
    /// A virtual address, which is what every usable definition is.
    Addr(u64),
    /// An `STT_GNU_IFUNC`, carrying the address of its resolver.
    Ifunc(u64),
    /// An `STT_TLS` symbol, carrying an offset into the thread-local block.
    Tls(u64),
}

/// What the target has to offer for each of the names a link asked about,
/// built by [`Binary::target_symbols`].
///
/// Only the names asked for are in here. A reference to a name that was not —
/// which nothing in this crate can produce, because the same set drives both
/// calls — is an error rather than a silent "the target does not define it".
pub struct TargetSymbols<'n> {
    /// One entry per name asked about, `None` for a name the target neither
    /// defines nor imports.
    bound: HashMap<&'n str, Option<Binding>>,
}

/// Record `binding` for a name in the map [`Binary::target_symbols`] builds.
///
/// The first binding seen for a name wins, which is where that function's
/// documented precedence comes from. A usable one is the exception: it replaces
/// an unusable binding recorded earlier, whichever table that came from, so an
/// ifunc or thread-local definition cannot shadow a plain address the target
/// also offers for the same name.
fn bind(slot: &mut Option<Binding>, binding: Binding) {
    match slot {
        None => *slot = Some(binding),
        Some(prev) => {
            if matches!(binding, Binding::Addr(_)) && !matches!(prev, Binding::Addr(_)) {
                *slot = Some(binding);
            }
        }
    }
}

impl TargetSymbols<'_> {
    /// What the target offers for `name`: its address, `None` when the target
    /// neither defines nor imports it, or an error when what it defines is not
    /// something a reference can be pointed at (see [`Binding`]).
    ///
    /// `name` must be one of the names [`Binary::target_symbols`] was asked
    /// about; anything else is a bug in the caller rather than a name the
    /// target lacks, and saying so beats resolving a real reference to nothing.
    pub fn resolve(&self, name: &str) -> Result<Option<u64>> {
        let Some(slot) = self.bound.get(name) else {
            return Err(anyhow!(
                "internal error: {name:?} was not among the names the target's \
                 symbol tables were searched for"
            ));
        };
        match *slot {
            Some(Binding::Addr(va)) => Ok(Some(va)),
            None => Ok(None),
            // An IFUNC's `st_value` is the address of the *resolver*: a
            // function the loader calls once, with no arguments, to be told
            // which implementation this machine should use. A static glibc
            // defines `memcpy`, `memmove`, `memset`, `strlen` and friends that
            // way, so this is what a splice that copies a slice or formats a
            // string used to bind to -- the resolver ran with the splice's own
            // arguments and its return value, a function pointer, was used as
            // the result.
            Some(Binding::Ifunc(va)) => Err(anyhow!(
                "the target defines {name:?} as an ifunc (STT_GNU_IFUNC), whose \
                 address {va:#x} is its *resolver*: calling it would run the \
                 resolver with this reference's arguments and use the \
                 implementation pointer it returns as the result. Resolving an \
                 ifunc takes the target's own IRELATIVE slot, which resplice does \
                 not read; provide the function from the rlib instead"
            )),
            // A TLS symbol's `st_value` is an offset within the thread-local
            // block, meaningless until a thread pointer is added to it. glibc's
            // `errno` is 0x30, which fits every relocation field there is, so
            // binding it as an address was silent: the splice read or wrote
            // absolute address 0x30.
            Some(Binding::Tls(va)) => Err(anyhow!(
                "the target defines {name:?} as a thread-local (STT_TLS), whose \
                 value {va:#x} is an offset into the thread-local block rather \
                 than an address; reaching it takes the thread pointer and the \
                 target's own TLS relocations, which resplice does not emit"
            )),
        }
    }
}

/// The NUL-terminated string at `off` in a string table, read straight out of
/// its bytes.
///
/// A preparsed index of the table's entries is the obvious alternative and the
/// wrong one twice over. It costs a pass over the whole table (and an owned
/// entry per string) to answer a question that is a NUL scan from `off`, and it
/// cannot answer it correctly: a string table may *suffix-merge* its entries,
/// so `printf` can exist only as the tail of `sprintf` and a symbol naming it
/// points into the middle of that entry. Looking `off` up in an index of entry
/// starts then has to find the entry it lands inside and take a substring of
/// it, which is what `goblin`'s `Strtab::get_at` binary-searches for — ~16
/// cache-missing probes into a megabyte of index, per symbol, where this reads
/// the bytes the offset already names.
fn strtab_str(strtab: &[u8], off: usize) -> Option<&str> {
    let rest = strtab.get(off..)?;
    let end = rest.iter().position(|&b| b == 0)?;
    std::str::from_utf8(&rest[..end]).ok()
}

/// The entry in `bound` for the symbol named at `off` in `strtab`, or `None`
/// when it is not one of the names [`Binary::target_symbols`] was asked about.
///
/// `first_bytes` is a 256-bit set of the first byte of every name asked about,
/// and it is what makes this cheap enough to run per symbol of the *target*:
/// patching a handful of functions into a 6MB binary asks two questions of a
/// table of 60k symbols, so a symbol that cannot be an answer has to be
/// rejected without decoding its name or hashing it. One byte load does it.
///
/// A nameless symbol — `st_name == 0`, which every section symbol has — names
/// the empty string rather than nothing, because the first byte of a string
/// table is the NUL terminating `""`. Binding that would make a reference with
/// no name at all (one against `STN_UNDEF`, which `R_*_NONE` and a damaged
/// object use) resolve to whichever section came first instead of being
/// reported as unresolvable, so the empty name is dropped here.
fn slot_for<'b>(
    bound: &'b mut HashMap<&str, Option<Binding>>,
    first_bytes: &[u64; 4],
    strtab: &[u8],
    off: usize,
) -> Option<&'b mut Option<Binding>> {
    let &first = strtab.get(off)?;
    if first == 0 || first_bytes[(first >> 6) as usize] >> (first & 63) & 1 == 0 {
        return None;
    }
    bound.get_mut(strtab_str(strtab, off)?)
}

/// Where the fields resplice reads and writes sit in one ELF class.
///
/// The 32- and 64-bit structures do not just differ in width: ELF32 has no
/// 8-byte `sh_flags`, keeps `p_flags` after `p_memsz` rather than after
/// `p_type`, and `Elf64_Sym` moves `st_value` ahead of `st_info`/`st_shndx`.
/// Every reader and writer of the target's headers takes its offsets from
/// here, so that knowledge lives in one place. See [`ELF32`] and [`ELF64`].
struct Class {
    /// Width of an address or offset field.
    word: usize,
    /// Size of the ELF header.
    ehsize: usize,
    // ELF header fields.
    e_phoff: usize,
    e_shoff: usize,
    e_phentsize: usize,
    e_phnum: usize,
    e_shentsize: usize,
    e_shnum: usize,
    e_shstrndx: usize,
    // Program-header entry; `p_type` is at 0 in both classes.
    phentsize: usize,
    p_flags: usize,
    p_offset: usize,
    p_vaddr: usize,
    p_paddr: usize,
    p_filesz: usize,
    p_memsz: usize,
    p_align: usize,
    // Section-header entry; `sh_name` and `sh_type` are at 0 and 4 in both.
    shentsize: usize,
    sh_addr: usize,
    sh_offset: usize,
    sh_size: usize,
    sh_link: usize,
    sh_entsize: usize,
    // Symbol-table entry; `st_name` is at 0 in both.
    symsize: usize,
    st_value: usize,
    st_info: usize,
    st_shndx: usize,
    // Relocation entries. `r_info` follows `r_offset` in both forms.
    relsize: usize,
    relasize: usize,
}

impl Class {
    /// The layout of the 64-bit class if `is_64`, else of the 32-bit one.
    fn of(is_64: bool) -> &'static Class {
        if is_64 { &ELF64 } else { &ELF32 }
    }
}

const ELF32: Class = Class {
    word: 4,
    ehsize: 52,
    e_phoff: 0x1c,
    e_shoff: 0x20,
    e_phentsize: 0x2a,
    e_phnum: 0x2c,
    e_shentsize: 0x2e,
    e_shnum: 0x30,
    e_shstrndx: 0x32,
    phentsize: 32,
    p_offset: 4,
    p_vaddr: 8,
    p_paddr: 12,
    p_filesz: 16,
    p_memsz: 20,
    p_flags: 24,
    p_align: 28,
    shentsize: 40,
    sh_addr: 12,
    sh_offset: 16,
    sh_size: 20,
    sh_link: 24,
    sh_entsize: 36,
    symsize: 16,
    st_value: 4,
    st_info: 12,
    st_shndx: 14,
    relsize: 8,
    relasize: 12,
};

const ELF64: Class = Class {
    word: 8,
    ehsize: 64,
    e_phoff: 0x20,
    e_shoff: 0x28,
    e_phentsize: 0x36,
    e_phnum: 0x38,
    e_shentsize: 0x3a,
    e_shnum: 0x3c,
    e_shstrndx: 0x3e,
    phentsize: 56,
    p_flags: 4,
    p_offset: 8,
    p_vaddr: 16,
    p_paddr: 24,
    p_filesz: 32,
    p_memsz: 40,
    p_align: 48,
    shentsize: 64,
    sh_addr: 16,
    sh_offset: 24,
    sh_size: 32,
    sh_link: 40,
    sh_entsize: 56,
    symsize: 24,
    st_info: 4,
    st_shndx: 6,
    st_value: 8,
    relsize: 16,
    relasize: 24,
};

/// One entry of the target's section-header table.
#[derive(Debug, Clone, Copy)]
struct Shdr {
    sh_name: u32,
    sh_type: u32,
    sh_addr: u64,
    sh_offset: u64,
    sh_size: u64,
    sh_link: u32,
    sh_entsize: u64,
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
        }
    }

    /// Pin the injected segment to `base` instead of the default.
    ///
    /// The default places the segment one page past the end of the image, which
    /// is only safe when nothing else claims that address. It often is not: a
    /// game console binary typically has its allocator hand out memory starting
    /// at the end of `.bss`, so the default lands one page into the heap and the
    /// injected code and data are live only until the first big allocation.
    /// There is no portable way to detect that, so the operator supplies a known
    /// free address instead.
    ///
    /// Nothing is validated here — the base is only a request until the segment
    /// is laid out. It is rejected later if it is not aligned to the target's
    /// page size (see [`Binary::injected_base`]), if it is not actually free
    /// (see [`Binary::inject_segment`]), or if it is so far from the splice
    /// sites that their PC-relative references cannot reach it (see
    /// [`Binary::jump_bytes`] and `arch::apply`).
    pub fn set_injected_base(&mut self, base: Option<u64>) {
        self.inject_base = base;
    }

    /// Detect the architecture and byte order of an ELF target.
    ///
    /// Every operation this crate performs on a target — translating virtual
    /// addresses, reading its symbol table, injecting a segment — is ELF-only,
    /// so non-ELF inputs are rejected here rather than later.
    ///
    /// All three answers live in `e_ident` and `e_machine`, at fixed offsets in
    /// the first 20 bytes of the image, so this reads them rather than parsing
    /// the image to be told them. A full parse here cost 1.5ms of a 20ms run on
    /// a 6MB target, all of it spent indexing string tables and reading
    /// relocation sections that nothing then asked about; everything the rest
    /// of this file reads out of a target it reads (and bounds-checks) itself.
    fn detect_arch(data: &[u8]) -> Result<(Architecture, Endian)> {
        use goblin::elf::header::{EM_386, EM_AARCH64, EM_ARM, EM_MIPS, EM_X86_64};

        if data.len() < 0x14 || data[..4] != [0x7f, b'E', b'L', b'F'] {
            bail!("target is not an ELF binary");
        }
        // `EI_CLASS` and `EI_DATA`, the two bytes every later read depends on:
        // they fix the width of the header fields and the order of their
        // bytes, so a value neither standard class nor standard byte order has
        // to be refused rather than guessed at.
        let is_64 = match data[4] {
            1 => false,
            2 => true,
            other => bail!("target has unsupported ELF class {other}"),
        };
        let endian = match data[5] {
            1 => Endian::Little,
            2 => Endian::Big,
            other => bail!("target has unsupported ELF byte order {other}"),
        };
        // An image too short for its class's header is not one anything here
        // can read: the program-header table's location alone lives past the
        // end of it. Every `Binary` is built from an image that passed this,
        // and an image only ever grows, so the header readers below index the
        // ELF header without checking its length again.
        if data.len() < Class::of(is_64).ehsize {
            bail!(
                "image is {} bytes, too short to hold an ELF header",
                data.len()
            );
        }

        let machine = endian.read_uint(data, 0x12, 2) as u16;
        let arch = match machine {
            EM_386 => Architecture::X86,
            EM_X86_64 => Architecture::X86_64,
            EM_ARM => Architecture::Arm,
            EM_AARCH64 => Architecture::Arm64,
            EM_MIPS if is_64 => Architecture::Mips64,
            EM_MIPS => Architecture::Mips,
            other => return Err(anyhow!("unsupported ELF machine type {other}")),
        };
        // The machine type implies the pointer width everywhere except MIPS,
        // which is resolved above; a header that disagrees with itself would
        // make every subsequent header field be read at the wrong width.
        if arch.is_64() != is_64 {
            return Err(anyhow!("ELF class contradicts machine type {machine}"));
        }
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
        //
        // A reimplementation is usually well short of the function it replaces,
        // so most of a region is this fill, and a crate of a few thousand
        // splices fills a megabyte or two of it. Written one instruction at a
        // time through a bounds-checked slice copy of a length the compiler
        // could not see, that was a `memcpy` call per four bytes; a whole-tail
        // `fill` for a one-byte NOP and a fixed-width store per word for a
        // four-byte one are a fraction of that. The tail may end mid-word, in
        // which case it holds as much of the instruction as fits, as before.
        if end_pos < region_end {
            let (nop, nop_len) = self.get_nop_instruction();
            let tail = &mut self.data[end_pos..region_end];
            if nop_len == 1 {
                tail.fill(nop[0]);
            } else {
                // The only other width is the whole buffer, which is what
                // lets the copy below be a fixed-width store.
                debug_assert_eq!(nop_len, nop.len());
                let mut words = tail.chunks_exact_mut(nop.len());
                for word in &mut words {
                    word.copy_from_slice(&nop);
                }
                let rest = words.into_remainder();
                rest.copy_from_slice(&nop[..rest.len()]);
            }
        }

        Ok(())
    }

    /// Get the NOP instruction bytes for the current architecture, encoded in
    /// the target's byte order: a four-byte buffer and the length of it the
    /// instruction occupies.
    ///
    /// By value rather than as a `Vec`: the NOP fill runs once per splice, and
    /// a crate that reimplements a few thousand functions has as many, so
    /// returning it on the heap was an allocation per splice for four bytes.
    fn get_nop_instruction(&self) -> ([u8; 4], usize) {
        match self.arch {
            Architecture::X86 | Architecture::X86_64 => ([0x90, 0, 0, 0], 1),
            Architecture::Arm => (self.encode_insn(0xe1a0_0000), 4), // mov r0, r0
            Architecture::Arm64 => (self.encode_insn(0xd503_201f), 4), // nop
            Architecture::Mips | Architecture::Mips64 => ([0; 4], 4),
        }
    }

    /// Encode a 32-bit instruction word in the target's byte order.
    fn encode_insn(&self, insn: u32) -> [u8; 4] {
        let mut b = [0u8; 4];
        self.endian.write_uint(&mut b, 0, 4, insn as u64);
        b
    }

    /// Machine code for an unconditional jump from `from` to `to`, for the
    /// current architecture. Used to build trampolines for oversized splices.
    pub fn jump_bytes(&self, from: u64, to: u64) -> Result<Vec<u8>> {
        match self.arch {
            Architecture::X86 | Architecture::X86_64 => {
                // JMP rel32 (E9 XX XX XX XX)
                let disp = to as i64 - (from as i64 + 5);
                // The displacement is a signed 32-bit field, and on x86-64 the
                // two addresses can be further apart than it can express: with
                // an `--inject-base` more than 2GB from the splice site the
                // high bits were dropped, leaving a jump to an unrelated
                // address -- reported as a successfully trampolined splice.
                // (A 32-bit target cannot get there: its whole address space
                // fits in the field, where a backwards jump correctly wraps.)
                if self.is_64() && i32::try_from(disp).is_err() {
                    return Err(anyhow!(
                        "trampoline at {from:#x} is {disp:#x} bytes from its target \
                         {to:#x}, which does not fit the signed 32-bit displacement \
                         of a `jmp rel32`; choose an --inject-base within 2GB of the \
                         splice"
                    ));
                }
                let offset = disp as i32;
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
                let offset = branch_offset(from, to, pc, 0x80_0000, "`b` (24-bit imm)")?;
                let insn = 0xEA000000u32 | ((offset as u32) & 0x00FFFFFF);
                Ok(self.encode_insn(insn).to_vec())
            }
            Architecture::Arm64 => {
                // ARM64 branch instruction: B <offset>
                // Encoding: 0x14000000 | ((offset >> 2) & 0x03FFFFFF)
                let offset = branch_offset(from, to, from, 0x200_0000, "`b` (26-bit imm)")?;
                let insn = 0x14000000u32 | ((offset as u32) & 0x03FFFFFF);
                Ok(self.encode_insn(insn).to_vec())
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
                let mut out = self.encode_insn(0x08000000u32 | addr_bits).to_vec();
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

    /// The header layout of the target's ELF class.
    fn class(&self) -> &'static Class {
        Class::of(self.is_64())
    }

    /// Whether the target is position-independent (`ET_DYN`) — a PIE
    /// executable, which is what every mainstream toolchain produces by
    /// default, or a shared library.
    ///
    /// The loader maps such an image at a base it picks at run time, so the
    /// addresses its headers name are only link-time *offsets*. Everything
    /// `resplice` writes is resolved at link time against those offsets, and
    /// `resplice` adds no dynamic relocations, so whether a reference survives
    /// depends on what kind of value its field holds:
    ///
    /// - A PC-relative displacement survives unchanged. The splice, the
    ///   injected segment and the target's own code are all shifted by the same
    ///   amount, so the distance between any two of them is the same as it was.
    ///   That covers a call, a jump, a trampoline, and the `adrp`/`lea` form of
    ///   a data reference — the bulk of what a replacement compiles into.
    /// - An *absolute* address does not. Nothing relocates it once the image is
    ///   loaded, so the target dereferences a link-time offset as if it were an
    ///   address and reads from (or jumps into) the bottom of its address
    ///   space.
    ///
    /// [`link_rlib`] therefore refuses the absolute references rather than
    /// reporting a successful splice whose binary faults. See
    /// `arch::writes_absolute_address`.
    pub fn is_position_independent(&self) -> bool {
        use goblin::elf::header::ET_DYN;
        // `e_type` is the first field after the 16-byte identifier in both ELF
        // classes, so reading it needs nothing `detect_arch` has not already
        // validated.
        self.endian.read_uint(&self.data, 0x10, 2) == ET_DYN as u64
    }

    /// Read the section-header table straight out of the image, for the same
    /// reason [`program_headers`](Self::program_headers) does: the symbol
    /// lookup that needs it wants four of the target's sections, while a full
    /// parse of the image preparses every string table it has — each into an
    /// owned index of all its entries, with the whole table validated as UTF-8
    /// — and reads every relocation section, none of which any caller here
    /// looks at.
    ///
    /// Entries whose declared size is not this ELF class's are refused rather
    /// than read at the wrong stride, and a table the header places outside the
    /// image is refused rather than read past the end of it.
    fn section_headers(&self) -> Result<Vec<Shdr>> {
        let endian = self.endian;
        let c = self.class();
        let (word, shentsize) = (c.word, c.shentsize);

        let shoff = endian.read_uint(&self.data, c.e_shoff, word) as usize;
        let got_shentsize = endian.read_uint(&self.data, c.e_shentsize, 2) as usize;
        let mut shnum = endian.read_uint(&self.data, c.e_shnum, 2) as usize;
        if shoff == 0 {
            // No section-header table at all: a stripped image, which has no
            // symbol tables to read either.
            return Ok(Vec::new());
        }
        if got_shentsize != shentsize {
            return Err(anyhow!("unexpected section-header entry size {got_shentsize}"));
        }
        let outside = |entries: usize, end: Option<usize>| {
            end.is_none_or(|end| end > self.data.len()).then(|| {
                anyhow!(
                    "section-header table at {shoff:#x} ({entries} entries of {shentsize} \
                     bytes) runs past the end of the {}-byte image",
                    self.data.len()
                )
            })
        };
        // `e_shnum` is 16 bits wide, so an image with more sections than it can
        // express keeps the real count in the null section header's `sh_size`
        // and leaves the field zero. The null header has to be in the image to
        // be asked.
        if shnum == 0 {
            if let Some(err) = outside(1, shoff.checked_add(shentsize)) {
                return Err(err);
            }
            shnum = endian.read_uint(&self.data, shoff + c.sh_size, word) as usize;
            if shnum == 0 {
                return Ok(Vec::new());
            }
        }
        let end = shnum.checked_mul(shentsize).and_then(|n| shoff.checked_add(n));
        if let Some(err) = outside(shnum, end) {
            return Err(err);
        }

        Ok((0..shnum)
            .map(|i| {
                let off = shoff + i * shentsize;
                Shdr {
                    sh_name: endian.read_uint(&self.data, off, 4) as u32,
                    sh_type: endian.read_uint(&self.data, off + 4, 4) as u32,
                    sh_addr: endian.read_uint(&self.data, off + c.sh_addr, word),
                    sh_offset: endian.read_uint(&self.data, off + c.sh_offset, word),
                    sh_size: endian.read_uint(&self.data, off + c.sh_size, word),
                    sh_link: endian.read_uint(&self.data, off + c.sh_link, 4) as u32,
                    sh_entsize: endian.read_uint(&self.data, off + c.sh_entsize, word),
                }
            })
            .collect())
    }

    /// The bytes of `sh`, or `None` when the file range it claims is not
    /// wholly inside the image.
    fn section_bytes(&self, sh: &Shdr) -> Option<&[u8]> {
        let start = usize::try_from(sh.sh_offset).ok()?;
        let len = usize::try_from(sh.sh_size).ok()?;
        self.data.get(start..start.checked_add(len)?)
    }

    /// The bytes of the section `idx` names, or `None` for an index outside the
    /// table or a section the image does not back.
    fn section_bytes_at(&self, shdrs: &[Shdr], idx: usize) -> Option<&[u8]> {
        self.section_bytes(shdrs.get(idx)?)
    }

    /// Find the section named `name`.
    ///
    /// The section names live in a string table of their own, named by
    /// `e_shstrndx` — or, when the index does not fit that 16-bit field, by the
    /// `sh_link` of the null section header that `SHN_XINDEX` defers to.
    fn section_by_name(&self, shdrs: &[Shdr], name: &str) -> Option<Shdr> {
        use goblin::elf::section_header::SHN_XINDEX;
        if shdrs.is_empty() {
            return None;
        }
        let mut idx = self
            .endian
            .read_uint(&self.data, self.class().e_shstrndx, 2) as usize;
        if idx == SHN_XINDEX as usize {
            idx = shdrs.first()?.sh_link as usize;
        }
        let names = self.section_bytes_at(shdrs, idx)?;
        shdrs
            .iter()
            .find(|sh| strtab_str(names, sh.sh_name as usize) == Some(name))
            .copied()
    }

    /// Read the program-header table straight out of the image.
    ///
    /// This is deliberately not a full ELF parse: the table is a dozen
    /// fixed-layout entries, while parsing the whole image walks its section
    /// headers, symbol tables and string tables. Address translation
    /// needs nothing but this table and is called once per splice, so paying
    /// for the rest of the parse each time dominated the tool's runtime.
    ///
    /// Reading the bytes directly also keeps the result current: the table is
    /// rewritten by [`inject_segment`](Self::inject_segment), and every caller
    /// here must see that edit.
    fn program_headers(&self) -> Result<Vec<Phdr>> {
        let endian = self.endian;
        let c = self.class();
        let (word, phentsize) = (c.word, c.phentsize);

        let phoff = endian.read_uint(&self.data, c.e_phoff, word) as usize;
        let got_phentsize = endian.read_uint(&self.data, c.e_phentsize, 2) as usize;
        let phnum = endian.read_uint(&self.data, c.e_phnum, 2) as usize;
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
                    p_offset: endian.read_uint(&self.data, off + c.p_offset, word),
                    p_vaddr: endian.read_uint(&self.data, off + c.p_vaddr, word),
                    p_filesz: endian.read_uint(&self.data, off + c.p_filesz, word),
                    p_memsz: endian.read_uint(&self.data, off + c.p_memsz, word),
                    p_align: endian.read_uint(&self.data, off + c.p_align, word),
                }
            })
            .collect())
    }

    /// Translate a virtual address to a file offset using the program headers.
    pub fn va_to_offset(&self, va: u64) -> Result<usize> {
        self.va_range_to_offset(va, 1)
    }

    /// Translate `[va, va + len)` to the file offset of its first byte,
    /// requiring the *whole* range to be file-backed by a single `PT_LOAD`.
    ///
    /// Translating only `va` is not enough for a caller that goes on to write
    /// `len` bytes there. A virtual address range is contiguous in the file only
    /// for as long as one segment maps it: past `p_offset + p_filesz` the file
    /// carries something else entirely — inter-segment padding, the next
    /// segment's content, or the section headers — while the addresses past
    /// `p_vaddr + p_filesz` are `.bss` or not mapped at all.
    ///
    /// So a patch region that runs off the end of its segment lands on file
    /// bytes that have nothing to do with the addresses asked for. A `static`
    /// splice whose range overshot `.data` into `.bss` used to overwrite the
    /// symbol table, and a longer overshoot reached the next segment's
    /// `.dynamic` and `.got`, producing a binary that segfaults in the loader —
    /// both reported as a successful splice.
    pub fn va_range_to_offset(&self, va: u64, len: u64) -> Result<usize> {
        self.translate(&self.program_headers()?, va, len)
    }

    /// Translate every range in `ranges`, reading the program-header table once
    /// instead of once per range.
    ///
    /// The table cannot be cached in the `Binary`: [`patch_bytes`](Self::patch_bytes)
    /// may rewrite the very bytes it is read from, so nothing derived from it
    /// survives a patch. A caller with a few thousand splices to place therefore
    /// re-read it once per splice, which is most of what translating them cost.
    /// The ranges are all known before any of them is written, so they are
    /// translated in one go against a single read of the table.
    ///
    /// The outer `Result` is the one read of the table; each range then gets a
    /// `Result` of its own, so the caller can name whichever of *its* ranges
    /// failed rather than only the address in the message.
    pub fn va_range_to_offsets(
        &self,
        ranges: impl IntoIterator<Item = (u64, u64)>,
    ) -> Result<Vec<Result<usize>>> {
        let phdrs = self.program_headers()?;
        Ok(ranges
            .into_iter()
            .map(|(va, len)| self.translate(&phdrs, va, len))
            .collect())
    }

    /// [`va_range_to_offset`](Self::va_range_to_offset) against an
    /// already-read program-header table.
    fn translate(&self, phdrs: &[Phdr], va: u64, len: u64) -> Result<usize> {
        use goblin::elf::program_header::PT_LOAD;

        let end = va
            .checked_add(len)
            .ok_or_else(|| anyhow!("range {va:#x}..+{len:#x} wraps the address space"))?;

        // Saturating throughout: `p_vaddr`/`p_filesz`/`p_offset` come straight
        // out of the target's headers, which nothing has vetted, and a sum that
        // wrapped would make a segment appear to map addresses it does not.
        let mut truncated = None;
        let mut zero_filled = None;
        for ph in phdrs {
            if ph.p_type != PT_LOAD {
                continue;
            }
            let seg_end = ph.p_vaddr.saturating_add(ph.p_filesz);
            // `p_memsz` is required to be at least `p_filesz`, but these fields
            // are unvetted: clamping keeps a bogus one from making a
            // file-backed address look unmapped.
            let mem_end = seg_end.max(ph.p_vaddr.saturating_add(ph.p_memsz));
            if va < ph.p_vaddr || va >= mem_end {
                continue;
            }
            if va >= seg_end {
                // Mapped, but by the segment's zero-filled tail -- `.bss`,
                // which the loader creates out of nothing. There is no file
                // byte at this address to patch, and saying "not mapped" sends
                // the user to re-check an address `readelf -l` agrees is
                // mapped. Another segment may still file-back it, so record it
                // and keep looking.
                zero_filled = Some((ph.p_vaddr, seg_end, mem_end));
                continue;
            }
            if end > seg_end {
                // Another segment may still map the whole range (they are not
                // required to be disjoint), so keep looking and report this one
                // only if none does.
                truncated = Some((ph.p_vaddr, seg_end));
                continue;
            }
            let off = ph.p_offset.saturating_add(va - ph.p_vaddr);
            if off.saturating_add(len) > self.data.len() as u64 {
                return Err(anyhow!(
                    "{va:#x} maps to file offset {off:#x}, where {len:#x} bytes run \
                     past the end of the {}-byte image",
                    self.data.len()
                ));
            }
            return Ok(off as usize);
        }

        match truncated {
            Some((seg_start, seg_end)) => Err(anyhow!(
                "the PT_LOAD that maps {va:#x} holds file-backed data only for \
                 {seg_start:#x}..{seg_end:#x}, so the {} bytes of {va:#x}..{end:#x} \
                 past {seg_end:#x} are not mapped there at all -- writing them would \
                 land on whatever the file happens to hold instead (inter-segment \
                 padding, the next segment's content, the section headers)",
                end - seg_end
            )),
            None => match zero_filled {
                Some((seg_start, seg_end, mem_end)) => Err(anyhow!(
                    "{va:#x} is in the zero-filled tail (`.bss`) of the PT_LOAD at \
                     {seg_start:#x}..{mem_end:#x}: that segment is backed by the file \
                     only up to {seg_end:#x}, and the loader zeroes the rest out of \
                     nothing, so there is no byte in the image at {va:#x} to patch. \
                     A `.bss` address cannot be given an initializer by splicing -- \
                     have a code splice write the value at run time, or splice an \
                     address in `.data`/`.rodata` instead"
                )),
                None => Err(anyhow!(
                    "virtual address {va:#x} is not mapped by any PT_LOAD segment"
                )),
            },
        }
    }

    /// The highest virtual address occupied by any loadable segment.
    fn max_vaddr(&self) -> Result<u64> {
        use goblin::elf::program_header::PT_LOAD;
        Ok(self
            .program_headers()?
            .iter()
            .filter(|ph| ph.p_type == PT_LOAD)
            // Saturating: the header fields are unvetted, and a wrapping sum
            // here would under-report the end of the image, putting the default
            // injected base inside it.
            .map(|ph| ph.p_vaddr.saturating_add(ph.p_memsz))
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

    /// The base virtual address the injected segment will be mapped at.
    ///
    /// With no [pinned base](Self::set_injected_base), that is one page above
    /// the end of the target's image — one *target* page, the largest `p_align`
    /// of any `PT_LOAD`, which is 64K on a typical aarch64 image rather than 4K.
    ///
    /// A pinned base is returned as given, but only after it is checked against
    /// that same granularity: `p_vaddr ≡ p_offset (mod p_align)` has to hold for
    /// the loader to accept the image at all, and the file offset the blob is
    /// appended at is derived from the base. Whether the base is *free* is not
    /// knowable until the blob's length is, and is checked in
    /// [`inject_segment`](Self::inject_segment).
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

    /// The lowest file offset at or after `limit` that something in the target
    /// already occupies, or `None` if the target carries no information that
    /// answers the question.
    ///
    /// [`inject_segment`](Self::inject_segment) uses this to size the padding
    /// gap that follows the program-header table, so it must report *content*,
    /// not mere mapping. Section headers are the authoritative source: a
    /// section is exactly a claimed range of file bytes, and neither the ELF
    /// header nor the program-header table is a section, so the first section
    /// body at or after `limit` is where the padding stops.
    ///
    /// Segment `p_offset`s cannot answer it, because a `PT_LOAD` says which
    /// bytes are *mapped* rather than which hold content. The first `PT_LOAD`
    /// of an executable starts at file offset 0 and spans the ELF header, the
    /// program-header table, the padding *and* the content that follows; and
    /// `PT_PHDR` describes the table itself, so its `p_offset` lies below the
    /// table's end. They are only consulted for a target with no section
    /// headers, and then only for segments beginning at or after `limit`,
    /// where their start really is the start of content.
    fn first_claimed_after(&self, limit: u64) -> Result<Option<u64>> {
        use goblin::elf::program_header::PT_PHDR;
        use goblin::elf::section_header::SHT_NOBITS;

        let shdrs = self.section_headers()?;
        let mut first: Option<u64> = None;
        let mut claim = |start: u64, size: u64| {
            if size == 0 {
                return;
            }
            // A range beginning below `limit` but reaching past it occupies
            // `limit` itself, leaving no gap at all.
            let start = if start < limit && start.saturating_add(size) > limit {
                limit
            } else {
                start
            };
            if start >= limit {
                first = Some(first.map_or(start, |f: u64| f.min(start)));
            }
        };

        if shdrs.is_empty() {
            for ph in self.program_headers()? {
                if ph.p_type != PT_PHDR && ph.p_offset >= limit {
                    // `p_filesz` can be 0 for a segment that still pins its
                    // `p_offset`; treat the offset itself as occupied.
                    claim(ph.p_offset, ph.p_filesz.max(1));
                }
            }
        } else {
            for sh in &shdrs {
                // `SHT_NOBITS` sections (`.bss`, `.tbss`) occupy no file
                // bytes, and their `sh_offset` routinely overlaps content that
                // does.
                if sh.sh_type != SHT_NOBITS {
                    claim(sh.sh_offset, sh.sh_size);
                }
            }
        }
        // The section-header table, which conventionally ends the file. Its
        // location is read from the ELF header rather than carried by a `Shdr`,
        // since no section describes it.
        let c = self.class();
        claim(
            self.endian.read_uint(&self.data, c.e_shoff, c.word),
            self.endian.read_uint(&self.data, c.e_shentsize, 2) * shdrs.len() as u64,
        );
        Ok(first)
    }

    /// Resolve `names` against the target binary's own symbols, in one pass
    /// over its symbol tables.
    ///
    /// Each name is tried, in order, against: a defined symbol in `.symtab`, a
    /// defined symbol in `.dynsym`, then a function the target imports through
    /// its PLT (the stub's address). A usable definition is the exception to
    /// that precedence: it replaces an unusable one recorded earlier (see
    /// [`bind`]). [`TargetSymbols::resolve`] reports what each name got.
    ///
    /// The names a link needs are known before any of them is looked up — they
    /// are what the rlib leaves undefined — so they are all asked for at once.
    /// Resolving them one at a time is what this used to do, and it has no
    /// cheap implementation: `.symtab` carries no index to look a name up in,
    /// so each query is a scan, and the whole-table map that avoided
    /// re-scanning cost an owned `String` and a hash insert for every symbol
    /// the *target* defines. That is the wrong size entirely: patching a
    /// handful of functions into a 6MB binary of 60k symbols spent two thirds
    /// of the run building 60k map entries to answer two questions.
    ///
    /// Asking for nothing parses nothing, which is the whole cost for a
    /// replacement that reaches none of the target's own symbols.
    ///
    /// What the pass costs per symbol matters as much as there being one of
    /// them, because the table it walks is the target's: a 6MB binary has 60k
    /// symbols whatever the rlib asks about. So the tables are read straight
    /// out of the image ([`section_headers`](Self::section_headers)), each
    /// name is read from the string table at the offset the symbol gives
    /// ([`strtab_str`]), and a symbol whose name cannot be one of the names
    /// asked about is rejected by a single byte load ([`slot_for`]).
    pub fn target_symbols<'n>(
        &self,
        names: impl IntoIterator<Item = &'n str>,
    ) -> Result<TargetSymbols<'n>> {
        use goblin::elf::section_header::{SHN_UNDEF, SHT_DYNSYM, SHT_RELA, SHT_SYMTAB};
        use goblin::elf::sym::{STT_GNU_IFUNC, STT_TLS};

        let mut bound: HashMap<&'n str, Option<Binding>> =
            names.into_iter().map(|name| (name, None)).collect();
        if bound.is_empty() {
            return Ok(TargetSymbols { bound });
        }

        // A 256-bit set of the first byte of every name asked about, which is
        // what keeps the per-symbol test to a byte load. See [`slot_for`].
        let mut first_bytes = [0u64; 4];
        for name in bound.keys() {
            if let Some(&first) = name.as_bytes().first() {
                first_bytes[(first >> 6) as usize] |= 1 << (first & 63);
            }
        }

        let endian = self.endian;
        let is_64 = self.is_64();
        let c = self.class();
        let (word, sym_size) = (c.word, c.symsize);
        let shdrs = self.section_headers()?;

        // 1 & 2: symbols the target defines itself.
        for kind in [SHT_SYMTAB, SHT_DYNSYM] {
            let Some(sh) = shdrs.iter().rfind(|sh| sh.sh_type == kind) else {
                continue;
            };
            // `sh_link` of a symbol table names the string table its names
            // live in. A table the image does not back, or one whose names are
            // nowhere, is skipped rather than read out of bounds.
            let (Some(table), Some(strtab)) = (
                self.section_bytes(sh),
                self.section_bytes_at(&shdrs, sh.sh_link as usize),
            ) else {
                continue;
            };
            // `sh_entsize` is the stride between entries, which is this
            // class's entry size in every real image. A larger one (padded
            // entries) is honoured; a smaller or absent one would make entries
            // overlap, so the real size is used instead.
            let stride = (sh.sh_entsize as usize).max(sym_size);
            for entry in (0..).map_while(|i| {
                let at = i * stride;
                table.get(at..at.checked_add(sym_size)?)
            }) {
                // Only `st_shndx` and `st_name` are read for every symbol of
                // the target; the rest of the entry is read for the few that
                // name something the link asked about.
                if endian.read_uint(entry, c.st_shndx, 2) as u32 == SHN_UNDEF {
                    continue;
                }
                let name_off = endian.read_uint(entry, 0, 4) as usize;
                let Some(slot) = slot_for(&mut bound, &first_bytes, strtab, name_off) else {
                    continue;
                };
                let value = endian.read_uint(entry, c.st_value, word);
                if value == 0 {
                    continue;
                }
                let st_type = entry[c.st_info] & 0xf;
                bind(
                    slot,
                    match st_type {
                        STT_GNU_IFUNC => Binding::Ifunc(value),
                        STT_TLS => Binding::Tls(value),
                        _ => Binding::Addr(value),
                    },
                );
            }
        }

        // 3: functions the target imports through the PLT. The i-th
        // `.rela.plt` entry maps to the `.plt` stub at
        // `plt_base + header + i * entry`, whose sizes are
        // architecture-specific. (MIPS classically uses `.MIPS.stubs` rather
        // than a `.plt` of this shape; its import binding is best-effort and
        // target-*defined* symbols above are the reliable path there.)
        let (plt_header, plt_entry) = match self.arch {
            Architecture::X86 | Architecture::X86_64 => (16, 16),
            Architecture::Arm64 => (32, 16),
            Architecture::Arm => (20, 12),
            Architecture::Mips | Architecture::Mips64 => (32, 16),
        };
        let plt = self.section_by_name(&shdrs, ".plt");
        // The PLT's own relocations, which both name the imported function and
        // fix the order of the stubs. `.rela.plt`/`.rel.plt` is the section
        // `DT_JMPREL` points into, and finding it by name asks no more of the
        // image than finding `.plt` by name already does.
        let jmprel = self
            .section_by_name(&shdrs, ".rela.plt")
            .or_else(|| self.section_by_name(&shdrs, ".rel.plt"));
        if let (Some(plt), Some(jmprel)) = (plt, jmprel) {
            let dynsym = shdrs.iter().rfind(|sh| sh.sh_type == SHT_DYNSYM);
            let rels = self.section_bytes(&jmprel);
            let syms = dynsym.and_then(|sh| self.section_bytes(sh));
            let names = dynsym.and_then(|sh| self.section_bytes_at(&shdrs, sh.sh_link as usize));
            if let (Some(rels), Some(syms), Some(names)) = (rels, syms, names) {
                // A `Rela` carries an addend the `Rel` form leaves in the
                // field; only its `r_info`, which holds the symbol index in
                // its high bits, is read here.
                let rel_size = if jmprel.sh_type == SHT_RELA {
                    c.relasize
                } else {
                    c.relsize
                };
                let stride = (jmprel.sh_entsize as usize).max(rel_size);
                let sym_stride =
                    dynsym.map_or(sym_size, |sh| (sh.sh_entsize as usize).max(sym_size));
                for (i, rel) in (0..)
                    .map_while(|i| {
                        let at = i * stride;
                        rels.get(at..at.checked_add(rel_size)?)
                    })
                    .enumerate()
                {
                    let r_info = endian.read_uint(rel, word, word);
                    let sym = if is_64 { r_info >> 32 } else { r_info >> 8 } as usize;
                    let at = sym.saturating_mul(sym_stride);
                    let Some(entry) = at
                        .checked_add(sym_size)
                        .and_then(|end| syms.get(at..end))
                    else {
                        continue;
                    };
                    let name_off = endian.read_uint(entry, 0, 4) as usize;
                    if let Some(slot) = slot_for(&mut bound, &first_bytes, names, name_off) {
                        let stub = plt
                            .sh_addr
                            .saturating_add(plt_header)
                            .saturating_add(i as u64 * plt_entry);
                        bind(slot, Binding::Addr(stub));
                    }
                }
            }
        }

        Ok(TargetSymbols { bound })
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
    /// `base`.
    ///
    /// The segment needs a program header, and neither way of getting one
    /// relocates the program-header table itself:
    ///
    /// - An existing `PT_NOTE` entry is rewritten as a `PT_LOAD`, when the
    ///   target has one to spare.
    /// - Otherwise the table is grown by one entry into the padding that
    ///   follows it — many statically linked targets carry no `PT_NOTE` at all.
    ///   A target with no such padding is refused.
    ///
    /// `base` must be aligned to the target's page size (see
    /// [`injected_base`](Self::injected_base)); the file is padded to the same
    /// granularity before the blob is appended so the loader's
    /// `p_vaddr ≡ p_offset (mod p_align)` requirement holds. Now that `blob`'s
    /// length is known, this is also where `base` is checked for being free: a
    /// base whose segment would share a page with one of the target's own
    /// `PT_LOAD`s replaces that mapping rather than adding to it, and is
    /// refused.
    pub fn inject_segment(&mut self, base: u64, blob: &[u8]) -> Result<()> {
        use goblin::elf::program_header::{PF_R, PF_X, PT_LOAD, PT_NOTE, PT_PHDR};

        let align = self.load_align()?;
        if base % align != 0 {
            return Err(anyhow!(
                "injected base {base:#x} is not aligned to the target's page size \
                 ({align:#x})"
            ));
        }
        self.check_base_is_free(base, blob.len() as u64, align)?;
        let endian = self.endian;
        let c = self.class();
        let phentsize = c.phentsize;

        // Where the program-header table lives. The no-PT_NOTE path below
        // appends an entry to the table and bumps the count.
        let phoff = endian.read_uint(&self.data, c.e_phoff, c.word) as usize;
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
        // table itself and the first file byte the target already occupies —
        // the same trick linkers use to leave slack for PT_NOTE in the first
        // place. This never touches bytes any parser needs, since it strictly
        // stays inside currently-unclaimed padding.
        let note_off = match note_off {
            Some(off) => off,
            None => {
                let table_end = (phoff + phnum * phentsize) as u64;
                // See `first_claimed_after` for why this cannot be answered
                // from the program headers' own `p_offset`s.
                let Some(first_claimed) = self.first_claimed_after(table_end)? else {
                    return Err(anyhow!(
                        "no PT_NOTE segment to convert, and the target has no section \
                         headers and no segment past its program-header table, so \
                         there is no way to tell whether the bytes after the table \
                         are padding or content"
                    ));
                };
                // A claim past the end of the file cannot vouch for bytes that
                // are not there.
                let gap = first_claimed.min(self.data.len() as u64) - table_end;
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
                endian.write_uint(&mut self.data, c.e_phnum, 2, new_phnum as u64);
                // PT_PHDR describes the table, so it has to grow with it.
                // Leaving it behind makes it contradict `e_phnum`, and every
                // reader that trusts it -- `dl_iterate_phdr` in the target
                // itself included -- stops one entry short of the segment just
                // added.
                let table_size = (new_phnum * phentsize) as u64;
                for ph in phdrs.iter().filter(|ph| ph.p_type == PT_PHDR) {
                    let off = ph.entry_off;
                    endian.write_uint(&mut self.data, off + c.p_filesz, c.word, table_size);
                    endian.write_uint(&mut self.data, off + c.p_memsz, c.word, table_size);
                }
                table_end as usize
            }
        };

        // Append the blob at a file offset congruent to `base` modulo `align`.
        let file_off = align_up(self.data.len() as u64, align);
        self.data.resize(file_off as usize, 0);
        self.data.extend_from_slice(blob);
        let len = blob.len() as u64;

        // Rewrite the note phdr in place as a PT_LOAD covering the appended blob.
        let d = &mut self.data;
        endian.write_uint(d, note_off, 4, PT_LOAD as u64);
        endian.write_uint(d, note_off + c.p_flags, 4, (PF_R | PF_X) as u64);
        for (field, value) in [
            (c.p_offset, file_off),
            (c.p_vaddr, base),
            (c.p_paddr, base),
            (c.p_filesz, len),
            (c.p_memsz, len),
            (c.p_align, align),
        ] {
            endian.write_uint(d, note_off + field, c.word, value);
        }

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

    /// The four-byte NOP of the fixed-width architectures is written a whole
    /// instruction at a time, in the target's byte order, and a tail that does
    /// not end on an instruction boundary holds as much of one as fits -- the
    /// same bytes the one-instruction-at-a-time fill produced.
    #[test]
    fn test_patch_bytes_nop_fill_keeps_word_alignment_and_a_partial_tail() {
        let nop_le = 0xd503_201fu32.to_le_bytes();
        let nop_be = 0xd503_201fu32.to_be_bytes();
        for (endian, nop) in [(Endian::Little, nop_le), (Endian::Big, nop_be)] {
            let mut binary = Binary::new(vec![0xcc; 64], Architecture::Arm64, endian);

            // Two bytes of code, then a 13-byte tail: three whole NOPs and the
            // first byte of a fourth.
            binary.patch_bytes(8, 15, &[0xAA, 0xBB]).unwrap();

            let mut want = vec![0xAA, 0xBB];
            want.extend_from_slice(&nop);
            want.extend_from_slice(&nop);
            want.extend_from_slice(&nop);
            want.push(nop[0]);
            assert_eq!(&binary.data[8..23], &want[..], "{endian:?}");
            assert_eq!(binary.data[7], 0xcc, "{endian:?}: before the region");
            assert_eq!(binary.data[23], 0xcc, "{endian:?}: past the region");
        }
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

    /// An AArch64 `b` encodes its displacement in instruction units in a signed
    /// 26-bit immediate, so it reaches ±128MB. The range used to be checked
    /// *after* narrowing the displacement to `i32`, which lets one far too large
    /// to encode wrap into the permitted range: an `--inject-base` 16GB away
    /// gives `0x1_0000_0000` instruction units, which narrows to `0` and passes
    /// the check as a branch to the trampoline's own address. `resplice`
    /// reported a successful "trampoline to injected segment" and the patched
    /// binary died on the ELF header it branched into.
    #[test]
    fn arm64_trampoline_rejects_displacement_that_wraps_a_narrower_type() {
        let binary = Binary::new(vec![0; 0x1000], Architecture::Arm64, Endian::Little);

        // The case a real run produced: a splice at 0x400764 with
        // `--inject-base 0x400400000`, 0x3fffff8ac bytes away. That is
        // 0xfffffe2b instruction units, which narrows to a small *negative*
        // `i32` and used to encode a backwards branch into the ELF header.
        let err = binary.jump_bytes(0x400764, 0x4_0040_0010).unwrap_err();
        assert!(err.to_string().contains("out of reach"), "got: {err}");

        // 16GB either way narrows to a displacement of exactly zero, which the
        // old check read as a branch to the trampoline's own address.
        let from = 0x8_0000_0000u64;
        for to in [from + 0x4_0000_0000, from - 0x4_0000_0000] {
            let err = binary.jump_bytes(from, to).unwrap_err();
            assert!(err.to_string().contains("out of reach"), "got: {err}");
        }

        // The edges of what the immediate really holds still encode.
        let far = binary.jump_bytes(from, from + 0x7ff_fffc).unwrap();
        assert_eq!(far, (0x1400_0000u32 | 0x1ff_ffff).to_le_bytes());
        let back = binary.jump_bytes(from, from - 0x800_0000).unwrap();
        assert_eq!(back, (0x1400_0000u32 | 0x200_0000).to_le_bytes());
        // One instruction past each is not encodable.
        assert!(binary.jump_bytes(from, from + 0x800_0000).is_err());
        assert!(binary.jump_bytes(from, from - 0x800_0004).is_err());
    }

    /// Same bug, same shape, on 32-bit ARM: a signed 24-bit immediate reaching
    /// ±32MB, resolved against `pc = from + 8`.
    #[test]
    fn arm_trampoline_rejects_displacement_that_wraps_a_narrower_type() {
        let binary = Binary::new(vec![0; 0x1000], Architecture::Arm, Endian::Little);

        let err = binary.jump_bytes(0x8000, 0x8008 + 0x4_0000_0000).unwrap_err();
        assert!(err.to_string().contains("out of reach"), "got: {err}");

        // Encoded relative to pc = from + 8, at the edges of the immediate.
        let fwd = binary.jump_bytes(0x8000, 0x8008 + 0x1ff_fffc).unwrap();
        assert_eq!(fwd, (0xEA00_0000u32 | 0x7f_ffff).to_le_bytes());
        assert!(binary.jump_bytes(0x8000, 0x8008 + 0x200_0000).is_err());
    }

    /// Both branches count whole instructions, so a target that is not 4-byte
    /// aligned is not reachable at all -- the division used to round it towards
    /// the branch and emit a jump to a different address than the one asked for.
    #[test]
    fn arm_trampolines_reject_a_misaligned_target() {
        for arch in [Architecture::Arm, Architecture::Arm64] {
            let binary = Binary::new(vec![0; 0x1000], arch, Endian::Little);
            let err = binary.jump_bytes(0x400764, 0x400802).unwrap_err();
            assert!(err.to_string().contains("4-byte aligned"), "got: {err}");
        }
    }

    /// One program header, as
    /// `(p_type, p_offset, p_vaddr, p_filesz, p_flags, p_align)`.
    type Phdr = (u64, u64, u64, u64, u64, u64);

    /// A `Binary` of `len` bytes carrying an ELF header of the class implied by
    /// `arch` plus `phdrs`, ready for `inject_segment`.
    fn craft_injectable(endian: Endian, arch: Architecture, len: usize, phdrs: &[Phdr]) -> Binary {
        let is_64 = arch.is_64();
        let l = Class::of(is_64);
        let mut d = vec![0u8; len];
        d[..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
        d[4] = if is_64 { 2 } else { 1 };
        d[5] = match endian {
            Endian::Little => 1,
            Endian::Big => 2,
        };
        d[6] = 1;
        endian.write_uint(&mut d, l.e_phoff, l.word, l.ehsize as u64);
        endian.write_uint(&mut d, l.e_phentsize, 2, l.phentsize as u64);
        endian.write_uint(&mut d, l.e_phnum, 2, phdrs.len() as u64);
        for (i, &(p_type, p_off, p_vaddr, p_filesz, p_flags, p_align)) in phdrs.iter().enumerate() {
            let po = l.ehsize + i * l.phentsize;
            endian.write_uint(&mut d, po, 4, p_type);
            endian.write_uint(&mut d, po + l.p_offset, l.word, p_off);
            endian.write_uint(&mut d, po + l.p_vaddr, l.word, p_vaddr);
            endian.write_uint(&mut d, po + l.p_paddr, l.word, p_vaddr);
            endian.write_uint(&mut d, po + l.p_filesz, l.word, p_filesz);
            endian.write_uint(&mut d, po + l.p_memsz, l.word, p_filesz);
            endian.write_uint(&mut d, po + l.p_flags, 4, p_flags);
            endian.write_uint(&mut d, po + l.p_align, l.word, p_align);
        }
        Binary::new(d, arch, endian)
    }

    /// Give `bin` a section-header table at file offset `shoff` describing a
    /// `SHT_NULL` entry plus one `(sh_type, sh_offset, sh_size)` per element of
    /// `sections`. 64-bit little-endian only, which is all the callers need.
    ///
    /// Real targets all carry section headers, and they are the only thing that
    /// says where the padding after the program-header table ends -- so the
    /// no-`PT_NOTE` fallback cannot be tested honestly without them.
    fn craft_sections(bin: &mut Binary, shoff: u64, sections: &[(u32, u64, u64)]) {
        const SHENTSIZE: usize = 64;
        let endian = bin.endian;
        let shnum = sections.len() + 1;
        assert!(bin.data.len() >= shoff as usize + shnum * SHENTSIZE);
        endian.write_uint(&mut bin.data, 0x28, 8, shoff); // e_shoff
        endian.write_uint(&mut bin.data, 0x3a, 2, SHENTSIZE as u64); // e_shentsize
        endian.write_uint(&mut bin.data, 0x3c, 2, shnum as u64); // e_shnum
        endian.write_uint(&mut bin.data, 0x3e, 2, 0); // e_shstrndx -> the null entry
        for (i, &(sh_type, sh_offset, sh_size)) in sections.iter().enumerate() {
            let so = shoff as usize + (i + 1) * SHENTSIZE;
            endian.write_uint(&mut bin.data, so + 4, 4, sh_type as u64);
            endian.write_uint(&mut bin.data, so + 24, 8, sh_offset);
            endian.write_uint(&mut bin.data, so + 32, 8, sh_size);
        }
    }

    /// A `PT_PHDR` + `PT_LOAD` pair as every dynamically linked executable has
    /// it: the `PT_LOAD` begins at file offset 0, so it spans the ELF header,
    /// the program-header table and everything after.
    fn phdr_and_load(phnum: usize) -> [Phdr; 2] {
        use goblin::elf::program_header::{PT_LOAD, PT_PHDR};
        let table = (phnum * 56) as u64;
        [
            (PT_PHDR as u64, 0x40, 0x400040, table, 4, 8),
            (PT_LOAD as u64, 0, 0x400000, 0x400, 5, 0x1000),
        ]
    }

    /// The file offset of every program header in `bin`.
    fn phdr_offsets(bin: &Binary) -> Vec<usize> {
        let l = bin.class();
        let phoff = bin.endian().read_uint(bin.data(), l.e_phoff, l.word) as usize;
        let phnum = bin.endian().read_uint(bin.data(), l.e_phnum, 2) as usize;
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
        let l = bin.class();

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
        let l = &ELF64;
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
                let layout = Class::of(arch.is_64());
                assert_eq!(o.entry_off, layout.ehsize + i * layout.phentsize);
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
        let l = &ELF64;
        bin.endian.write_uint(&mut bin.data, l.e_phnum, 2, 1000);

        let err = bin.program_headers().unwrap_err().to_string();
        assert!(err.contains("runs past the end"), "{err}");
        // Every entry point that reads the table has to report it, not panic.
        assert!(bin.va_to_offset(0x400000).is_err());
        assert!(bin.injected_base().is_err());
        assert!(bin.inject_segment(INJECT_AT, &PAYLOAD).is_err());
    }

    /// A patch region that starts inside a segment but runs past the end of its
    /// file-backed data must be refused.
    ///
    /// Translating `begin` alone said nothing about the rest of the region, so
    /// the bytes past the segment's `p_filesz` were written at file offsets that
    /// hold something else entirely -- padding, the next segment's content, the
    /// section headers -- while the addresses they were meant for were never
    /// touched. The image is long enough for the write either way, so
    /// `patch_bytes`' own bounds check does not see it.
    #[test]
    fn va_range_to_offset_refuses_a_region_past_its_segment() {
        use goblin::elf::program_header::PT_LOAD;
        // Two segments, each with 0x400 bytes of file-backed data, inside a
        // 0x2000-byte image: everything past the first one's 0x400 bytes is
        // file content that does not belong to its addresses.
        let phdrs: [Phdr; 2] = [
            (PT_LOAD as u64, 0, 0x400000, 0x400, 5, 0x1000),
            (PT_LOAD as u64, 0x1000, 0x500000, 0x400, 6, 0x1000),
        ];
        let bin = craft_injectable(Endian::Little, Architecture::X86_64, 0x2000, &phdrs);

        // A region ending exactly at the segment's end still translates.
        assert_eq!(bin.va_range_to_offset(0x4003f0, 0x10).unwrap(), 0x3f0);
        assert_eq!(bin.va_range_to_offset(0x500000, 0x400).unwrap(), 0x1000);

        // One byte more runs off the segment, and must be reported rather than
        // written to the file byte that follows (which is not at that address).
        let err = bin.va_range_to_offset(0x4003f0, 0x11).unwrap_err().to_string();
        assert!(err.contains("file-backed data only for"), "{err}");
        assert!(err.contains("0x400000..0x400400"), "{err}");

        // Unmapped addresses, and ranges that wrap, are still refused.
        assert!(bin.va_range_to_offset(0x450000, 1).is_err());
        assert!(bin.va_range_to_offset(0x400000, u64::MAX).is_err());
    }

    /// An address in a segment's zero-filled tail -- `.bss` -- has to be
    /// reported as what it is, not as unmapped.
    ///
    /// Splicing a zero-initialized table is a reasonable thing to try, and
    /// `readelf -l` says plainly that its address is mapped: the RW `PT_LOAD`'s
    /// `p_memsz` covers it. Only `p_filesz` does not, because the loader zeroes
    /// `.bss` out of nothing rather than reading it, so the image holds no byte
    /// there to patch. Calling that "not mapped by any PT_LOAD segment" sends
    /// the user to re-check a `begin`/`end` that was right all along.
    #[test]
    fn va_range_to_offset_names_the_bss_tail_rather_than_calling_it_unmapped() {
        use goblin::elf::program_header::PT_LOAD;
        // One R+X segment, and an RW one shaped like a real `.data` + `.bss`:
        // 0x40 bytes of file-backed data at 0x420000, then 0x100 bytes that
        // exist only in memory.
        let phdrs: [Phdr; 2] = [
            (PT_LOAD as u64, 0, 0x400000, 0x400, 5, 0x1000),
            (PT_LOAD as u64, 0x400, 0x420000, 0x40, 6, 0x1000),
        ];
        let mut bin = craft_injectable(Endian::Little, Architecture::X86_64, 0x440, &phdrs);
        let l = &ELF64;
        let memsz_off = l.ehsize + l.phentsize + l.p_memsz;
        bin.endian.write_uint(&mut bin.data, memsz_off, l.word, 0x140);

        // `.data` still translates, right up to the end of the file-backed part.
        assert_eq!(bin.va_range_to_offset(0x420000, 0x40).unwrap(), 0x400);

        // The `.bss` tail does not, and the error says why.
        let err = bin.va_range_to_offset(0x420040, 8).unwrap_err().to_string();
        assert!(err.contains("`.bss`"), "{err}");
        assert!(err.contains("0x420000..0x420140"), "{err}");
        assert!(err.contains("only up to 0x420040"), "{err}");
        assert!(
            !err.contains("not mapped by any PT_LOAD"),
            "the address is mapped -- just not file-backed: {err}"
        );

        // A region straddling the boundary is still reported as the overshoot
        // it is, which is the more specific of the two diagnoses.
        let err = bin.va_range_to_offset(0x420038, 0x10).unwrap_err().to_string();
        assert!(err.contains("file-backed data only for"), "{err}");

        // Past `p_memsz` nothing maps the address at all, and that still says so.
        let err = bin.va_range_to_offset(0x420140, 8).unwrap_err().to_string();
        assert!(err.contains("not mapped by any PT_LOAD"), "{err}");
    }

    /// A segment claiming more file-backed data than the image holds, and one
    /// whose `p_vaddr + p_filesz` overflows, both have to be reported rather
    /// than translated (or panicked on): the header fields come straight out of
    /// a target nothing has vetted.
    #[test]
    fn va_range_to_offset_refuses_headers_the_image_cannot_back() {
        use goblin::elf::program_header::PT_LOAD;
        // 0x400 bytes of claimed file data starting at file offset 0x300, in an
        // image only 0x400 bytes long.
        let phdrs: [Phdr; 1] = [(PT_LOAD as u64, 0x300, 0x400000, 0x400, 5, 0x1000)];
        let bin = craft_injectable(Endian::Little, Architecture::X86_64, 0x400, &phdrs);
        assert_eq!(bin.va_range_to_offset(0x400000, 0x100).unwrap(), 0x300);
        let err = bin.va_range_to_offset(0x400000, 0x101).unwrap_err().to_string();
        assert!(err.contains("run past the end"), "{err}");

        // A segment whose end wraps must not appear to map everything.
        let phdrs: [Phdr; 1] = [(PT_LOAD as u64, 0, u64::MAX - 0xff, 0x400, 5, 0x1000)];
        let bin = craft_injectable(Endian::Little, Architecture::X86_64, 0x400, &phdrs);
        assert!(bin.va_range_to_offset(0x1000, 1).is_err());
        assert!(bin.va_range_to_offset(u64::MAX - 0xff, 0x100).is_err());
    }

    /// Translating a batch against one read of the program-header table has to
    /// give exactly what translating each range on its own gives -- the same
    /// offsets, and the same refusal for the same reason -- or the speedup is
    /// paid for in behaviour. A batch is also what the linker hands over, so an
    /// empty one (an rlib whose splices all trampoline) must not be an error.
    #[test]
    fn va_range_to_offsets_agrees_with_translating_one_at_a_time() {
        use goblin::elf::program_header::PT_LOAD;
        let phdrs: [Phdr; 2] = [
            (PT_LOAD as u64, 0, 0x400000, 0x400, 5, 0x1000),
            (PT_LOAD as u64, 0x400, 0x401000, 0x40, 6, 0x1000),
        ];
        let bin = craft_injectable(Endian::Little, Architecture::X86_64, 0x440, &phdrs);

        assert!(bin.va_range_to_offsets(std::iter::empty()).unwrap().is_empty());

        let ranges = [
            (0x400000, 1),     // start of the first segment
            (0x4003ff, 1),     // its last file-backed byte
            (0x401000, 0x40),  // the whole of the second segment's file data
            (0x400000, 0x401), // runs off the end of the segment mapping it
            (0x500000, 1),     // mapped by nothing
            (u64::MAX, 2),     // wraps the address space
        ];
        let batch = bin.va_range_to_offsets(ranges).unwrap();
        assert_eq!(batch.len(), ranges.len());
        for (got, &(va, len)) in batch.iter().zip(&ranges) {
            match (got, bin.va_range_to_offset(va, len)) {
                (Ok(a), Ok(b)) => assert_eq!(*a, b, "{va:#x}..+{len:#x}"),
                (Err(a), Err(b)) => {
                    assert_eq!(format!("{a:#}"), format!("{b:#}"), "{va:#x}..+{len:#x}")
                }
                (a, b) => panic!("{va:#x}..+{len:#x}: batch {a:?} vs single {b:?}"),
            }
        }
        // Not vacuously equal: the batch really does translate and really does
        // refuse.
        assert_eq!(*batch[0].as_ref().unwrap(), 0);
        assert_eq!(*batch[2].as_ref().unwrap(), 0x400);
        assert!(batch[3].as_ref().unwrap_err().to_string().contains("file-backed"));
        assert!(batch[4].as_ref().unwrap_err().to_string().contains("not mapped"));
        assert!(batch[5].as_ref().unwrap_err().to_string().contains("wraps"));
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

    /// A `PT_PHDR` must not be mistaken for content blocking the growth of the
    /// table it describes. Its `p_offset` *is* the table, so it necessarily
    /// starts below the table's end -- and taking the minimum `p_offset` over
    /// every segment therefore reported "no room" on every dynamically linked
    /// executable, which is to say on the overwhelming majority of targets.
    #[test]
    fn test_inject_segment_grows_phdr_table_past_pt_phdr() {
        use goblin::elf::program_header::PT_PHDR;
        let phdrs = phdr_and_load(2);
        let mut bin = craft_injectable(Endian::Little, Architecture::X86_64, 0x400, &phdrs);
        // Content starts at 0x200, well past the table's end at 0xb0, so there
        // is room for one more 56-byte entry.
        craft_sections(&mut bin, 0x300, &[(1 /* SHT_PROGBITS */, 0x200, 0x100)]);

        bin.inject_segment(INJECT_AT, &PAYLOAD).unwrap();

        injected_phdr(&bin, INJECT_AT, &PAYLOAD);
        let types = phdr_types(&bin);
        assert_eq!(types.len(), 3, "table should have grown");

        // The PT_PHDR has to describe the grown table, not the old one, or it
        // contradicts `e_phnum` and hides the segment just added from every
        // reader that trusts it.
        let l = bin.class();
        let po = phdr_offsets(&bin)
            .into_iter()
            .find(|&po| bin.endian().read_uint(bin.data(), po, 4) == PT_PHDR as u64)
            .expect("PT_PHDR should survive");
        let expected = (types.len() * l.phentsize) as u64;
        assert_eq!(
            bin.endian().read_uint(bin.data(), po + l.p_filesz, l.word),
            expected,
            "PT_PHDR p_filesz"
        );
        assert_eq!(
            bin.endian().read_uint(bin.data(), po + l.p_memsz, l.word),
            expected,
            "PT_PHDR p_memsz"
        );
    }

    /// The padding gap after the program-header table cannot be measured from
    /// segment `p_offset`s: the first `PT_LOAD` of an executable starts at file
    /// offset 0 and covers the headers, the padding *and* the content that
    /// follows, so it says nothing about where that content begins. Reading the
    /// gap that way declared it clear on a target whose `.text` began
    /// immediately after the table, and the new program header was written over
    /// the entry point -- with `resplice` reporting success.
    #[test]
    fn test_inject_segment_refuses_to_overwrite_content_after_the_table() {
        use goblin::elf::program_header::PT_LOAD;
        // A freestanding `-nostdlib -static` executable, which is exactly the
        // no-PT_NOTE shape this fallback exists for: one PT_LOAD from file
        // offset 0, and `.text` right behind the single program header.
        let phdrs: [Phdr; 1] = [(PT_LOAD as u64, 0, 0x400000, 0x400, 5, 0x1000)];
        let mut bin = craft_injectable(Endian::Little, Architecture::X86_64, 0x400, &phdrs);
        // `.text` begins at 0x78, exactly where the table ends: no padding.
        craft_sections(&mut bin, 0x300, &[(1 /* SHT_PROGBITS */, 0x78, 0x100)]);
        let before = bin.data().to_vec();

        let msg = bin
            .inject_segment(INJECT_AT, &PAYLOAD)
            .unwrap_err()
            .to_string();
        assert!(
            msg.contains("no PT_NOTE") && msg.contains("padding"),
            "{msg}"
        );
        assert_eq!(bin.data(), &before[..], "target must be left untouched");
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

    /// Virtual address of the one `.text` section in [`craft_with_symtab`]'s
    /// output, and so the value its nameless section symbol carries.
    const TEXT_ADDR: u64 = 0x0040_1000;

    /// Address of the `.plt` [`craft_elf`] writes, and so the base its stub
    /// addresses are measured from.
    const PLT_ADDR: u64 = 0x0040_2000;

    /// Build a goblin-parseable ELF executable, in the class and byte order
    /// `arch`/`endian` imply, whose `.symtab` holds `syms` — each given as
    /// `(name, st_type, st_value)` and defined in `.text` — and whose
    /// `.dynsym`/`.rela.plt`/`.plt` trio names `imports` as functions reached
    /// through the PLT, in that order.
    ///
    /// Every `.symtab` entry is `STB_GLOBAL`, and a nameless `STT_SECTION`
    /// symbol for `.text` precedes them, as a real target's symbol table has.
    /// Each import is an undefined `.dynsym` entry, which is what a real one
    /// is, so only its `.rela.plt` entry says where its stub is.
    fn craft_elf(
        arch: Architecture,
        endian: Endian,
        syms: &[(&str, u8, u64)],
        imports: &[&str],
    ) -> Binary {
        use goblin::elf::sym::{STT_FUNC, STT_SECTION};
        let is_64 = arch.is_64();
        let l = Class::of(is_64);

        // A string table opens with the NUL that terminates the empty string,
        // which is what `st_name == 0` points at.
        let strings = |names: &[&str]| {
            let mut tab = vec![0u8];
            let offs: Vec<u64> = names
                .iter()
                .map(|name| {
                    let at = tab.len() as u64;
                    tab.extend_from_slice(name.as_bytes());
                    tab.push(0);
                    at
                })
                .collect();
            (tab, offs)
        };

        let push_sym = |out: &mut Vec<u8>, name: u64, st_type: u8, value: u64, shndx: u64| {
            let mut e = vec![0u8; l.symsize];
            endian.write_uint(&mut e, 0, 4, name);
            e[l.st_info] = (1 << 4) | st_type; // STB_GLOBAL
            endian.write_uint(&mut e, l.st_shndx, 2, shndx);
            endian.write_uint(&mut e, l.st_value, l.word, value);
            out.extend_from_slice(&e);
        };

        let (strtab, name_offs) = strings(&syms.iter().map(|s| s.0).collect::<Vec<_>>());
        let mut symtab = vec![0u8; l.symsize]; // the mandatory STN_UNDEF entry
        push_sym(&mut symtab, 0, STT_SECTION, TEXT_ADDR, 1);
        for (i, &(_, st_type, value)) in syms.iter().enumerate() {
            push_sym(&mut symtab, name_offs[i], st_type, value, 1);
        }

        // An import is undefined (`st_shndx == SHN_UNDEF`) with no value, so
        // the PLT pass is the only thing that can bind it.
        let (dynstr, import_offs) = strings(imports);
        let mut dynsym = vec![0u8; l.symsize];
        for &off in &import_offs {
            push_sym(&mut dynsym, off, STT_FUNC, 0, 0);
        }
        // One `R_*_JUMP_SLOT` per import, in stub order; only the symbol index
        // in `r_info` is read, and the i-th entry's stub is the i-th one.
        let mut relaplt = Vec::new();
        for i in 0..imports.len() {
            let mut e = vec![0u8; l.relasize];
            let sym = (i + 1) as u64; // entry 0 of `.dynsym` is STN_UNDEF
            let r_info = if is_64 { sym << 32 } else { sym << 8 };
            endian.write_uint(&mut e, l.word, l.word, r_info);
            relaplt.extend_from_slice(&e);
        }

        let shstrtab = b"\0.text\0.symtab\0.strtab\0.shstrtab\0.dynsym\0.dynstr\0.rela.plt\0.plt\0";
        let align8 = |x: usize| (x + 7) & !7;
        let off_str = align8(if is_64 { 64 } else { 52 });
        let off_sym = align8(off_str + strtab.len());
        let off_dynstr = align8(off_sym + symtab.len());
        let off_dynsym = align8(off_dynstr + dynstr.len());
        let off_rela = align8(off_dynsym + dynsym.len());
        let off_shstr = align8(off_rela + relaplt.len());
        let shoff = align8(off_shstr + shstrtab.len());

        /// One section header's fields, in the order [`craft_elf`] writes them.
        type Fields = (u64, u64, u64, usize, usize, u64, usize);
        // `(sh_name, sh_type, sh_addr, sh_offset, sh_size, sh_link, sh_entsize)`
        // for the null section and the seven real ones. `sh_name` is an offset
        // into `shstrtab` above.
        let shdrs: [Fields; 8] = [
            (0, 0, 0, 0, 0, 0, 0),
            (1, 1, TEXT_ADDR, off_str, 0, 0, 0), // .text, PROGBITS
            (7, 2, 0, off_sym, symtab.len(), 3, l.symsize), // .symtab -> .strtab
            (15, 3, 0, off_str, strtab.len(), 0, 0), // .strtab
            (33, 11, 0, off_dynsym, dynsym.len(), 5, l.symsize), // .dynsym -> .dynstr
            (41, 3, 0, off_dynstr, dynstr.len(), 0, 0), // .dynstr
            (49, 4, 0, off_rela, relaplt.len(), 4, l.relasize), // .rela.plt, SHT_RELA
            (59, 1, PLT_ADDR, off_str, 0, 0, 0),  // .plt
            // 23 is `.shstrtab`, written as the last header below.
        ];
        let shnum = shdrs.len() + 1;

        let mut d = vec![0u8; shoff + shnum * l.shentsize];
        d[..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
        d[4] = if is_64 { 2 } else { 1 };
        d[5] = match endian {
            Endian::Little => 1,
            Endian::Big => 2,
        };
        d[6] = 1; // EV_CURRENT
        let machine = match arch {
            Architecture::X86 => 3,
            Architecture::X86_64 => 0x3e,
            Architecture::Arm => 40,
            Architecture::Arm64 => 183,
            Architecture::Mips | Architecture::Mips64 => 8,
        };
        let (etype_off, machine_off, ehsize_off, phentsize_off) =
            if is_64 { (0x10, 0x12, 0x34, 0x36) } else { (0x10, 0x12, 0x28, 0x2a) };
        endian.write_uint(&mut d, etype_off, 2, 2); // e_type = ET_EXEC
        endian.write_uint(&mut d, machine_off, 2, machine);
        endian.write_uint(&mut d, 0x14, 4, 1); // e_version
        endian.write_uint(&mut d, ehsize_off, 2, if is_64 { 64 } else { 52 });
        endian.write_uint(&mut d, phentsize_off, 2, if is_64 { 56 } else { 32 });
        endian.write_uint(&mut d, l.e_shoff, l.word, shoff as u64);
        endian.write_uint(&mut d, l.e_shentsize, 2, l.shentsize as u64);
        endian.write_uint(&mut d, l.e_shnum, 2, shnum as u64);
        endian.write_uint(&mut d, l.e_shstrndx, 2, (shnum - 1) as u64);
        for (at, bytes) in [
            (off_str, &strtab[..]),
            (off_sym, &symtab),
            (off_dynstr, &dynstr),
            (off_dynsym, &dynsym),
            (off_rela, &relaplt),
            (off_shstr, &shstrtab[..]),
        ] {
            d[at..at + bytes.len()].copy_from_slice(bytes);
        }

        let write_shdr =
            |d: &mut Vec<u8>, i: usize, (name, kind, addr, off, size, link, entsize): Fields| {
                let sh = shoff + i * l.shentsize;
                endian.write_uint(d, sh, 4, name);
                endian.write_uint(d, sh + 4, 4, kind);
                endian.write_uint(d, sh + l.sh_addr, l.word, addr);
                endian.write_uint(d, sh + l.sh_offset, l.word, off as u64);
                endian.write_uint(d, sh + l.sh_size, l.word, size as u64);
                endian.write_uint(d, sh + l.sh_link, 4, link);
                endian.write_uint(d, sh + l.sh_entsize, l.word, entsize as u64);
            };
        for (i, &shdr) in shdrs.iter().enumerate() {
            write_shdr(&mut d, i, shdr);
        }
        write_shdr(&mut d, shnum - 1, (23, 3, 0, off_shstr, shstrtab.len(), 0, 0));

        let (got_arch, got_endian) = Binary::detect_arch(&d).expect("crafted target must parse");
        assert_eq!((got_arch, got_endian), (arch, endian));
        Binary::new(d, arch, endian)
    }

    /// [`craft_elf`] for the ELF64 LE target most of these tests want, with no
    /// PLT imports.
    fn craft_with_symtab(syms: &[(&str, u8, u64)]) -> Binary {
        craft_elf(Architecture::X86_64, Endian::Little, syms, &[])
    }

    /// `st_value` is an address only for some symbol types, and a target that
    /// defines a name as one of the others has to be reported rather than bound.
    ///
    /// A statically linked glibc defines `memcpy`, `memmove`, `memset` and
    /// `strlen` as `STT_GNU_IFUNC`, whose `st_value` is the *resolver*, and
    /// `errno` as `STT_TLS`, whose `st_value` is an offset into the thread-local
    /// block (0x30 in glibc 2.42). Both came back as ordinary addresses, so a
    /// splice that copied a slice called the resolver -- which takes no
    /// arguments and returns a function pointer -- and one that touched `errno`
    /// read absolute address 0x30. 0x30 fits every relocation field there is, so
    /// nothing downstream noticed either.
    #[test]
    fn target_symbols_that_are_not_addresses_are_refused() {
        use goblin::elf::sym::{STT_FUNC, STT_GNU_IFUNC, STT_TLS};
        let bin = craft_with_symtab(&[
            ("plain", STT_FUNC, 0x0040_1100),
            ("strlen", STT_GNU_IFUNC, 0x0040_adf6),
            ("errno", STT_TLS, 0x30),
        ]);

        let found = bin
            .target_symbols(["plain", "absent", "strlen", "errno"])
            .unwrap();

        // An ordinary definition still resolves, and an absent name is still
        // left to the caller to report.
        assert_eq!(found.resolve("plain").unwrap(), Some(0x0040_1100));
        assert_eq!(found.resolve("absent").unwrap(), None);

        let err = found.resolve("strlen").unwrap_err().to_string();
        assert!(err.contains("ifunc") && err.contains("resolver"), "{err}");
        assert!(err.contains("0x40adf6"), "{err}");

        let err = found.resolve("errno").unwrap_err().to_string();
        assert!(err.contains("thread-local"), "{err}");
        assert!(err.contains("0x30"), "{err}");
    }

    /// Only the names asked for are searched for, so a name that was not is a
    /// caller's mistake rather than a name the target lacks. Reporting it as
    /// the latter would resolve a real reference to nothing.
    #[test]
    fn a_name_that_was_not_asked_for_is_an_error() {
        let bin = craft_with_symtab(&[("plain", goblin::elf::sym::STT_FUNC, 0x0040_1100)]);
        let found = bin.target_symbols(["other"]).unwrap();
        let err = found.resolve("plain").unwrap_err().to_string();
        assert!(err.contains("not among the names"), "{err}");
    }

    /// Asking for nothing is not a reason to walk the target's symbol tables,
    /// and must not be a reason to refuse an image whose tables cannot be
    /// parsed either: a replacement that reaches none of the target's own
    /// symbols needs neither.
    #[test]
    fn asking_for_no_names_reads_nothing() {
        let bin = Binary::new(vec![], Architecture::X86_64, Endian::Little);
        let found = bin.target_symbols([]).unwrap();
        assert!(found.resolve("anything").is_err());
    }

    /// An unusable definition must not shadow a usable one for the same name,
    /// in either order: a target that offers both has an address to bind.
    #[test]
    fn a_usable_definition_wins_over_an_unusable_one() {
        use goblin::elf::sym::{STT_FUNC, STT_GNU_IFUNC};
        for syms in [
            [("memcpy", STT_GNU_IFUNC, 0x0040_a000), ("memcpy", STT_FUNC, 0x0040_b000)],
            [("memcpy", STT_FUNC, 0x0040_b000), ("memcpy", STT_GNU_IFUNC, 0x0040_a000)],
        ] {
            let bin = craft_with_symtab(&syms);
            assert_eq!(
                bin.target_symbols(["memcpy"]).unwrap().resolve("memcpy").unwrap(),
                Some(0x0040_b000),
                "{syms:?}"
            );
        }
    }

    /// The class/byte-order combinations a crafted target can be built in, and
    /// which this crate supports.
    const CLASSES: [(Architecture, Endian); 4] = [
        (Architecture::X86_64, Endian::Little),
        (Architecture::X86, Endian::Little),
        (Architecture::Mips64, Endian::Big),
        (Architecture::Mips, Endian::Big),
    ];

    /// The section-header table is read out of the image directly rather than
    /// through a full parse, which preparses every string table the image has
    /// and reads every relocation section -- so that reader has to agree with a
    /// real ELF parse, in both classes and both byte orders. Their
    /// section-header layouts differ by more than field width: ELF32 has no
    /// 8-byte `sh_flags`, so everything after it sits 4 bytes earlier.
    #[test]
    fn section_headers_match_a_full_elf_parse() {
        use goblin::elf::sym::STT_FUNC;
        for (arch, endian) in CLASSES {
            let bin = craft_elf(arch, endian, &[("f", STT_FUNC, 0x40_1100)], &["printf"]);
            let elf = goblin::elf::Elf::parse(bin.data()).unwrap();
            let ours = bin.section_headers().unwrap();

            assert_eq!(ours.len(), elf.section_headers.len(), "{arch:?}");
            for (i, (o, g)) in ours.iter().zip(&elf.section_headers).enumerate() {
                assert_eq!(
                    (
                        o.sh_name as u64,
                        o.sh_type as u64,
                        o.sh_addr,
                        o.sh_offset,
                        o.sh_size,
                        o.sh_link as u64,
                        o.sh_entsize
                    ),
                    (
                        g.sh_name as u64,
                        g.sh_type as u64,
                        g.sh_addr,
                        g.sh_offset,
                        g.sh_size,
                        g.sh_link as u64,
                        g.sh_entsize
                    ),
                    "{arch:?} section header {i}"
                );
            }
            // The names come out of the table `e_shstrndx` points at, which is
            // the other thing a full parse would have handed over.
            assert_eq!(
                bin.section_by_name(&ours, ".plt").map(|sh| sh.sh_addr),
                Some(PLT_ADDR),
                "{arch:?}"
            );
            assert!(bin.section_by_name(&ours, ".absent").is_none(), "{arch:?}");
        }
    }

    /// A header claiming a section-header table the file does not hold must be
    /// refused rather than read past the end of the image, and a table of
    /// entries of the wrong size must not be read at that stride.
    #[test]
    fn section_headers_refuse_a_table_the_image_cannot_back() {
        use goblin::elf::sym::STT_FUNC;
        let good = craft_elf(
            Architecture::X86_64,
            Endian::Little,
            &[("f", STT_FUNC, 0x40_1100)],
            &[],
        );

        let mut bin = Binary::new(good.data.clone(), Architecture::X86_64, Endian::Little);
        bin.endian.write_uint(&mut bin.data, ELF64.e_shnum, 2, 1000);
        let err = bin.section_headers().unwrap_err().to_string();
        assert!(err.contains("runs past the end"), "{err}");
        // Reported by the lookup that needs the table, rather than panicking.
        assert!(bin.target_symbols(["f"]).is_err(), "{err}");

        let mut bin = Binary::new(good.data.clone(), Architecture::X86_64, Endian::Little);
        bin.endian
            .write_uint(&mut bin.data, ELF64.e_shentsize, 2, 48);
        let err = bin.section_headers().unwrap_err().to_string();
        assert!(err.contains("entry size 48"), "{err}");

        // A stripped image has no table and no symbols to find in one, which is
        // not an error: the splice may need nothing from the target.
        let mut bin = Binary::new(good.data, Architecture::X86_64, Endian::Little);
        bin.endian.write_uint(&mut bin.data, ELF64.e_shoff, 8, 0);
        assert!(bin.section_headers().unwrap().is_empty());
        assert_eq!(bin.target_symbols(["f"]).unwrap().resolve("f").unwrap(), None);
    }

    /// `e_shnum` is 16 bits wide, so an image with more sections than it can
    /// express leaves the field zero and keeps the real count in the null
    /// section header's `sh_size`. Reading the field as it stands would find no
    /// sections, and so no symbol tables: every reference into the target would
    /// be reported as a name it does not define.
    #[test]
    fn section_headers_read_a_count_too_large_for_its_field() {
        use goblin::elf::sym::STT_FUNC;
        let bin = craft_with_symtab(&[("helper", STT_FUNC, 0x0040_1100)]);
        let shnum = bin.section_headers().unwrap().len();

        let mut bin = Binary::new(bin.data, Architecture::X86_64, Endian::Little);
        let shoff = bin.endian.read_uint(&bin.data, ELF64.e_shoff, 8) as usize;
        bin.endian.write_uint(&mut bin.data, ELF64.e_shnum, 2, 0);
        bin.endian.write_uint(&mut bin.data, shoff + ELF64.sh_size, 8, shnum as u64); // of [0]

        assert_eq!(bin.section_headers().unwrap().len(), shnum);
        assert_eq!(goblin::elf::Elf::parse(bin.data()).unwrap().section_headers.len(), shnum);
        let found = bin.target_symbols(["helper"]).unwrap();
        assert_eq!(found.resolve("helper").unwrap(), Some(0x0040_1100));

        // A count that the image cannot hold is still refused, not trusted
        // because of where it was read from.
        bin.endian.write_uint(&mut bin.data, shoff + ELF64.sh_size, 8, u64::MAX);
        let err = bin.section_headers().unwrap_err().to_string();
        assert!(err.contains("runs past the end"), "{err}");
    }

    /// `Elf64_Sym` reorders its fields relative to `Elf32_Sym` rather than just
    /// widening them, and the symbol tables are read out of the image directly,
    /// so a definition has to be found in either class and either byte order.
    /// Each is cross-checked against a real ELF parse of the same image.
    #[test]
    fn target_symbols_read_both_classes_and_byte_orders() {
        use goblin::elf::sym::STT_FUNC;
        for (arch, endian) in CLASSES {
            let want = if arch.is_64() { 0x4000_1100 } else { 0x40_1100 };
            let bin = craft_elf(arch, endian, &[("helper", STT_FUNC, want)], &[]);

            // What a full parse makes of the same bytes, as an independent
            // reading of the entry layout.
            let elf = goblin::elf::Elf::parse(bin.data()).unwrap();
            let sym = elf
                .syms
                .iter()
                .find(|s| elf.strtab.get_at(s.st_name) == Some("helper"))
                .unwrap_or_else(|| panic!("{arch:?}: crafted symbol not found by goblin"));
            assert_eq!(sym.st_value, want, "{arch:?}");

            let found = bin.target_symbols(["helper", "absent"]).unwrap();
            assert_eq!(found.resolve("helper").unwrap(), Some(want), "{arch:?}");
            assert_eq!(found.resolve("absent").unwrap(), None, "{arch:?}");
        }
    }

    /// A function the target does not define but *imports* is reached through
    /// its PLT, and the only thing that says which stub belongs to which name
    /// is the order of the PLT's relocations: the i-th one's stub is the i-th,
    /// at `plt + header + i * entry` for this architecture's stub sizes.
    ///
    /// This is how a splice calls libc in a dynamically linked target, and
    /// binding the wrong stub would call a different library function with the
    /// splice's arguments -- reported as a successful splice.
    #[test]
    fn target_symbols_bind_an_import_to_its_plt_stub() {
        use goblin::elf::sym::STT_FUNC;
        for (arch, endian) in CLASSES {
            let imports = ["__libc_start_main", "getpid", "abort"];
            let bin = craft_elf(arch, endian, &[("f", STT_FUNC, 0x40_1100)], &imports);
            let (header, entry) = match arch {
                Architecture::X86 | Architecture::X86_64 => (16, 16),
                Architecture::Arm64 => (32, 16),
                Architecture::Arm => (20, 12),
                Architecture::Mips | Architecture::Mips64 => (32, 16),
            };

            let found = bin.target_symbols(imports).unwrap();
            for (i, name) in imports.iter().enumerate() {
                assert_eq!(
                    found.resolve(name).unwrap(),
                    Some(PLT_ADDR + header + i as u64 * entry),
                    "{arch:?} import {name}"
                );
            }
        }
    }

    /// A string table may *suffix-merge* its entries, so a name can exist only
    /// as the tail of a longer one and a symbol naming it points into the
    /// middle of that entry. Reading from the offset handles that for free,
    /// which an index of entry starts does not.
    #[test]
    fn strtab_str_reads_a_name_the_table_only_holds_as_a_suffix() {
        let tab = b"\0sprintf\0";
        assert_eq!(strtab_str(tab, 1), Some("sprintf"));
        assert_eq!(strtab_str(tab, 2), Some("printf"));
        assert_eq!(strtab_str(tab, 8), Some(""));
        // An offset outside the table, an unterminated tail, and bytes that are
        // not a string are all "no name here" rather than a panic: the table is
        // read out of a file nothing has vetted.
        assert_eq!(strtab_str(tab, 9), None);
        assert_eq!(strtab_str(b"printf", 0), None);
        assert_eq!(strtab_str(b"\0\xff\xfe\0", 1), None);
    }

    /// Symbols are rejected by the first byte of their name before the name is
    /// read, so names that share one must still be told apart: a prefix of a
    /// name the target defines is not that name, and a longer name that starts
    /// with one asked about is not it either.
    #[test]
    fn names_sharing_a_first_byte_are_not_confused() {
        use goblin::elf::sym::STT_FUNC;
        let bin = craft_with_symtab(&[
            ("printf", STT_FUNC, 0x0040_1100),
            ("print", STT_FUNC, 0x0040_1200),
            ("puts", STT_FUNC, 0x0040_1300),
        ]);
        let found = bin.target_symbols(["print", "pr", "puts", "putsx"]).unwrap();
        assert_eq!(found.resolve("print").unwrap(), Some(0x0040_1200));
        assert_eq!(found.resolve("puts").unwrap(), Some(0x0040_1300));
        assert_eq!(found.resolve("pr").unwrap(), None);
        assert_eq!(found.resolve("putsx").unwrap(), None);
    }

    /// A nameless symbol names the empty string, not nothing: `st_name == 0`
    /// points at the NUL that opens the string table. Every section symbol is
    /// nameless, so the empty name used to bind to whichever section came first
    /// -- and a reference carrying no name (one against `STN_UNDEF`, which
    /// `R_*_NONE` and a damaged object use) resolved to it instead of being
    /// reported as unresolvable.
    #[test]
    fn the_empty_name_binds_nothing() {
        let bin = craft_with_symtab(&[("plain", goblin::elf::sym::STT_FUNC, 0x0040_1100)]);
        assert_eq!(bin.target_symbols([""]).unwrap().resolve("").unwrap(), None);
    }
}
