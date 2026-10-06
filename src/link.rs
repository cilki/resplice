//! Turn a compiled reimplementation rlib into concrete patches for a target
//! binary.
//!
//! A `#[Splice]` function is emitted into a `.rspl.<begin>.<end>` section, but
//! its machine code is rarely self-contained: it may `call` a helper, read a
//! `static`, or call a libc function. Those references appear as *relocations*
//! against symbols living in other sections (or in the target binary). Copying
//! the section bytes verbatim would leave those references dangling.
//!
//! This module acts as a small static linker for that job:
//!
//! 1. Read every object member of the rlib and find its splice sections.
//! 2. Recursively collect the sections those splices reference (their helpers,
//!    their helpers' `.rodata`, and so on) — the transitive closure.
//! 3. Lay the collected sections out in a fresh region, assign them virtual
//!    addresses, and synthesize GOT slots for `GOTPCREL`-style references.
//! 4. Resolve every relocation — to an injected section, to a symbol the target
//!    already defines or imports, or else error — and apply it.
//! 5. Inject the collected bytes as a new segment and patch each splice's
//!    resolved code into its `[begin, end)` range.

use anyhow::{anyhow, bail, Context, Result};
use object::read::archive::ArchiveFile;
use object::elf::RelocationType;
use object::{
    Object, ObjectSection, ObjectSymbol, RelocationFlags, RelocationTarget, SectionKind,
};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::arch::{self, Endian};
use crate::{Architecture, Binary};

/// A splice that has been applied to the target, for reporting.
#[derive(Debug)]
pub struct Applied {
    pub begin: u64,
    pub end: u64,
    pub code_len: usize,
    /// True when the replacement did not fit in `[begin, end)` and was placed in
    /// the injected segment, with a jump written at `begin` to reach it.
    pub trampoline: bool,
}

/// One patch to write into the target: `code` at `[begin, end)`, NOP-filled.
struct Patch {
    begin: u64,
    end: u64,
    code: Vec<u8>,
    trampoline: bool,
}

/// Read `rlib_path`, resolve every splice's relocations against `binary` (and
/// the sections the splices reference), inject the referenced code/data as a new
/// segment, and patch each splice into place.
pub fn link_rlib(binary: &mut Binary, rlib_path: &Path) -> Result<Vec<Applied>> {
    let data = fs::read(rlib_path)
        .with_context(|| format!("failed to read rlib {rlib_path:?}"))?;
    let archive = ArchiveFile::parse(&*data).context("failed to parse rlib archive")?;

    // Virtual address the injected segment will be mapped at. All injected
    // sections across all object members share this single segment (there is
    // only one PT_NOTE slot to convert), so `base + blob.len()` is always the
    // next free virtual address.
    let base = binary.injected_base()?;

    let mut blob: Vec<u8> = Vec::new();
    let mut patches: Vec<Patch> = Vec::new();

    for member in archive.members() {
        let member = member.context("failed to read archive member")?;
        let member_data = member.data(&*data).context("failed to read member data")?;

        // Skip members that are not object files (e.g. the rmeta blob).
        let obj = match object::File::parse(member_data) {
            Ok(obj) => obj,
            Err(_) => continue,
        };
        if !obj.sections().any(|s| is_splice_section(&s)) {
            continue;
        }
        link_object(&obj, binary, base, &mut blob, &mut patches)?;
    }

    // Two splices claiming overlapping ranges cannot both survive: each patch
    // NOP-fills the whole of its own region, so whichever is written last erases
    // part of the other's code. Nothing downstream would notice, and the result
    // depends on the order the rlib's sections happened to be visited in, so
    // refuse rather than emit a silently corrupt binary. Checked before anything
    // is written. Sorting also makes the report below independent of that order.
    patches.sort_by_key(|p| (p.begin, p.end));
    for pair in patches.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        if b.begin < a.end {
            bail!(
                "splices {:#x}..{:#x} and {:#x}..{:#x} overlap; each one NOP-fills its \
                 whole region, so they would overwrite each other",
                a.begin,
                a.end,
                b.begin,
                b.end
            );
        }
    }

    if !blob.is_empty() {
        binary.inject_segment(base, &blob)?;
    }

    let mut applied = Vec::new();
    for patch in patches {
        let offset = binary.va_to_offset(patch.begin)?;
        let region = (patch.end - patch.begin) as usize;
        binary.patch_bytes(offset, region, &patch.code)?;
        applied.push(Applied {
            begin: patch.begin,
            end: patch.end,
            code_len: patch.code.len(),
            trampoline: patch.trampoline,
        });
    }
    Ok(applied)
}

/// Owned copy of a section we may need to inject or patch.
struct Sec {
    name: String,
    align: u64,
    writable: bool,
    bytes: Vec<u8>,
    relocs: Vec<Rel>,
    /// `Some((begin, end))` for a pinned `.rspl.*` section.
    splice: Option<(u64, u64)>,
}

/// Owned copy of a relocation.
struct Rel {
    offset: u64,
    r_type: RelocationType,
    sym: usize,
    addend: i64,
    /// True for REL relocations (ARM, MIPS o32) whose addend is stored in the
    /// instruction field rather than carried explicitly.
    has_implicit: bool,
}

/// Owned copy of the parts of a symbol we need to resolve a relocation.
struct SymInfo {
    name: String,
    /// Section index the symbol is defined in, if any.
    section: Option<usize>,
    /// Value (offset within its section, or absolute value).
    value: u64,
    undefined: bool,
}

fn is_splice_section<'a, S: ObjectSection<'a>>(section: &S) -> bool {
    matches!(section.name(), Ok(n) if n.starts_with(".rspl."))
}

/// Whether a section is still writable once the target is running, and so
/// cannot live in the injected segment (which is mapped read-only + executable).
///
/// `SHF_WRITE` alone is not the answer. RELRO sections (`.data.rel.ro*`) carry
/// *immutable* data that merely contains pointers: they are writable only for as
/// long as the dynamic linker needs to relocate them, and are mapped read-only
/// before any user code runs. `resplice` resolves those relocations statically,
/// so by the time the target executes there is nothing left to write — they are
/// read-only data and belong in the injected segment like any other `.rodata`.
///
/// rustc puts an immutable `static` holding a reference (`static NAMES: [&T; 2]`)
/// there, as well as the `core::panic::Location` records behind a bounds check,
/// so rejecting them would turn ordinary Rust into an error.
fn is_runtime_writable<'a, S: ObjectSection<'a>>(section: &S) -> bool {
    match section.kind() {
        SectionKind::Data | SectionKind::UninitializedData => !matches!(
            section.name(),
            Ok(n) if n == ".data.rel.ro" || n.starts_with(".data.rel.ro.")
        ),
        _ => false,
    }
}

/// Parse a `.rspl.<begin>.<end>` section name into its `(begin, end)` range.
///
/// `end <= begin` is refused. `#[Splice]` already rejects such a range at
/// compile time, but the name is read back out of an arbitrary rlib file, and
/// `end - begin` is used as a region *length* throughout: an unsigned underflow
/// there yields a region spanning most of the address space, which slips past
/// `patch_bytes`' bounds check by overflowing it.
fn splice_range(name: &str) -> Result<Option<(u64, u64)>> {
    let Some(rest) = name.strip_prefix(".rspl.") else {
        return Ok(None);
    };
    let parsed = rest.split_once('.').and_then(|(b, e)| {
        Some((
            u64::from_str_radix(b, 16).ok()?,
            u64::from_str_radix(e, 16).ok()?,
        ))
    });
    let Some((begin, end)) = parsed else {
        bail!("splice section {name:?} does not name a `.rspl.<begin>.<end>` range");
    };
    if end <= begin {
        bail!(
            "splice section {name:?} has an empty or reversed range \
             ({begin:#x}..{end:#x}); `end` must be past `begin`"
        );
    }
    Ok(Some((begin, end)))
}

/// Verify that an rlib object member targets the same architecture and byte
/// order as the binary being patched (the architecture implies the pointer
/// width). Splicing wrong-arch machine code in would silently corrupt the
/// target, so this is a hard gate.
fn verify_arch(obj: &object::File, target: &Binary) -> Result<()> {
    use object::Architecture as OA;
    let obj_arch = match obj.architecture() {
        OA::I386 => Architecture::X86,
        OA::X86_64 => Architecture::X86_64,
        OA::Arm => Architecture::Arm,
        OA::Aarch64 => Architecture::Arm64,
        OA::Mips => Architecture::Mips,
        OA::Mips64 => Architecture::Mips64,
        other => bail!("rlib has unsupported architecture {other:?}"),
    };
    let obj_endian = match obj.endianness() {
        object::Endianness::Little => Endian::Little,
        object::Endianness::Big => Endian::Big,
    };

    if obj_arch != target.architecture() || obj_endian != target.endian() {
        bail!(
            "rlib is {} but target is {}",
            arch::describe(obj_arch, obj_endian),
            arch::describe(target.architecture(), target.endian()),
        );
    }
    Ok(())
}

/// The paired `R_MIPS_LO16` addend for an `R_MIPS_HI16`, read from the original
/// section bytes (the low half of the first matching-symbol LO16 instruction).
fn find_pair_lo(arch: Architecture, sec: &Sec, sym: usize, endian: Endian) -> Option<i64> {
    sec.relocs
        .iter()
        .find(|r| {
            arch::is_mips_lo16(arch, r.r_type)
                && r.sym == sym
                // The instruction is read straight out of the section, so an
                // offset that does not hold a whole instruction is no pair.
                && r.offset.saturating_add(4) <= sec.bytes.len() as u64
        })
        .map(|r| arch::mips_lo16_addend(endian, &sec.bytes, r.offset as usize))
}

/// Link one object member into the shared injected blob + patch list.
fn link_object(
    obj: &object::File,
    target: &Binary,
    base: u64,
    blob: &mut Vec<u8>,
    patches: &mut Vec<Patch>,
) -> Result<()> {
    verify_arch(obj, target)?;
    let arch = target.architecture();
    let endian = target.endian();
    let is_64 = target.is_64();

    // Extract everything we need into owned structures so the borrow of `obj`
    // does not outlive this block.
    let mut secs: HashMap<usize, Sec> = HashMap::new();
    let mut syms: HashMap<usize, SymInfo> = HashMap::new();

    for section in obj.sections() {
        let idx = section.index().0;
        let name = section.name().unwrap_or("").to_string();
        let mut relocs = Vec::new();
        for (offset, reloc) in section.relocations() {
            let r_type = match reloc.flags() {
                RelocationFlags::Elf { r_type } => r_type,
                _ => return Err(anyhow!("non-ELF relocation in {name}")),
            };
            let sym = match reloc.target() {
                RelocationTarget::Symbol(i) => i.0,
                other => {
                    return Err(anyhow!("unsupported relocation target {other:?} in {name}"))
                }
            };
            relocs.push(Rel {
                offset,
                r_type,
                sym,
                addend: reloc.addend(),
                has_implicit: reloc.has_implicit_addend(),
            });

            // Record the referenced symbol.
            if let std::collections::hash_map::Entry::Vacant(slot) = syms.entry(sym) {
                let s = obj
                    .symbol_by_index(object::SymbolIndex(sym))
                    .map_err(|e| anyhow!("bad symbol index {sym} in {name}: {e}"))?;
                slot.insert(SymInfo {
                    name: s.name().unwrap_or("").to_string(),
                    section: s.section_index().map(|i| i.0),
                    value: s.address(),
                    undefined: s.is_undefined(),
                });
            }
        }

        secs.insert(
            idx,
            Sec {
                splice: splice_range(&name)?,
                name,
                align: section.align().max(1),
                writable: is_runtime_writable(&section),
                bytes: section.data().unwrap_or(&[]).to_vec(),
                relocs,
            },
        );
    }

    // Transitive closure: starting from the splice sections, pull in every
    // section reachable through a relocation. Splice sections are pinned to
    // their `begin` address; everything else is injected.
    let splice_idxs: Vec<usize> = secs
        .iter()
        .filter(|(_, s)| s.splice.is_some())
        .map(|(&i, _)| i)
        .collect();

    let mut inject_order: Vec<usize> = Vec::new();
    let mut included: Vec<usize> = splice_idxs.clone();
    let mut work: Vec<usize> = splice_idxs.clone();
    while let Some(si) = work.pop() {
        let reloc_syms: Vec<usize> = secs[&si].relocs.iter().map(|r| r.sym).collect();
        for sym in reloc_syms {
            if let Some(secj) = syms.get(&sym).and_then(|s| s.section) {
                if !included.contains(&secj) {
                    included.push(secj);
                    inject_order.push(secj);
                    work.push(secj);
                }
            }
        }
    }

    // A splice whose code fits its `[begin, end)` region is pinned there and
    // patched directly. One that is too large is relocated into the injected
    // segment, with a jump written at `begin` to reach it (a trampoline).
    let mut oversized: Vec<usize> = Vec::new();
    let mut sec_va: HashMap<usize, u64> = HashMap::new();
    for &si in &splice_idxs {
        let (begin, end) = secs[&si].splice.unwrap();
        if secs[&si].bytes.len() <= (end - begin) as usize {
            sec_va.insert(si, begin);
        } else {
            oversized.push(si);
        }
    }

    // Lay the injected referenced sections, then any relocated oversized
    // splices, into the shared blob and assign each a virtual address.
    let mut inj_blob_off: HashMap<usize, usize> = HashMap::new();
    let blob_layout = inject_order.iter().chain(oversized.iter());
    for &si in blob_layout {
        let sec = &secs[&si];
        if sec.writable {
            return Err(anyhow!(
                "splice references writable section {:?}; injecting writable data \
                 (mutable statics/.bss) is not yet supported",
                sec.name
            ));
        }
        pad_to(blob, sec.align);
        let va = base + blob.len() as u64;
        sec_va.insert(si, va);
        inj_blob_off.insert(si, blob.len());
        blob.extend_from_slice(&sec.bytes);
    }

    // Synthesize a pointer-sized GOT slot per symbol referenced by a GOT-style
    // relocation, so an indirect load through the GOT reaches a real pointer.
    let got_width = arch::got_slot_width(is_64);
    let mut got: HashMap<usize, (u64, usize)> = HashMap::new();
    for &si in &included {
        for r in &secs[&si].relocs {
            if arch::needs_got(arch, r.r_type) && !got.contains_key(&r.sym) {
                pad_to(blob, got_width as u64);
                let va = base + blob.len() as u64;
                got.insert(r.sym, (va, blob.len()));
                blob.resize(blob.len() + got_width, 0);
            }
        }
    }

    // Resolve a symbol index to its final virtual address.
    let resolve = |sym: usize| -> Result<u64> {
        let s = syms
            .get(&sym)
            .ok_or_else(|| anyhow!("relocation references unknown symbol {sym}"))?;
        if let Some(secj) = s.section {
            let base_va = sec_va
                .get(&secj)
                .ok_or_else(|| anyhow!("symbol {:?} in un-laid-out section", s.name))?;
            Ok(base_va + s.value)
        } else if s.undefined {
            target
                .resolve_target_symbol(&s.name)?
                .ok_or_else(|| anyhow!("unresolved external symbol {:?}", s.name))
        } else {
            // Absolute symbol.
            Ok(s.value)
        }
    };

    // Fill each synthesized GOT slot with its resolved symbol address (done in
    // its own pass so the field-writing loop below can borrow `blob` freely).
    for (&sym, &(_, goff)) in &got {
        let s = resolve(sym)?;
        endian.write_uint(&mut blob[..], goff, got_width, s);
    }

    // Apply relocations for every included section. A splice that fits is
    // relocated into a private buffer and patched into `[begin, end)` later;
    // everything else (referenced sections, oversized splices) is relocated in
    // place inside the injected blob.
    for &si in &included {
        let sec = &secs[&si];
        let site_base = sec_va[&si];
        let is_fit_splice = sec.splice.is_some() && !oversized.contains(&si);

        let mut out = if is_fit_splice { sec.bytes.clone() } else { Vec::new() };
        let bo = if is_fit_splice { 0 } else { inj_blob_off[&si] };

        for r in &sec.relocs {
            let s = resolve(r.sym)?;
            let p = site_base + r.offset;
            let a = if r.has_implicit { 0 } else { r.addend };
            let got_va = got.get(&r.sym).map(|&(gva, _)| gva);
            let pair_lo = if arch::is_mips_hi16(arch, r.r_type) && r.has_implicit {
                find_pair_lo(arch, sec, r.sym, endian)
            } else {
                None
            };

            // Hand the encoder exactly this section's bytes. Slicing only from
            // `bo` would let a relocation whose offset lies past the end of its
            // own section quietly rewrite whichever section follows it in the
            // injected blob, and `arch::apply` would have no way to tell.
            let buf: &mut [u8] = if is_fit_splice {
                &mut out
            } else {
                &mut blob[bo..bo + sec.bytes.len()]
            };
            arch::apply(
                arch,
                endian,
                r.r_type,
                buf,
                r.offset as usize,
                s,
                a,
                p,
                got_va,
                r.has_implicit,
                pair_lo,
            )
            .with_context(|| {
                format!(
                    "relocation against {:?} in {:?}",
                    syms[&r.sym].name, sec.name
                )
            })?;
        }

        if let Some((begin, end)) = sec.splice {
            if is_fit_splice {
                patches.push(Patch {
                    begin,
                    end,
                    code: out,
                    trampoline: false,
                });
            }
        }
    }

    // Emit a trampoline for each oversized splice: an unconditional jump from
    // `begin` to the relocated code now living in the injected segment.
    for &si in &oversized {
        let (begin, end) = secs[&si].splice.unwrap();
        let dest = sec_va[&si];
        let jump = target.jump_bytes(begin, dest)?;
        if jump.len() > (end - begin) as usize {
            return Err(anyhow!(
                "region {begin:#x}..{end:#x} is too small for a {}-byte trampoline jump",
                jump.len()
            ));
        }
        patches.push(Patch {
            begin,
            end,
            code: jump,
            trampoline: true,
        });
    }

    Ok(())
}

/// Pad `blob` with zeroes so its length is a multiple of `align`.
fn pad_to(blob: &mut Vec<u8>, align: u64) {
    let aligned = (blob.len() as u64 + align - 1) & !(align - 1);
    blob.resize(aligned as usize, 0);
}

#[cfg(test)]
mod tests {
    use super::*;

    const TARGET_BASE: u64 = 0x400000;
    const SPLICE_VA: u64 = 0x401000;
    const FILE_LEN: usize = 0x2000;
    /// Deterministic base for the injected segment given the crafted target
    /// (max PT_LOAD vaddr 0x402000, aligned up + one page gap).
    const INJECT_BASE: u64 = 0x403000;

    /// Build a minimal but goblin-parseable ELF64 executable with one R+X
    /// `PT_LOAD` mapping the whole file at `0x400000` and one `PT_NOTE` (for
    /// `inject_segment` to convert). The splice region lives at `SPLICE_VA`.
    fn crafted_target() -> Vec<u8> {
        let mut d = vec![0u8; FILE_LEN];
        d[..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
        d[4] = 2; // ELFCLASS64
        d[5] = 1; // ELFDATA2LSB
        d[6] = 1; // EV_CURRENT
        let w16 = |d: &mut [u8], at: usize, v: u16| d[at..at + 2].copy_from_slice(&v.to_le_bytes());
        let w32 = |d: &mut [u8], at: usize, v: u32| d[at..at + 4].copy_from_slice(&v.to_le_bytes());
        let w64 = |d: &mut [u8], at: usize, v: u64| d[at..at + 8].copy_from_slice(&v.to_le_bytes());
        w16(&mut d, 0x10, 2); // e_type = ET_EXEC
        w16(&mut d, 0x12, 0x3e); // e_machine = EM_X86_64
        w32(&mut d, 0x14, 1); // e_version
        w64(&mut d, 0x18, SPLICE_VA); // e_entry
        w64(&mut d, 0x20, 64); // e_phoff
        w16(&mut d, 0x34, 64); // e_ehsize
        w16(&mut d, 0x36, 56); // e_phentsize
        w16(&mut d, 0x38, 2); // e_phnum

        // PT_LOAD covering the whole file at 0x400000 (R+X).
        let ph = 64;
        w32(&mut d, ph, 1); // PT_LOAD
        w32(&mut d, ph + 4, 5); // R|X
        w64(&mut d, ph + 8, 0); // p_offset
        w64(&mut d, ph + 16, TARGET_BASE); // p_vaddr
        w64(&mut d, ph + 24, TARGET_BASE); // p_paddr
        w64(&mut d, ph + 32, FILE_LEN as u64); // p_filesz
        w64(&mut d, ph + 40, FILE_LEN as u64); // p_memsz
        w64(&mut d, ph + 48, 0x1000); // p_align

        // PT_NOTE (contents irrelevant; only the phdr slot matters).
        let ph = 64 + 56;
        w32(&mut d, ph, 4); // PT_NOTE
        w32(&mut d, ph + 4, 4); // R
        w64(&mut d, ph + 8, 0x100); // p_offset
        w64(&mut d, ph + 16, TARGET_BASE + 0x100);
        w64(&mut d, ph + 24, TARGET_BASE + 0x100);
        w64(&mut d, ph + 32, 0x20);
        w64(&mut d, ph + 40, 0x20);
        w64(&mut d, ph + 48, 4);

        // Original bytes at the splice site (int3 fill) so we can see them change.
        for b in &mut d[0x1000..0x1000 + 0x20] {
            *b = 0xcc;
        }
        d
    }

    fn temp_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "resplice_link_{tag}_{}_{:p}.bin",
            std::process::id(),
            &tag
        ))
    }

    fn load(bytes: &[u8]) -> Binary {
        let path = temp_path("target");
        fs::write(&path, bytes).unwrap();
        let bin = Binary::load(&path).unwrap();
        fs::remove_file(&path).ok();
        bin
    }

    /// Wrap object bytes in a minimal GNU `ar` archive.
    fn ar_wrap(name: &str, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::from(&b"!<arch>\n"[..]);
        let mut header = format!("{:<16}", format!("{name}/"));
        header.push_str(&format!("{:<12}{:<6}{:<6}{:<8}{:<10}`\n", 0, 0, 0, "644", data.len()));
        assert_eq!(header.len(), 60);
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(data);
        if data.len() % 2 == 1 {
            out.push(b'\n');
        }
        out
    }

    #[test]
    fn test_va_to_offset() {
        let bin = load(&crafted_target());
        assert_eq!(bin.va_to_offset(SPLICE_VA).unwrap(), 0x1000);
        assert_eq!(bin.va_to_offset(TARGET_BASE).unwrap(), 0);
        assert!(bin.va_to_offset(0x500000).is_err());
        assert_eq!(bin.injected_base().unwrap(), INJECT_BASE);
    }

    #[test]
    fn test_inject_segment_converts_note() {
        use goblin::elf::program_header::{PT_LOAD, PT_NOTE};
        let mut bin = load(&crafted_target());
        let base = bin.injected_base().unwrap();
        let payload = [0xde, 0xad, 0xbe, 0xef, 0x11, 0x22];
        bin.inject_segment(base, &payload).unwrap();

        let elf = goblin::elf::Elf::parse(bin.data()).unwrap();
        // The PT_NOTE was consumed; a PT_LOAD now maps our payload at `base`.
        assert!(!elf.program_headers.iter().any(|ph| ph.p_type == PT_NOTE));
        let seg = elf
            .program_headers
            .iter()
            .find(|ph| ph.p_type == PT_LOAD && ph.p_vaddr == base)
            .expect("injected PT_LOAD");
        assert_eq!(seg.p_filesz, payload.len() as u64);
        let off = seg.p_offset as usize;
        assert_eq!(&bin.data()[off..off + payload.len()], &payload);
    }

    const HELPER_CODE: [u8; 6] = [0xb8, 0x2a, 0x00, 0x00, 0x00, 0xc3]; // mov eax,42; ret
    // A 13-byte splice: call rel32 (operand@1); lea rax,[rip+d32] (operand@8); ret.
    const SPLICE_CODE: [u8; 13] = [
        0xe8, 0, 0, 0, 0, // call helper
        0x48, 0x8d, 0x05, 0, 0, 0, 0, // lea rax, [rip + TABLE]
        0xc3, // ret
    ];

    fn table_data() -> [u8; 16] {
        std::array::from_fn(|i| (i as u8) + 1)
    }

    /// Build an rlib with a splice covering `[SPLICE_VA, splice_end)` that calls
    /// a `helper` (PLT32, section symbol) and reads a `TABLE` (PC32, data
    /// symbol). Both referenced sections must be pulled in and injected.
    fn build_rlib(splice_end: u64) -> Vec<u8> {
        use object::write::{Object, Relocation, Symbol, SymbolSection as WSection};
        use object::{
            Architecture, BinaryFormat, Endianness, RelocationFlags, SectionKind, SymbolFlags,
            SymbolKind, SymbolScope,
        };

        let mut obj = Object::new(BinaryFormat::Elf, Architecture::X86_64, Endianness::Little);

        let helper_sec = obj.add_section(vec![], b".text.helper".to_vec(), SectionKind::Text);
        obj.append_section_data(helper_sec, &HELPER_CODE, 1);
        let helper_sym = obj.section_symbol(helper_sec);

        let table = table_data();
        let rodata = obj.add_section(vec![], b".rodata.table".to_vec(), SectionKind::ReadOnlyData);
        obj.append_section_data(rodata, &table, 1);
        let table_sym = obj.add_symbol(Symbol {
            name: b"TABLE".to_vec(),
            value: 0,
            size: table.len() as u64,
            kind: SymbolKind::Data,
            scope: SymbolScope::Compilation,
            weak: false,
            section: WSection::Section(rodata),
            flags: SymbolFlags::None,
        });

        let name = format!(".rspl.{SPLICE_VA:x}.{splice_end:x}").into_bytes();
        let rspl = obj.add_section(vec![], name, SectionKind::Text);
        obj.append_section_data(rspl, &SPLICE_CODE, 16);
        for (offset, symbol, r_type) in [
            (1, helper_sym, object::elf::R_X86_64_PLT32),
            (8, table_sym, object::elf::R_X86_64_PC32),
        ] {
            obj.add_relocation(
                rspl,
                Relocation {
                    offset,
                    symbol,
                    addend: -4,
                    flags: RelocationFlags::Elf { r_type },
                },
            )
            .unwrap();
        }

        ar_wrap("splice.o", &obj.write().unwrap())
    }

    fn run_link(rlib: &[u8]) -> (Binary, Vec<Applied>) {
        let rlib_path = temp_path("rlib");
        fs::write(&rlib_path, rlib).unwrap();
        let mut bin = load(&crafted_target());
        let applied = link_rlib(&mut bin, &rlib_path).unwrap();
        fs::remove_file(&rlib_path).ok();
        (bin, applied)
    }

    /// Read the resolved call/lea targets out of a copy of `SPLICE_CODE` at file
    /// offset `code_off` (whose site is at virtual address `site_va`) and assert
    /// they point at the injected helper code and table data.
    fn assert_refs_resolved(bin: &Binary, code_off: usize, site_va: u64) {
        let d = bin.data();
        assert_eq!(d[code_off], 0xe8, "call opcode");
        assert_eq!(d[code_off + 5], 0x48, "lea opcode");

        let d1 = i32::from_le_bytes(d[code_off + 1..code_off + 5].try_into().unwrap());
        let helper_va = (site_va as i64 + 1 + 4 + d1 as i64) as u64;
        let d2 = i32::from_le_bytes(d[code_off + 8..code_off + 12].try_into().unwrap());
        let table_va = (site_va as i64 + 8 + 4 + d2 as i64) as u64;

        assert!(helper_va >= INJECT_BASE, "helper not in injected segment");
        assert!(table_va >= INJECT_BASE, "table not in injected segment");
        let hoff = bin.va_to_offset(helper_va).unwrap();
        assert_eq!(&bin.data()[hoff..hoff + HELPER_CODE.len()], &HELPER_CODE);
        let toff = bin.va_to_offset(table_va).unwrap();
        assert_eq!(&bin.data()[toff..toff + table_data().len()], &table_data());
    }

    #[test]
    fn test_link_rlib_resolves_call_and_data_refs() {
        // Region (0x20) comfortably holds the 13-byte splice: patched directly.
        let (bin, applied) = run_link(&build_rlib(SPLICE_VA + 0x20));

        assert_eq!(applied.len(), 1);
        assert!(!applied[0].trampoline);
        assert_eq!((applied[0].begin, applied[0].end), (SPLICE_VA, SPLICE_VA + 0x20));

        // Patched directly at file offset 0x1000 (VA 0x401000).
        assert_refs_resolved(&bin, 0x1000, SPLICE_VA);
    }

    #[test]
    fn test_link_rlib_trampolines_oversized_splice() {
        // Region is only 8 bytes — smaller than the 13-byte splice — so the code
        // is relocated into the injected segment and reached via a jump.
        let (bin, applied) = run_link(&build_rlib(SPLICE_VA + 8));

        assert_eq!(applied.len(), 1);
        assert!(applied[0].trampoline);
        assert_eq!((applied[0].begin, applied[0].end), (SPLICE_VA, SPLICE_VA + 8));

        let d = bin.data();
        // A `jmp rel32` sits at the splice site, with the rest NOP-filled.
        assert_eq!(d[0x1000], 0xe9, "jmp opcode");
        assert_eq!(&d[0x1005..0x1008], &[0x90, 0x90, 0x90], "NOP fill");

        // The jump lands on the relocated code inside the injected segment.
        let disp = i32::from_le_bytes(d[0x1001..0x1005].try_into().unwrap());
        let dest = (SPLICE_VA as i64 + 5 + disp as i64) as u64;
        assert!(dest >= INJECT_BASE, "trampoline target not in injected segment");

        // The relocated splice's own references are resolved at its new home.
        let code_off = bin.va_to_offset(dest).unwrap();
        assert_refs_resolved(&bin, code_off, dest);
    }

    /// The crafted x86-64 target with its `e_machine` field changed to
    /// `EM_AARCH64` (183), so `Binary::load` classifies it as aarch64.
    fn crafted_target_aarch64() -> Vec<u8> {
        let mut d = crafted_target();
        d[0x12] = 183;
        d[0x13] = 0;
        d
    }

    /// AArch64 helper (`movz w0,#42; ret`) and splice
    /// (`bl helper; adrp x0,TABLE; add x0,x0,:lo12:TABLE; ret`).
    const A64_HELPER: [u8; 8] = [0x40, 0x05, 0x80, 0x52, 0xc0, 0x03, 0x5f, 0xd6];
    const A64_SPLICE: [u8; 16] = [
        0x00, 0x00, 0x00, 0x94, // bl   helper
        0x00, 0x00, 0x00, 0x90, // adrp x0, TABLE page
        0x00, 0x00, 0x00, 0x91, // add  x0, x0, :lo12:TABLE
        0xc0, 0x03, 0x5f, 0xd6, // ret
    ];

    fn build_rlib_aarch64(splice_end: u64) -> Vec<u8> {
        use object::write::{Object, Relocation, Symbol, SymbolSection as WSection};
        use object::{
            Architecture, BinaryFormat, Endianness, RelocationFlags, SectionKind, SymbolFlags,
            SymbolKind, SymbolScope,
        };

        let mut obj = Object::new(BinaryFormat::Elf, Architecture::Aarch64, Endianness::Little);

        let helper_sec = obj.add_section(vec![], b".text.helper".to_vec(), SectionKind::Text);
        obj.append_section_data(helper_sec, &A64_HELPER, 4);
        let helper_sym = obj.section_symbol(helper_sec);

        let table = table_data();
        let rodata = obj.add_section(vec![], b".rodata.table".to_vec(), SectionKind::ReadOnlyData);
        obj.append_section_data(rodata, &table, 4);
        let table_sym = obj.add_symbol(Symbol {
            name: b"TABLE".to_vec(),
            value: 0,
            size: table.len() as u64,
            kind: SymbolKind::Data,
            scope: SymbolScope::Compilation,
            weak: false,
            section: WSection::Section(rodata),
            flags: SymbolFlags::None,
        });

        let name = format!(".rspl.{SPLICE_VA:x}.{splice_end:x}").into_bytes();
        let rspl = obj.add_section(vec![], name, SectionKind::Text);
        obj.append_section_data(rspl, &A64_SPLICE, 16);
        for (offset, symbol, r_type) in [
            (0, helper_sym, object::elf::R_AARCH64_CALL26),
            (4, table_sym, object::elf::R_AARCH64_ADR_PREL_PG_HI21),
            (8, table_sym, object::elf::R_AARCH64_ADD_ABS_LO12_NC),
        ] {
            obj.add_relocation(
                rspl,
                Relocation {
                    offset,
                    symbol,
                    addend: 0,
                    flags: RelocationFlags::Elf { r_type },
                },
            )
            .unwrap();
        }
        ar_wrap("splice.o", &obj.write().unwrap())
    }

    #[test]
    fn test_link_rlib_aarch64_resolves_refs() {
        let rlib_path = temp_path("rlib_a64");
        fs::write(&rlib_path, build_rlib_aarch64(SPLICE_VA + 0x20)).unwrap();
        let mut bin = load(&crafted_target_aarch64());
        let applied = link_rlib(&mut bin, &rlib_path).unwrap();
        fs::remove_file(&rlib_path).ok();

        assert_eq!(applied.len(), 1);
        assert!(!applied[0].trampoline);

        let d = bin.data();
        let off = 0x1000; // SPLICE_VA

        // BL: sign-extended imm26 * 4 gives the helper's displacement.
        let bl = u32::from_le_bytes(d[off..off + 4].try_into().unwrap());
        assert_eq!(bl & 0xfc00_0000, 0x9400_0000, "BL opcode");
        let imm26 = (((bl & 0x03ff_ffff) as i32) << 6) >> 6;
        let helper_va = (SPLICE_VA as i64 + imm26 as i64 * 4) as u64;
        assert!(helper_va >= INJECT_BASE, "helper not injected");
        let hoff = bin.va_to_offset(helper_va).unwrap();
        assert_eq!(&bin.data()[hoff..hoff + A64_HELPER.len()], &A64_HELPER);

        // ADRP page + ADD lo12 reconstruct the TABLE address.
        let adrp = u32::from_le_bytes(d[off + 4..off + 8].try_into().unwrap());
        let immlo = ((adrp >> 29) & 0x3) as i64;
        let immhi = ((adrp >> 5) & 0x7ffff) as i64;
        let mut pageimm = (immhi << 2) | immlo;
        if pageimm & (1 << 20) != 0 {
            pageimm -= 1 << 21;
        }
        let page = ((SPLICE_VA + 4) & !0xfff) as i64 + pageimm * 0x1000;
        let add = u32::from_le_bytes(d[off + 8..off + 12].try_into().unwrap());
        let lo12 = ((add >> 10) & 0xfff) as i64;
        let table_va = (page + lo12) as u64;
        assert!(table_va >= INJECT_BASE, "table not injected");
        let toff = bin.va_to_offset(table_va).unwrap();
        assert_eq!(&bin.data()[toff..toff + table_data().len()], &table_data());
    }

    #[test]
    fn test_link_rlib_arch_mismatch_errors() {
        // An aarch64 rlib spliced into an x86-64 target must be rejected.
        let rlib_path = temp_path("rlib_mismatch");
        fs::write(&rlib_path, build_rlib_aarch64(SPLICE_VA + 0x20)).unwrap();
        let mut bin = load(&crafted_target()); // x86-64
        let err = link_rlib(&mut bin, &rlib_path).unwrap_err();
        fs::remove_file(&rlib_path).ok();

        let msg = err.to_string();
        assert!(msg.contains("aarch64"), "message was {msg:?}");
        assert!(msg.contains("x86-64"), "message was {msg:?}");
    }

    const STR_DATA: [u8; 4] = *b"abc\0";
    /// An 8-byte splice: `lea rax, [rip + NAMES]` (operand@3); `ret`.
    const PTR_SPLICE_CODE: [u8; 8] = [0x48, 0x8d, 0x05, 0, 0, 0, 0, 0xc3];

    /// An rlib whose splice reads a `static` that itself holds a pointer. rustc
    /// emits such a `static` into a `SHF_WRITE` `.data.rel.ro` section (named
    /// `data_section` here so the same builder can produce genuinely mutable
    /// data), with a relocation at `ptr_reloc_off` for the pointer it contains
    /// (an offset past the end of the 8-byte static makes it malformed).
    fn build_rlib_relro(data_section: &str, ptr_reloc_off: u64) -> Vec<u8> {
        use object::write::{Object, Relocation, Symbol, SymbolSection as WSection};
        use object::{
            Architecture, BinaryFormat, Endianness, RelocationFlags, SectionKind, SymbolFlags,
            SymbolKind, SymbolScope,
        };

        let mut obj = Object::new(BinaryFormat::Elf, Architecture::X86_64, Endianness::Little);

        let strs = obj.add_section(vec![], b".rodata.str1.1".to_vec(), SectionKind::ReadOnlyData);
        obj.append_section_data(strs, &STR_DATA, 1);
        let str_sym = obj.section_symbol(strs);

        // `static NAMES: &[u8; 4] = &STR;` — one pointer, relocated at link time.
        let names_sec =
            obj.add_section(vec![], data_section.as_bytes().to_vec(), SectionKind::Data);
        obj.append_section_data(names_sec, &[0u8; 8], 8);
        let names_sym = obj.add_symbol(Symbol {
            name: b"NAMES".to_vec(),
            value: 0,
            size: 8,
            kind: SymbolKind::Data,
            scope: SymbolScope::Compilation,
            weak: false,
            section: WSection::Section(names_sec),
            flags: SymbolFlags::None,
        });
        obj.add_relocation(
            names_sec,
            Relocation {
                offset: ptr_reloc_off,
                symbol: str_sym,
                addend: 0,
                flags: RelocationFlags::Elf { r_type: object::elf::R_X86_64_64 },
            },
        )
        .unwrap();

        let name = format!(".rspl.{SPLICE_VA:x}.{:x}", SPLICE_VA + 0x20).into_bytes();
        let rspl = obj.add_section(vec![], name, SectionKind::Text);
        obj.append_section_data(rspl, &PTR_SPLICE_CODE, 16);
        obj.add_relocation(
            rspl,
            Relocation {
                offset: 3,
                symbol: names_sym,
                addend: -4,
                flags: RelocationFlags::Elf { r_type: object::elf::R_X86_64_PC32 },
            },
        )
        .unwrap();

        ar_wrap("splice.o", &obj.write().unwrap())
    }

    /// RELRO data is only writable while a dynamic linker relocates it, and
    /// `resplice` does that statically — so it must be injected like `.rodata`,
    /// not rejected as mutable. Rejecting it would fail on an ordinary immutable
    /// `static` that happens to contain a reference.
    #[test]
    fn test_link_rlib_injects_relro_data() {
        let (bin, applied) = run_link(&build_rlib_relro(".data.rel.ro.NAMES", 0));
        assert_eq!(applied.len(), 1);
        assert!(!applied[0].trampoline);

        // Follow `lea rax, [rip + NAMES]` to the injected `static`.
        let d = bin.data();
        assert_eq!(d[0x1000 + 2], 0x05, "lea opcode");
        let disp = i32::from_le_bytes(d[0x1003..0x1007].try_into().unwrap());
        let names_va = (SPLICE_VA as i64 + 7 + disp as i64) as u64;
        assert!(names_va >= INJECT_BASE, "RELRO data not in injected segment");

        // The pointer it holds must have been relocated to the injected string.
        let noff = bin.va_to_offset(names_va).unwrap();
        let str_va = u64::from_le_bytes(d[noff..noff + 8].try_into().unwrap());
        assert!(str_va >= INJECT_BASE, "pointer {str_va:#x} not relocated");
        let soff = bin.va_to_offset(str_va).unwrap();
        assert_eq!(&d[soff..soff + STR_DATA.len()], &STR_DATA);
    }

    /// Data that really is mutable at runtime still has to be refused: the
    /// injected segment is mapped read-only.
    #[test]
    fn test_link_rlib_rejects_mutable_data() {
        let rlib_path = temp_path("rlib_mutable");
        fs::write(&rlib_path, build_rlib_relro(".data.COUNTER", 0)).unwrap();
        let mut bin = load(&crafted_target());
        let err = link_rlib(&mut bin, &rlib_path).unwrap_err();
        fs::remove_file(&rlib_path).ok();

        let msg = err.to_string();
        assert!(msg.contains("writable section"), "message was {msg:?}");
        assert!(msg.contains(".data.COUNTER"), "message was {msg:?}");
    }

    /// Link `rlib` into the crafted target and return the error it must produce.
    fn link_err(tag: &str, rlib: &[u8]) -> String {
        let rlib_path = temp_path(tag);
        fs::write(&rlib_path, rlib).unwrap();
        let mut bin = load(&crafted_target());
        let result = link_rlib(&mut bin, &rlib_path);
        fs::remove_file(&rlib_path).ok();
        match result {
            Ok(applied) => panic!("expected an error, linked {} splice(s)", applied.len()),
            Err(e) => format!("{e:#}"),
        }
    }

    /// `end - begin` is used as the length of the replaced region throughout, so
    /// a reversed range (easy to produce: `begin`/`end` are hand-typed hex
    /// addresses) underflows into a region spanning nearly the whole address
    /// space. That then overflows `patch_bytes`' bounds check instead of failing
    /// it, and the NOP fill runs off the end of the image.
    #[test]
    fn test_link_rlib_rejects_reversed_splice_range() {
        let msg = link_err("rlib_reversed", &build_rlib(SPLICE_VA - 8));
        assert!(msg.contains("reversed"), "message was {msg:?}");
        assert!(msg.contains("0x401000..0x400ff8"), "message was {msg:?}");
    }

    /// A relocation whose field lies outside the section it belongs to must be
    /// refused. Applying it would write into whatever section happens to follow
    /// in the injected blob -- the splice's own data relocated to the wrong
    /// place, with `resplice` reporting success.
    #[test]
    fn test_link_rlib_rejects_out_of_section_relocation() {
        // The `static` is 8 bytes, so an 8-byte pointer field at offset 8 is
        // entirely past its end.
        let msg = link_err("rlib_oob_reloc", &build_rlib_relro(".data.rel.ro.NAMES", 8));
        assert!(msg.contains("runs past the end"), "message was {msg:?}");
        assert!(msg.contains(".data.rel.ro.NAMES"), "message was {msg:?}");
    }

    /// Two splices over the same bytes: each NOP-fills the whole of its region,
    /// so applying both leaves one of them half-overwritten. Which one depends
    /// on section visit order, so this has to be an error, not a coin flip.
    #[test]
    fn test_link_rlib_rejects_overlapping_splices() {
        use object::write::Object;
        use object::{Architecture, BinaryFormat, Endianness, SectionKind};

        let mut obj = Object::new(BinaryFormat::Elf, Architecture::X86_64, Endianness::Little);
        for (begin, end) in [(SPLICE_VA, SPLICE_VA + 0x10), (SPLICE_VA + 8, SPLICE_VA + 0x18)] {
            let name = format!(".rspl.{begin:x}.{end:x}").into_bytes();
            let sec = obj.add_section(vec![], name, SectionKind::Text);
            obj.append_section_data(sec, &[0xc3], 1); // ret
        }
        let rlib = ar_wrap("splice.o", &obj.write().unwrap());

        let msg = link_err("rlib_overlap", &rlib);
        assert!(msg.contains("overlap"), "message was {msg:?}");
        assert!(
            msg.contains("0x401000..0x401010") && msg.contains("0x401008..0x401018"),
            "message was {msg:?}"
        );
    }
}
