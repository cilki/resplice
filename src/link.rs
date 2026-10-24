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
//!    their helpers' `.rodata`, and so on) — the transitive closure. The
//!    closure spans the whole archive: rustc splits a crate into one object
//!    member per codegen unit, so a splice and the helper it calls routinely
//!    live in different members and reference each other by symbol name.
//! 3. Lay the collected sections out in a fresh region, assign them virtual
//!    addresses, and synthesize GOT slots for `GOTPCREL`-style references.
//! 4. Resolve every relocation — to an injected section, to a symbol the target
//!    already defines or imports, or else error — and apply it.
//! 5. Inject the collected bytes as a new segment and patch each splice's
//!    resolved code into its `[begin, end)` range.
//!
//! Nothing is written to the target until step 5, so the checks that a splice
//! cannot survive — two splices claiming overlapping ranges, a `static` whose
//! size does not match its range, a range not wholly backed by one `PT_LOAD` —
//! fail with no output file rather than a half-patched one.

use anyhow::{anyhow, bail, Context, Result};
use object::read::archive::ArchiveFile;
use object::elf::RelocationType;
use object::{
    Object, ObjectSection, ObjectSymbol, RelocationFlags, RelocationTarget, SectionKind,
};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use crate::arch::{self, Endian};
use crate::{Architecture, Binary};

/// The rlib's bytes, however we got hold of them.
///
/// Everything downstream borrows from this -- section names, symbol names, the
/// section contents the closure injects -- so it is one buffer for the whole
/// link, read once and held until the patches are written.
enum Rlib {
    /// The file, mapped. See [`map_rlib`].
    Mapped(memmap2::Mmap),
    /// The file's bytes in memory, for an rlib that cannot be mapped.
    Read(Vec<u8>),
}

impl std::ops::Deref for Rlib {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Rlib::Mapped(map) => map,
            Rlib::Read(data) => data,
        }
    }
}

/// Get at an rlib's bytes, mapping the file rather than reading it where that
/// is possible.
///
/// Only the archive's object members are ever looked at, and they are the
/// smaller part of what rustc writes: a crate compiled the way `cargo build`
/// compiles one carries its `lib.rmeta` in the same archive, 14MB of an 18MB
/// rlib for a crate of a few thousand splices. Reading the archive copies all
/// of that into the process; mapping it means the metadata's pages are never
/// faulted in at all, because nothing touches them. Of the members that *are*
/// read, only the headers, symbol tables and the handful of sections a splice
/// reaches get touched, so the saving is not limited to `lib.rmeta`.
///
/// Mapping is `unsafe` because the mapping is a window onto a file another
/// process may still be writing: a truncation under the map turns a read into
/// `SIGBUS` rather than the short buffer `fs::read` would have produced. The
/// rlib is a build artifact the caller has just finished compiling, so this is
/// the same bet every linker makes about its inputs.
///
/// Not every input can be mapped, though -- a zero-length file, a pipe passed
/// as `<(...)`, a filesystem that refuses `mmap` -- and all of those are inputs
/// `fs::read` handles. Mapping failure is therefore not an error, just a reason
/// to read the file the slow way.
fn map_rlib(path: &Path) -> Result<Rlib> {
    let ctx = || format!("failed to read rlib {path:?}");
    let file = fs::File::open(path).with_context(ctx)?;
    match unsafe { memmap2::Mmap::map(&file) } {
        Ok(map) => Ok(Rlib::Mapped(map)),
        Err(_) => Ok(Rlib::Read(fs::read(path).with_context(ctx)?)),
    }
}

/// Decide what a member of the rlib that `object` could not parse means:
/// `Ok(())` for one the link is right to leave out, an error for one it was
/// supposed to read.
///
/// Not every member is an object file. rustc puts the crate's metadata in the
/// same archive, and on the targets where that is a raw blob rather than an
/// object there is nothing in it for a linker to read -- so a member that
/// cannot be parsed has to be skipped rather than refused.
///
/// Skipping *every* such member is how this started, and it hides the opposite
/// case: an object member that is damaged, or that holds something other than
/// machine code, is silently left out of the link. The splices compiled into
/// it, and the helpers and `static`s its siblings reference in it, are then
/// simply absent -- so `resplice` patches whichever splices the members that
/// did parse happened to carry, reports exactly that many as applied, and exits
/// successfully. A one-member rlib produces nothing but a "no splices found"
/// warning next to an unpatched copy of the target.
///
/// Two things mark a member as one that had to parse:
///
/// - `object` recognizes its container format, so it really is an object file
///   and the failure is in its contents -- a truncated or otherwise damaged
///   member, which is what a half-written rlib has.
/// - Failing that, its name: rustc names each codegen unit's member `*.o`,
///   while the metadata member is `lib.rmeta`. This catches a member whose
///   damage reaches the magic at its start, and LLVM bitcode (a container
///   `object` does not read at all), which `-C linker-plugin-lto` puts in a
///   `*.o` member in place of machine code.
fn unparseable_member(name: &str, data: &[u8], err: object::Error) -> Result<()> {
    /// Magic of a bare LLVM bitcode module, and of the wrapper form that
    /// carries one (`0x0b17c0de`, little-endian).
    const BITCODE: [&[u8]; 2] = [b"BC\xc0\xde", &[0xde, 0xc0, 0x17, 0x0b]];

    if !name.ends_with(".o") && object::FileKind::parse(data).is_err() {
        return Ok(());
    }
    if BITCODE.iter().any(|magic| data.starts_with(magic)) {
        bail!(
            "archive member {name:?} holds LLVM bitcode rather than machine code, so \
             there are no compiled splices in it to read. That is what \
             `-C linker-plugin-lto` produces -- it leaves the crate's code for the \
             linker's LTO plugin to compile -- so build the replacement crate without \
             it (a plain `cargo build --release` does)"
        )
    }
    bail!(
        "failed to parse archive member {name:?} of the rlib: {err}. Skipping it would \
         leave every `#[Splice]` compiled into it, and every helper and `static` the \
         other members reference in it, out of the link -- so the splices that do \
         resolve would be patched and reported as a successful run, with the rest of \
         the crate silently missing. Rebuild the rlib"
    )
}

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
    let data = map_rlib(rlib_path)?;
    let archive = ArchiveFile::parse(&*data).context("failed to parse rlib archive")?;

    // Every object member of the archive, kept alive together, and not just
    // the ones that carry a splice section: a splice's helpers and `static`s
    // do not have to live in the same member it does. rustc compiles a crate
    // into one object per codegen unit, split along module boundaries -- 16 of
    // them for `--release` and 256 for a debug build -- so a `#[Splice]` in
    // one module and the `fn` it calls in another end up in different members,
    // referencing each other through an undefined symbol that only the rest of
    // the archive can resolve.
    let mut objs = Vec::new();
    for member in archive.members() {
        let member = member.context("failed to read archive member")?;
        let member_data = member.data(&*data).context("failed to read member data")?;

        match object::File::parse(member_data) {
            Ok(obj) => objs.push(obj),
            // Not every member of an rlib is an object file, so a parse
            // failure is only an error for one that was meant to be: see
            // [`unparseable_member`].
            Err(e) => {
                let name = String::from_utf8_lossy(member.name()).into_owned();
                unparseable_member(&name, member_data, e)?;
            }
        }
    }

    // Virtual address the injected segment will be mapped at. All injected
    // sections across all object members share this single segment -- one run
    // injects once, so `base + blob.len()` is always the next free virtual
    // address.
    let base = binary.injected_base()?;

    let mut blob: Vec<u8> = Vec::new();
    let mut patches: Vec<Patch> = Vec::new();
    link_objects(&objs, binary, base, &mut blob, &mut patches)?;

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

    // Translate each region to a file offset before anything is written: the
    // *whole* region has to be file-backed by one segment, not just `begin`.
    // A region that runs off the end of the segment mapping it would otherwise
    // be written over whatever the file holds past that segment -- see
    // [`Binary::va_range_to_offset`]. Appending the injected segment only grows
    // the image, so these offsets stay valid across the injection below.
    let offsets = patches
        .iter()
        .map(|p| {
            binary
                .va_range_to_offset(p.begin, p.end - p.begin)
                .with_context(|| format!("splice {:#x}..{:#x}", p.begin, p.end))
        })
        .collect::<Result<Vec<usize>>>()?;

    if !blob.is_empty() {
        binary.inject_segment(base, &blob)?;
    }

    let mut applied = Vec::new();
    for (patch, offset) in patches.into_iter().zip(offsets) {
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

/// A section, identified across the whole archive: `(member, section index)`.
///
/// Section and symbol indices are per object member, so nothing below can key
/// on them alone -- the closure, the layout and the relocation passes all span
/// every member of the rlib at once.
type SecId = (usize, usize);

/// A symbol, identified across the whole archive: `(member, symbol index)`.
type SymId = (usize, usize);

/// What we may need to know about a section to inject or patch it.
///
/// Only the metadata is read for every section of every object member. The
/// contents -- `bytes` and `relocs` -- are what the linker is expensive about,
/// so they stay empty until the closure in [`link_objects`] reaches the
/// section. See [`extract_relocs`].
///
/// `name` is borrowed from the archive rather than copied: a crate that
/// reimplements a few thousand functions has rustc put each helper and each
/// constant in a section of its own, in whichever codegen unit got it, so this
/// is read for tens of thousands of sections to find the handful that matter.
struct Sec<'d> {
    name: &'d str,
    align: u64,
    writable: bool,
    /// Whether the section holds machine code (`SHF_EXECINSTR`). A `#[Splice]`
    /// on a `fn` lands in an executable section, one on a `static` does not,
    /// and the two cannot be patched the same way. See [`link_objects`].
    code: bool,
    /// The section's contents; empty until the closure reaches the section.
    bytes: Vec<u8>,
    /// The section's relocations; empty until the closure reaches the section.
    relocs: Vec<Rel>,
    /// `Some((begin, end))` for a pinned `.rspl.*` section.
    splice: Option<(u64, u64)>,
}

/// Owned copy of a relocation.
struct Rel {
    offset: u64,
    r_type: RelocationType,
    sym: SymId,
    addend: i64,
    /// True for REL relocations (ARM, MIPS o32) whose addend is stored in the
    /// instruction field rather than carried explicitly.
    has_implicit: bool,
}

/// The parts of a symbol we need to resolve a relocation.
struct SymInfo<'d> {
    name: &'d str,
    /// Section the symbol is defined in, if any. Not necessarily in the member
    /// the symbol itself belongs to: an undefined reference that another member
    /// of the rlib defines is recorded here as pointing at that definition, so
    /// everything downstream treats it like a local one.
    section: Option<SecId>,
    /// Value (offset within its section, or absolute value).
    value: u64,
    undefined: bool,
    /// What the symbol's `st_shndx` actually named, for the symbols that are
    /// neither defined in a section of this archive nor undefined.
    ///
    /// Only `SHN_ABS` carries a value a reference can be resolved against. The
    /// others reach [`SymInfo`] with a `st_value` that is not an address and no
    /// section to make one out of, and are refused rather than used: see the
    /// `resolve` closure in [`link_objects`].
    place: object::SymbolSection,
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

/// Refuse a `.rspl.*` section that carries more than one `#[Splice]` item.
///
/// `#[Splice]` encodes the range into the section name, so two items written
/// with the *same* `begin`/`end` -- a copy-pasted attribute whose body was
/// changed but whose addresses were not -- ask for the same section. When rustc
/// puts them in one codegen unit it merges the two, appending the second item's
/// bytes to the first, and nothing downstream can tell the result apart from a
/// single splice twice the size: the concatenation is patched over
/// `[begin, end)` (or trampolined to, if it no longer fits), so the target runs
/// the first item with the second sitting dead behind it, reported as one
/// successful splice.
///
/// Two such items in *different* codegen units land in different members and
/// are caught as overlapping patches instead. This is the same mistake when
/// they share a member, where the merge leaves only one section to look at and
/// only the member's symbol table still records that there were two.
///
/// The symbols are walked once per member that carries a splice at all, which
/// is a member the linker has to read anyway; members with no splice section
/// are not touched.
fn check_one_item_per_splice(obj: &object::File, splices: &HashSet<usize>) -> Result<()> {
    let mut owners: HashMap<usize, Vec<&str>> = HashMap::new();
    for sym in obj.symbols() {
        // One global definition per `#[Splice]` item: the macro adds
        // `no_mangle`, and the local symbols sharing the section (its section
        // symbol, a `$x`/`$d` mapping symbol) say nothing about how many items
        // landed there.
        if sym.is_undefined() || !sym.is_global() {
            continue;
        }
        let (Some(si), Ok(name)) = (sym.section_index(), sym.name()) else {
            continue;
        };
        if name.is_empty() || !splices.contains(&si.0) {
            continue;
        }
        owners.entry(si.0).or_default().push(name);
    }

    // The lowest-numbered collision, and its items named in sorted order, so
    // the diagnostic does not depend on `HashMap` iteration order (nor, for the
    // names, on the symbol table's).
    let Some((si, mut names)) = owners
        .into_iter()
        .filter(|(_, items)| items.len() > 1)
        .min_by_key(|&(si, _)| si)
    else {
        return Ok(());
    };
    names.sort_unstable();

    let sec_name = obj
        .section_by_index(object::SectionIndex(si))
        .ok()
        .and_then(|s| s.name().map(str::to_owned).ok())
        .unwrap_or_default();
    // Already parsed by `read_sec` before this runs, so the fallback is only
    // for the name the section header turned out not to have.
    let range = match splice_range(&sec_name) {
        Ok(Some((begin, end))) => format!("{begin:#x}..{end:#x}"),
        _ => sec_name.clone(),
    };
    bail!(
        "{} `#[Splice]` items ({}) all claim the range {range}; they share the \
         section {sec_name:?}, which rustc merged into one, so their bytes would be \
         spliced over that range end to end and only the first would ever run. Give \
         each item the range it is replacing.",
        names.len(),
        names.join(", "),
    )
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

/// [`build_exports`], memoized in `cache`.
///
/// The index is built at most once, and only when something actually needs it:
/// an rlib whose splices leave no reference undefined in their own member --
/// every single-member rlib, and any crate whose codegen units rustc could
/// inline across -- never asks, and walking every member's symbol table is not
/// free.
fn exports_for<'d, 'c>(
    objs: &[object::File<'d>],
    cache: &'c mut Option<HashMap<&'d str, (SecId, u64)>>,
) -> &'c HashMap<&'d str, (SecId, u64)> {
    cache.get_or_insert_with(|| {
        if objs.len() < 2 {
            // One member has nothing to resolve across members.
            HashMap::new()
        } else {
            build_exports(objs)
        }
    })
}

/// Index every symbol the rlib's members *define*, by name.
///
/// A splice that calls a helper or reads a `static` living in another codegen
/// unit references it as an **undefined** symbol in its own object member; the
/// definition exists only in the member that got that module. Resolving such a
/// reference needs an index across the whole archive -- what a real linker
/// takes from the archive's own symbol table.
///
/// Only global definitions can be referenced from another member, so local ones
/// are left out; their names are not unique across members anyway. Between two
/// definitions of a name a strong one wins over a weak one, as in a static
/// link, and between two of equal strength the earlier member keeps it.
fn build_exports<'d>(objs: &[object::File<'d>]) -> HashMap<&'d str, (SecId, u64)> {
    use std::collections::hash_map::Entry;

    let mut defs: HashMap<&'d str, (SecId, u64, bool)> = HashMap::new();
    for (mi, obj) in objs.iter().enumerate() {
        for sym in obj.symbols() {
            if sym.is_undefined() || !sym.is_global() {
                continue;
            }
            let (Some(si), Ok(name)) = (sym.section_index(), sym.name()) else {
                continue;
            };
            if name.is_empty() {
                continue;
            }
            let def = ((mi, si.0), sym.address(), sym.is_weak());
            match defs.entry(name) {
                Entry::Vacant(slot) => {
                    slot.insert(def);
                }
                Entry::Occupied(mut slot) if slot.get().2 && !def.2 => {
                    slot.insert(def);
                }
                Entry::Occupied(_) => {}
            }
        }
    }
    defs.into_iter()
        .map(|(name, (sec, value, _))| (name, (sec, value)))
        .collect()
}

/// Read one section's header into a [`Sec`].
///
/// Called for the sections the closure in [`link_objects`] reaches, and not for
/// the whole archive: a crate that reimplements a few thousand functions has
/// rustc put every helper and every constant in a section of its own, in
/// whichever codegen unit got it, and only a handful of those are reachable
/// from a splice.
fn read_sec<'d>(objs: &[object::File<'d>], id: SecId) -> Result<Sec<'d>> {
    let (mi, si) = id;
    let section = objs[mi]
        .section_by_index(object::SectionIndex(si))
        .map_err(|e| anyhow!("bad section index {si} in archive member {mi}: {e}"))?;
    let name = section.name().unwrap_or("");
    Ok(Sec {
        splice: splice_range(name)?,
        name,
        align: section.align().max(1),
        writable: is_runtime_writable(&section),
        code: matches!(section.kind(), SectionKind::Text),
        bytes: Vec::new(),
        relocs: Vec::new(),
    })
}

/// Read the relocations of one section, recording every symbol they reference
/// in `syms`.
///
/// Called for the sections the closure in [`link_objects`] actually reaches,
/// rather than for the whole object, because the bulk of an rlib's relocations
/// belong to sections no splice can reach. A `cargo build` compiles with debug
/// info, and the resulting `.debug_info`, `.debug_ranges`, `.debug_loc` and
/// `.eh_frame` carry tens of thousands of relocations apiece -- for a crate of
/// 2000 splices, 98k of the object's 110k relocations were debug info. Reading
/// those, and interning the name of every symbol they mention, was the single
/// most expensive thing linking did.
fn extract_relocs<'d>(
    objs: &[object::File<'d>],
    id: SecId,
    name: &str,
    syms: &mut HashMap<SymId, SymInfo<'d>>,
    exports: &mut Option<HashMap<&'d str, (SecId, u64)>>,
) -> Result<Vec<Rel>> {
    let (mi, index) = id;
    let obj = &objs[mi];
    let section = obj
        .section_by_index(object::SectionIndex(index))
        .map_err(|e| anyhow!("bad section index {index} ({name}): {e}"))?;

    let mut relocs = Vec::new();
    for (offset, reloc) in section.relocations() {
        let r_type = match reloc.flags() {
            RelocationFlags::Elf { r_type } => r_type,
            _ => return Err(anyhow!("non-ELF relocation in {name}")),
        };
        let sym = match reloc.target() {
            RelocationTarget::Symbol(i) => i,
            other => return Err(anyhow!("unsupported relocation target {other:?} in {name}")),
        };
        relocs.push(Rel {
            offset,
            r_type,
            sym: (mi, sym.0),
            addend: reloc.addend(),
            has_implicit: reloc.has_implicit_addend(),
        });

        // Record the referenced symbol.
        if let std::collections::hash_map::Entry::Vacant(slot) = syms.entry((mi, sym.0)) {
            let s = obj
                .symbol_by_index(sym)
                .map_err(|e| anyhow!("bad symbol index {} in {name}: {e}", sym.0))?;
            let sym_name = s.name().unwrap_or("");
            let mut section = s.section_index().map(|i| (mi, i.0));
            let mut value = s.address();
            let mut undefined = s.is_undefined();

            // An undefined reference may still be satisfied inside the rlib,
            // by the member that got the module holding the definition. Bind
            // it to that section so the closure pulls it in and injects it,
            // exactly as if it had been defined here; only what the archive
            // cannot supply is left to the target's own symbols.
            if undefined {
                if let Some(&(def_sec, def_value)) = exports_for(objs, exports).get(sym_name) {
                    section = Some(def_sec);
                    value = def_value;
                    undefined = false;
                }
            }

            slot.insert(SymInfo {
                name: sym_name,
                section,
                value,
                undefined,
                place: s.section(),
            });
        }
    }
    Ok(relocs)
}

/// The paired `R_MIPS_LO16` addend for an `R_MIPS_HI16`, read from the original
/// section bytes (the low half of the first matching-symbol LO16 instruction).
fn find_pair_lo(arch: Architecture, sec: &Sec, sym: SymId, endian: Endian) -> Option<i64> {
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

/// Refusal for a reference that can only be expressed as an absolute address,
/// against a target the loader will map somewhere else (see
/// [`Binary::is_position_independent`]).
///
/// `what` names the field holding the address and `remedy` says how to get rid
/// of it, both of which depend on how the reference ended up absolute.
fn absolute_in_pie(what: &str, remedy: &str) -> anyhow::Error {
    anyhow!(
        "the target is position-independent (ET_DYN), so the loader maps it at a \
         base chosen at run time, but {what} -- a link-time offset that nothing \
         relocates once the image is loaded, because resplice adds no dynamic \
         relocations. The patched binary would use it as an address and read from \
         (or jump into) the bottom of its address space, so this is refused \
         rather than reported as a successful splice. PC-relative references are \
         unaffected and need nothing: calls, jumps, trampolines and the \
         `adrp`/`lea` form of a data reference all keep working wherever the \
         image lands. Splice a target linked at fixed addresses (`cc -no-pie`), \
         or {remedy}"
    )
}

/// Link every object member of the rlib into the shared injected blob + patch
/// list.
fn link_objects<'d>(
    objs: &[object::File<'d>],
    target: &Binary,
    base: u64,
    blob: &mut Vec<u8>,
    patches: &mut Vec<Patch>,
) -> Result<()> {
    let arch = target.architecture();
    let endian = target.endian();
    let is_64 = target.is_64();

    // The splice sections, across every member. Everything else goes into
    // `secs` only once the closure below reaches it; see [`read_sec`].
    let mut secs: HashMap<SecId, Sec<'d>> = HashMap::new();
    let mut syms: HashMap<SymId, SymInfo<'d>> = HashMap::new();
    let mut splice_idxs: Vec<SecId> = Vec::new();

    for (mi, obj) in objs.iter().enumerate() {
        verify_arch(obj, target)?;
        let mut mine: HashSet<usize> = HashSet::new();
        for section in obj.sections() {
            if !matches!(section.name(), Ok(n) if n.starts_with(".rspl.")) {
                continue;
            }
            let id = (mi, section.index().0);
            splice_idxs.push(id);
            secs.insert(id, read_sec(objs, id)?);
            mine.insert(id.1);
        }
        // Two items given the same range share a section name, which rustc
        // merges within a codegen unit; only this member's symbol table still
        // shows that there were two. See [`check_one_item_per_splice`].
        if !mine.is_empty() {
            check_one_item_per_splice(obj, &mine)?;
        }
    }

    if splice_idxs.is_empty() {
        return Ok(());
    }

    // Index of what the rlib's members define, for references that are
    // undefined in the member carrying them. See [`exports_for`].
    let mut exports: Option<HashMap<&str, (SecId, u64)>> = None;

    // Transitive closure: starting from the splice sections, pull in every
    // section reachable through a relocation, in any member. Splice sections
    // are pinned to their `begin` address; everything else is injected.
    // Sorted, so the layout below -- and therefore every injected address and
    // every byte of the patched binary -- does not depend on the iteration
    // order of a `HashMap`, which varies from one run to the next.
    splice_idxs.sort_unstable();

    let mut inject_order: Vec<SecId> = Vec::new();
    let mut included: Vec<SecId> = splice_idxs.clone();
    // Membership test for the walk below, which `included` cannot serve: it has
    // to keep its insertion order (it is what the layout and relocation passes
    // iterate), and scanning it for every reference visited makes collecting
    // the closure quadratic in the number of sections reached. A crate that
    // reimplements a few thousand functions reaches tens of thousands of them,
    // each helper and `.rodata` constant in its own section.
    let mut seen: HashSet<SecId> = splice_idxs.iter().copied().collect();
    let mut work: Vec<SecId> = splice_idxs.clone();
    while let Some(si) = work.pop() {
        let relocs = extract_relocs(objs, si, secs[&si].name, &mut syms, &mut exports)?;
        for r in &relocs {
            if let Some(secj) = syms.get(&r.sym).and_then(|s| s.section) {
                if seen.insert(secj) {
                    secs.insert(secj, read_sec(objs, secj)?);
                    included.push(secj);
                    inject_order.push(secj);
                    work.push(secj);
                }
            }
        }
        secs.get_mut(&si).unwrap().relocs = relocs;
    }

    // The closure is settled, so now read the contents of just those sections.
    //
    // A read that fails has to be an error, not an empty section. `object`
    // refuses a section whose `sh_offset`/`sh_size` run past the end of the
    // archive member -- which a truncated or otherwise damaged rlib has -- and
    // substituting zero bytes for it turns that into a silent miscompile: a
    // splice with nothing to relocate is then patched as an empty replacement,
    // so `patch_bytes` NOP-fills the whole of `[begin, end)` and the target
    // simply falls through the function that used to be there, reported as a
    // successful splice. A referenced helper read that way is injected as zero
    // bytes, and the calls to it land on whichever section follows it in the
    // blob.
    for &si in &included {
        let name = secs[&si].name;
        let bytes = objs[si.0]
            .section_by_index(object::SectionIndex(si.1))
            .and_then(|s| s.data().map(<[u8]>::to_vec))
            .map_err(|e| anyhow!("failed to read the contents of section {name:?}: {e}"))?;
        secs.get_mut(&si).unwrap().bytes = bytes;
    }

    // A splice whose code fits its `[begin, end)` region is pinned there and
    // patched directly. One that is too large is relocated into the injected
    // segment, with a jump written at `begin` to reach it (a trampoline).
    //
    // Both of those moves assume the splice is *code*. A `#[Splice]` on a
    // `static` replaces data byte-for-byte, and neither works for it:
    //
    // - Oversized, it would be trampolined, leaving a branch instruction
    //   where the target expects a table. The target then reads the branch's
    //   encoding as data, and the real bytes sit unreferenced in the injected
    //   segment.
    // - Undersized, `patch_bytes` fills the remainder of the region with NOP
    //   *instructions*, so the tail of the replaced table becomes whatever
    //   this architecture's NOP happens to encode as.
    //
    // Both are invisible downstream -- `resplice` reports success -- so a data
    // splice has to match its range exactly.
    let mut oversized: Vec<SecId> = Vec::new();
    let mut sec_va: HashMap<SecId, u64> = HashMap::new();
    for &si in &splice_idxs {
        let sec = &secs[&si];
        let (begin, end) = sec.splice.unwrap();
        let region = (end - begin) as usize;
        if !sec.code && sec.bytes.len() != region {
            bail!(
                "data splice {begin:#x}..{end:#x} is {} bytes but its range is {region}; \
                 a `static` splice replaces its range byte-for-byte, so the two have to \
                 match exactly -- adjust `begin`/`end` or the size of the `static`",
                sec.bytes.len()
            );
        }
        if sec.bytes.len() <= region {
            sec_va.insert(si, begin);
        } else {
            oversized.push(si);
        }
    }

    // Lay the injected referenced sections, then any relocated oversized
    // splices, into the shared blob and assign each a virtual address.
    let mut inj_blob_off: HashMap<SecId, usize> = HashMap::new();
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
    let mut got: HashMap<SymId, (u64, usize)> = HashMap::new();
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
    let resolve = |sym: SymId| -> Result<u64> {
        let s = syms
            .get(&sym)
            .ok_or_else(|| anyhow!("relocation references unknown symbol {sym:?}"))?;
        if let Some(secj) = s.section {
            let base_va = sec_va
                .get(&secj)
                .ok_or_else(|| anyhow!("symbol {:?} in un-laid-out section", s.name))?;
            // `st_value` is an offset into the section the symbol is defined
            // in, and nothing has vetted it: it is read back out of an rlib
            // whose symbol table may be damaged or crafted. One past the end
            // is allowed (an end marker names the first byte after the
            // section), anything further is not an offset into the section at
            // all, and adding it to the section's address yields whatever the
            // injected blob happens to hold there -- the next section's bytes,
            // a synthesized GOT slot, or an address past the end of the
            // injected segment that nothing maps.
            //
            // Only `st_value` is checked, not `st_value + addend`: an addend is
            // part of the relocation's own arithmetic rather than an offset
            // into the symbol's section (every x86-64 PC-relative type carries
            // -4 to account for the instruction tail), so it has no meaning to
            // bound.
            let len = secs[&secj].bytes.len() as u64;
            if s.value > len {
                bail!(
                    "the rlib defines {:?} at offset {:#x} of section {:?}, which holds \
                     only {len} bytes. `st_value` is an offset into that section, so \
                     this is not a definition the reference can be pointed at: \
                     resolving it would aim the reference at whatever follows the \
                     section in the injected segment, or past the end of the segment \
                     altogether",
                    s.name,
                    s.value,
                    secs[&secj].name,
                )
            }
            Ok(base_va + s.value)
        } else if s.undefined {
            target.resolve_target_symbol(s.name)?.ok_or_else(|| {
                anyhow!(
                    "unresolved external symbol {:?}: no member of the rlib defines \
                     it, and the target neither defines nor imports it. A splice can \
                     only reach what its own rlib was compiled from, or a symbol the \
                     target itself provides; a precompiled `core`/`std` (or any other \
                     crate's) rlib is not read, so the `core::panicking` entry points \
                     a bounds check, an overflow check or an `unwrap` compiles into \
                     cannot be resolved either",
                    s.name
                )
            })
        } else if s.place == object::SymbolSection::Absolute {
            // `SHN_ABS`: a link-time constant, which is the one thing left
            // whose `st_value` is usable as it stands.
            Ok(s.value)
        } else if s.place == object::SymbolSection::Common {
            // `SHN_COMMON` is a *request* for space rather than a definition:
            // `st_value` holds the alignment the symbol wants and `st_size`
            // how much of it, and a linker is expected to allocate the object
            // itself. `resplice` allocates nothing, and `object` reports a
            // common symbol's address as 0 so the alignment is not mistaken
            // for one -- which is exactly the value that got written, leaving
            // the splice reading or writing through a null pointer.
            //
            // rustc emits no common symbols, but an rlib is not only rustc's
            // output: a replacement crate that compiles a C helper has those
            // objects bundled into the same archive, and a file-scope `int x;`
            // compiled with `-fcommon` is a common symbol.
            bail!(
                "the rlib defines {:?} as a common symbol (SHN_COMMON), which asks a \
                 linker to allocate space for it rather than giving it an address. \
                 resplice allocates none, so the reference would resolve to address 0 \
                 and the patched binary would dereference a null pointer. Give the \
                 symbol a definition with storage -- in C that means an initializer, \
                 or `-fno-common` (the default since GCC 10)",
                s.name
            )
        } else {
            // Everything else: an `SHN_XINDEX` whose `.symtab_shndx` table is
            // missing or out of range, one of the reserved indices between
            // `SHN_LORESERVE` and `SHN_HIRESERVE`, and the `SHN_ABS` of an
            // `STT_FILE` entry (a source file name, not a definition). There is
            // no section to resolve against and `st_value` is still a
            // section-relative offset, so writing it as it stands pointed the
            // reference at an arbitrary low address.
            let shndx = match s.place {
                object::SymbolSection::None => "names no section at all",
                _ => {
                    "is reserved or unresolvable (at or above SHN_LORESERVE, or an \
                     SHN_XINDEX with no .symtab_shndx table to read it from)"
                }
            };
            bail!(
                "the rlib defines {:?} with an `st_shndx` that {shndx}, so resplice \
                 cannot resolve it: its value {:#x} is neither an offset into a \
                 section it could inject nor an address a reference can be pointed at",
                s.name,
                s.value
            )
        }
    };

    // Fill each synthesized GOT slot with its resolved symbol address (done in
    // its own pass so the field-writing loop below can borrow `blob` freely).
    for (&sym, &(_, goff)) in &got {
        let s = resolve(sym)?;
        endian.write_uint(&mut blob[..], goff, got_width, s);
    }

    // A position-independent target is mapped at a base chosen at run time, so
    // the only references that keep working are the ones whose field holds a
    // displacement rather than an address. See
    // [`Binary::is_position_independent`] and the refusal in the loop below.
    let pie = target.is_position_independent()?;

    // Apply relocations for every included section. A splice that fits is
    // relocated into a private buffer and patched into `[begin, end)` later;
    // everything else (referenced sections, oversized splices) is relocated in
    // place inside the injected blob.
    for &si in &included {
        let sec = &secs[&si];
        let site_base = sec_va[&si];
        // Everything laid out into the blob above -- referenced sections and
        // oversized splices alike -- is relocated there in place; a splice that
        // was never laid out is the one that fits its region, and is relocated
        // in a private buffer. (Scanning `oversized` for that answer instead
        // cost a linear search per included section.)
        let bo = inj_blob_off.get(&si).copied();
        let is_fit_splice = sec.splice.is_some() && bo.is_none();

        let mut out = if is_fit_splice { sec.bytes.clone() } else { Vec::new() };

        for r in &sec.relocs {
            // Both the resolution and the encoding below can fail, and neither
            // message is actionable without knowing which splice (or which of
            // its helpers) carries the reference.
            let context = || {
                let sym = &syms[&r.sym];
                // A section symbol has no name of its own, and rustc relocates
                // against one for most of what a splice reaches -- every
                // `static` and every helper gets a section to itself, and the
                // reference is to that section rather than to the symbol. An
                // empty name said nothing about what the reference was to, so
                // fall back to naming the section it resolves into.
                let what = if sym.name.is_empty() {
                    sym.section
                        .and_then(|secj| secs.get(&secj).map(|s| s.name))
                        .unwrap_or("<unnamed symbol>")
                } else {
                    sym.name
                };
                format!("relocation against {what:?} in {:?}", sec.name)
            };
            let s = resolve(r.sym).with_context(context)?;
            let p = site_base + r.offset;
            let a = if r.has_implicit { 0 } else { r.addend };
            let got_va = got.get(&r.sym).map(|&(gva, _)| gva);

            // An absolute address baked into a position-independent target is
            // wrong by its load base, and there is nothing left to correct it:
            // the dynamic relocations that would have done so were emitted when
            // the target was linked, and `resplice` adds none. The patched
            // binary dereferences a link-time offset instead -- a `static`
            // holding a reference reads from the bottom of the address space --
            // and `resplice` reports a successful splice.
            //
            // A symbol that is not an address at all is exempt: an `SHN_ABS`
            // symbol (no section, not undefined) is a link-time constant and
            // means the same wherever the image lands.
            if pie && (syms[&r.sym].section.is_some() || syms[&r.sym].undefined) {
                if arch::writes_absolute_address(arch, r.r_type) {
                    return Err(absolute_in_pie(
                        &format!(
                            "relocation type {} writes the absolute address {s:#x}",
                            r.r_type.0
                        ),
                        "keep addresses out of the data the splice reaches. What \
                         stores one is a `static` holding a reference, a `&str`, a \
                         function pointer, or a table of any of those -- index a \
                         byte array rather than a slice of references",
                    ))
                    .with_context(context);
                }
                // A GOT reference *could* be resolved without a slot, by
                // relaxing the indirect load into the PC-relative form the
                // linker's own relaxations produce (`mov rax, [rip+got]` ->
                // `lea rax, [rip+sym]`, `adrp`+`ldr` -> `adrp`+`add`). That
                // rewrites the splice's instructions rather than just filling
                // in their fields, which nothing here does yet.
                if got_va.is_some() {
                    return Err(absolute_in_pie(
                        &format!(
                            "this reference goes through a GOT slot resplice \
                             synthesizes in the injected segment, which holds the \
                             absolute address {s:#x}"
                        ),
                        "avoid the indirection: rustc routes a reference through \
                         the GOT when its definition is in another codegen unit, \
                         so moving the definition into the module the splice is \
                         written in makes the reference PC-relative",
                    ))
                    .with_context(context);
                }
            }
            let pair_lo = if arch::is_mips_hi16(arch, r.r_type) && r.has_implicit {
                find_pair_lo(arch, sec, r.sym, endian)
            } else {
                None
            };

            // Hand the encoder exactly this section's bytes. Slicing only from
            // `bo` would let a relocation whose offset lies past the end of its
            // own section quietly rewrite whichever section follows it in the
            // injected blob, and `arch::apply` would have no way to tell.
            let buf: &mut [u8] = match bo {
                Some(bo) => &mut blob[bo..bo + sec.bytes.len()],
                None => &mut out,
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
            .with_context(context)?;
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

    /// Wrap several object members in a minimal GNU `ar` archive, in the order
    /// given -- which is the order `link_rlib` reads them in.
    fn ar_wrap_many(members: &[(&str, &[u8])]) -> Vec<u8> {
        let mut out = Vec::from(&b"!<arch>\n"[..]);
        for (name, data) in members {
            let mut header = format!("{:<16}", format!("{name}/"));
            header.push_str(&format!(
                "{:<12}{:<6}{:<6}{:<8}{:<10}`\n",
                0,
                0,
                0,
                "644",
                data.len()
            ));
            assert_eq!(header.len(), 60);
            out.extend_from_slice(header.as_bytes());
            out.extend_from_slice(data);
            if data.len() % 2 == 1 {
                out.push(b'\n');
            }
        }
        out
    }

    /// Wrap object bytes in a minimal GNU `ar` archive.
    fn ar_wrap(name: &str, data: &[u8]) -> Vec<u8> {
        ar_wrap_many(&[(name, data)])
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
        run_link_on(&crafted_target(), "rlib", rlib)
    }

    /// [`run_link`] against a target of the caller's choosing.
    fn run_link_on(target: &[u8], tag: &str, rlib: &[u8]) -> (Binary, Vec<Applied>) {
        let rlib_path = temp_path(tag);
        fs::write(&rlib_path, rlib).unwrap();
        let mut bin = load(target);
        let applied = link_rlib(&mut bin, &rlib_path).unwrap();
        fs::remove_file(&rlib_path).ok();
        (bin, applied)
    }

    /// The crafted target marked position-independent, which is what every
    /// mainstream toolchain emits by default. `e_type` is the first field after
    /// the 16-byte identifier in both ELF classes.
    fn crafted_pie_target() -> Vec<u8> {
        let mut d = crafted_target();
        d[0x10..0x12].copy_from_slice(&goblin::elf::header::ET_DYN.to_le_bytes());
        d
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

    /// A splice's helpers and `static`s do not have to live in the same object
    /// member the splice does. rustc compiles a crate into one object per
    /// codegen unit, split along module boundaries, so a `#[Splice]` in one
    /// module and the `fn` it calls in another land in different members of
    /// the rlib and reference each other through an undefined symbol.
    ///
    /// Resolution used to stop at the member being linked -- and members
    /// carrying no splice section of their own were not even read -- so such a
    /// reference was reported as an `unresolved external symbol` naming the
    /// crate's *own* helper. Any replacement crate whose splices and helpers
    /// were not all in one module failed to link, which `cargo build` (256
    /// codegen units) made almost unavoidable.
    #[test]
    fn test_link_rlib_resolves_refs_across_archive_members() {
        use object::write::{Object, Relocation, Symbol, SymbolSection as WSection};
        use object::{
            Architecture, BinaryFormat, Endianness, RelocationFlags, SectionKind, SymbolFlags,
            SymbolKind, SymbolScope,
        };

        // The member holding the splice. Both of its references are undefined
        // here: nothing in this object defines them.
        let mut obj = Object::new(BinaryFormat::Elf, Architecture::X86_64, Endianness::Little);
        let name = format!(".rspl.{SPLICE_VA:x}.{:x}", SPLICE_VA + 0x20).into_bytes();
        let rspl = obj.add_section(vec![], name, SectionKind::Text);
        obj.append_section_data(rspl, &SPLICE_CODE, 16);
        for (offset, sym_name, kind, r_type) in [
            (1u64, "helper", SymbolKind::Text, object::elf::R_X86_64_PLT32),
            (8, "TABLE", SymbolKind::Data, object::elf::R_X86_64_PC32),
        ] {
            let symbol = obj.add_symbol(Symbol {
                name: sym_name.as_bytes().to_vec(),
                value: 0,
                size: 0,
                kind,
                scope: SymbolScope::Dynamic,
                weak: false,
                section: WSection::Undefined,
                flags: SymbolFlags::None,
            });
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
        let splice_member = obj.write().unwrap();

        // The member that got the other module: it defines both of them, and
        // carries no splice of its own.
        let mut obj = Object::new(BinaryFormat::Elf, Architecture::X86_64, Endianness::Little);
        let text = obj.add_section(vec![], b".text.helper".to_vec(), SectionKind::Text);
        obj.append_section_data(text, &HELPER_CODE, 1);
        let table = table_data();
        let rodata = obj.add_section(vec![], b".rodata.table".to_vec(), SectionKind::ReadOnlyData);
        obj.append_section_data(rodata, &table, 1);
        for (sym_name, kind, section, size) in [
            ("helper", SymbolKind::Text, text, HELPER_CODE.len()),
            ("TABLE", SymbolKind::Data, rodata, table.len()),
        ] {
            obj.add_symbol(Symbol {
                name: sym_name.as_bytes().to_vec(),
                value: 0,
                size: size as u64,
                kind,
                // What rustc gives a cross-codegen-unit definition: global
                // binding, hidden visibility.
                scope: SymbolScope::Linkage,
                weak: false,
                section: WSection::Section(section),
                flags: SymbolFlags::None,
            });
        }
        let helper_member = obj.write().unwrap();

        // The splice comes first, so the definitions sit in a member the
        // linker has not reached yet when it reads the references.
        let rlib = ar_wrap_many(&[("splice.o", &splice_member), ("helper.o", &helper_member)]);
        let (bin, applied) = run_link(&rlib);

        assert_eq!(applied.len(), 1);
        assert!(!applied[0].trampoline);
        // Both references point into the injected segment, at the helper code
        // and table bytes pulled in from the other member.
        assert_refs_resolved(&bin, 0x1000, SPLICE_VA);
    }

    /// A reference that neither the rlib nor the target can satisfy is an
    /// error, and the message has to name the splice it came from: the symbol
    /// alone is a mangled `core` path for anything that can panic (a bounds
    /// check, an overflow check, an `unwrap`), which says nothing about which
    /// of a crate's splices needs rewriting.
    #[test]
    fn test_link_rlib_reports_unresolved_reference_with_its_splice() {
        use object::write::{Object, Relocation, Symbol, SymbolSection as WSection};
        use object::{
            Architecture, BinaryFormat, Endianness, RelocationFlags, SectionKind, SymbolFlags,
            SymbolKind, SymbolScope,
        };

        let mut obj = Object::new(BinaryFormat::Elf, Architecture::X86_64, Endianness::Little);
        let name = format!(".rspl.{SPLICE_VA:x}.{:x}", SPLICE_VA + 0x20).into_bytes();
        let rspl = obj.add_section(vec![], name, SectionKind::Text);
        obj.append_section_data(rspl, &SPLICE_CODE, 16);
        let symbol = obj.add_symbol(Symbol {
            name: b"nowhere_to_be_found".to_vec(),
            value: 0,
            size: 0,
            kind: SymbolKind::Text,
            scope: SymbolScope::Dynamic,
            weak: false,
            section: WSection::Undefined,
            flags: SymbolFlags::None,
        });
        obj.add_relocation(
            rspl,
            Relocation {
                offset: 1,
                symbol,
                addend: -4,
                flags: RelocationFlags::Elf {
                    r_type: object::elf::R_X86_64_PLT32,
                },
            },
        )
        .unwrap();

        let msg = link_err("rlib_unresolved", &ar_wrap("splice.o", &obj.write().unwrap()));
        assert!(msg.contains("unresolved external symbol"), "message was {msg:?}");
        assert!(msg.contains("nowhere_to_be_found"), "message was {msg:?}");
        assert!(msg.contains(".rspl.401000.401020"), "message was {msg:?}");
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

    /// An 8-byte splice: `mov rax, [rip + d32]` (operand@3); `ret`.
    const GOT_SPLICE_CODE: [u8; 8] = [0x48, 0x8b, 0x05, 0, 0, 0, 0, 0xc3];

    /// An rlib whose splice reaches its `TABLE` indirectly, through a
    /// `GOTPCREL`-style reference. There is no GOT in the injected segment to
    /// point it at, so the linker has to synthesize a slot holding `TABLE`'s
    /// address and aim the load at that.
    fn build_rlib_got() -> Vec<u8> {
        use object::write::{Object, Relocation, Symbol, SymbolSection as WSection};
        use object::{
            Architecture, BinaryFormat, Endianness, RelocationFlags, SectionKind, SymbolFlags,
            SymbolKind, SymbolScope,
        };

        let mut obj = Object::new(BinaryFormat::Elf, Architecture::X86_64, Endianness::Little);

        let table = table_data();
        let rodata = obj.add_section(vec![], b".rodata.table".to_vec(), SectionKind::ReadOnlyData);
        obj.append_section_data(rodata, &table, 8);
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

        let name = format!(".rspl.{SPLICE_VA:x}.{:x}", SPLICE_VA + 0x20).into_bytes();
        let rspl = obj.add_section(vec![], name, SectionKind::Text);
        obj.append_section_data(rspl, &GOT_SPLICE_CODE, 16);
        obj.add_relocation(
            rspl,
            Relocation {
                offset: 3,
                symbol: table_sym,
                addend: -4,
                flags: RelocationFlags::Elf {
                    r_type: object::elf::R_X86_64_REX_GOTPCRELX,
                },
            },
        )
        .unwrap();

        ar_wrap("splice.o", &obj.write().unwrap())
    }

    /// A GOT-style reference is satisfied by a slot the linker synthesizes in
    /// the injected segment: the load must reach that slot, and the slot must
    /// hold the address of the injected data.
    #[test]
    fn test_link_rlib_synthesizes_a_got_slot() {
        let (bin, applied) = run_link_on(&crafted_target(), "rlib_got", &build_rlib_got());
        assert_eq!(applied.len(), 1);

        // Follow `mov rax, [rip + d32]` to the slot, and the slot to the table.
        let d = bin.data();
        let disp = i32::from_le_bytes(d[0x1003..0x1007].try_into().unwrap());
        let slot_va = (SPLICE_VA as i64 + 7 + disp as i64) as u64;
        assert!(slot_va >= INJECT_BASE, "GOT slot {slot_va:#x} not injected");
        let soff = bin.va_to_offset(slot_va).unwrap();
        let table_va = u64::from_le_bytes(d[soff..soff + 8].try_into().unwrap());
        assert!(table_va >= INJECT_BASE, "slot holds {table_va:#x}, not injected data");
        let toff = bin.va_to_offset(table_va).unwrap();
        assert_eq!(&d[toff..toff + table_data().len()], &table_data());
    }

    /// Every value `resplice` resolves is a link-time one, and a
    /// position-independent target -- a PIE, which is what `cc` produces by
    /// default, or a shared library -- is mapped at a base the loader picks. A
    /// displacement survives that; an absolute address does not, and nothing
    /// relocates it afterwards, because `resplice` emits no dynamic
    /// relocations.
    ///
    /// Both ways of writing one have to be refused. The target used to be
    /// patched regardless, and reported as a successful splice: a `static`
    /// holding a reference (an ordinary `[&[u8; 4]; 2]`) was injected with the
    /// pointer it contains left as a link-time offset, so the patched binary
    /// dereferenced the bottom of its address space and died on the first read.
    #[test]
    fn test_link_rlib_rejects_absolute_references_into_a_pie_target() {
        for (tag, rlib, cue) in [
            // `R_X86_64_64`: the pointer inside an injected RELRO `static`.
            (
                "rlib_pie_abs",
                build_rlib_relro(".data.rel.ro.NAMES", 0),
                ".data.rel.ro.NAMES",
            ),
            // A synthesized GOT slot, which holds an address just the same.
            ("rlib_pie_got", build_rlib_got(), "GOT slot"),
        ] {
            let rlib_path = temp_path(tag);
            fs::write(&rlib_path, &rlib).unwrap();
            let mut bin = load(&crafted_pie_target());
            let result = link_rlib(&mut bin, &rlib_path);
            fs::remove_file(&rlib_path).ok();

            let msg = format!("{:#}", result.expect_err("expected an error"));
            assert!(msg.contains("position-independent"), "message was {msg:?}");
            assert!(msg.contains(cue), "message was {msg:?}");
            // Refused before anything was written over the target's own bytes.
            assert_eq!(&bin.data()[0x1000..0x1020], &[0xcc; 0x20], "target was patched");

            // The same rlib still links against the same target as `ET_EXEC`:
            // the absolute reference is the only thing being refused.
            let (_, applied) = run_link_on(&crafted_target(), tag, &rlib);
            assert_eq!(applied.len(), 1);
        }
    }

    /// The counterpart: a position-independent target is not refused wholesale.
    /// A call and a PC-relative data reference -- what the bulk of a replacement
    /// compiles into -- are shifted along with everything they point at, so they
    /// are as correct in a PIE as in a fixed-address executable and must still
    /// be linked.
    #[test]
    fn test_link_rlib_accepts_pc_relative_references_into_a_pie_target() {
        let (bin, applied) =
            run_link_on(&crafted_pie_target(), "rlib_pie_pcrel", &build_rlib(SPLICE_VA + 0x20));
        assert_eq!(applied.len(), 1);
        assert!(!applied[0].trampoline);
        assert_refs_resolved(&bin, 0x1000, SPLICE_VA);
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

    /// A splice whose region runs off the end of the segment that maps its
    /// `begin` must be refused.
    ///
    /// Only `begin` used to be translated, and the region length was then
    /// written from there — so the part of the region past the segment's
    /// file-backed data landed on file bytes belonging to something else, while
    /// the addresses it was meant for were never touched. On a real target that
    /// silently rewrote the symbol table (a `static` splice whose range overshot
    /// `.data` into `.bss`) or, for a longer overshoot, the next segment's
    /// `.dynamic` and `.got`, leaving a binary that segfaults in the loader —
    /// and `resplice` reported a successful splice either way.
    #[test]
    fn test_link_rlib_rejects_splice_past_its_segment() {
        // Shrink the target's PT_LOAD so its file-backed data stops at
        // 0x401010, leaving the 0x20-byte splice region half outside it. The
        // image is still 0x2000 bytes, so the write itself would fit in the
        // file: nothing downstream would notice.
        let mut target = crafted_target();
        for field in [32, 40] {
            // p_filesz, p_memsz of the first program header (at offset 64).
            target[64 + field..64 + field + 8].copy_from_slice(&0x1010u64.to_le_bytes());
        }

        let rlib_path = temp_path("rlib_past_segment");
        fs::write(&rlib_path, build_rlib(SPLICE_VA + 0x20)).unwrap();
        let mut bin = load(&target);
        let result = link_rlib(&mut bin, &rlib_path);
        fs::remove_file(&rlib_path).ok();

        let msg = format!("{:#}", result.expect_err("expected an error"));
        assert!(msg.contains("splice 0x401000..0x401020"), "message was {msg:?}");
        assert!(msg.contains("file-backed data only for"), "message was {msg:?}");
        assert!(msg.contains("0x400000..0x401010"), "message was {msg:?}");
        // Refused before anything was written over the target's own bytes.
        assert_eq!(&bin.data()[0x1000..0x1020], &[0xcc; 0x20], "target was patched");

        // The same splice is still applied when the segment does cover it.
        let (bin, applied) = run_link(&build_rlib(SPLICE_VA + 0x20));
        assert_eq!(applied.len(), 1);
        assert_ne!(&bin.data()[0x1000..0x1020], &[0xcc; 0x20]);
    }

    /// A reference that cannot reach the injected segment must be refused
    /// rather than written with its high bits dropped.
    ///
    /// Nothing malformed is needed to get there. The injected segment holds the
    /// splice's helpers and constants, so every reference into it is relative
    /// to the splice site, and on x86-64 those fields are 32 bits wide -- an
    /// `--inject-base` more than 2GB away (the README's whole reason for the
    /// flag is placing the segment somewhere known-free, which on a 64-bit
    /// target is easily that far) does not fit in them. The truncated
    /// displacement pointed the call at an unrelated address -- address 0, for
    /// the base used here -- and `resplice` reported a successful splice.
    /// aarch64 refused the same link from the start, since its branch encoder
    /// range-checks; this was the rest of the encoders catching up.
    #[test]
    fn test_link_rlib_rejects_references_beyond_their_field() {
        /// 4GB: free in the crafted target's address space, and far enough from
        /// the splice site that no 32-bit field can span the distance.
        const FAR: u64 = 0x1_0000_0000;

        // Link `rlib` with the injected segment pinned at `base`.
        let link_at = |tag: &str, rlib: &[u8], base: u64| {
            let rlib_path = temp_path(tag);
            fs::write(&rlib_path, rlib).unwrap();
            let mut bin = load(&crafted_target());
            bin.set_injected_base(Some(base));
            let result = link_rlib(&mut bin, &rlib_path);
            fs::remove_file(&rlib_path).ok();
            (bin, result)
        };

        // A splice that fits its region stays at `begin` and calls its injected
        // helper through a PLT32 displacement.
        let (bin, result) = link_at("rlib_far", &build_rlib(SPLICE_VA + 0x20), FAR);
        let msg = format!("{:#}", result.expect_err("expected an error"));
        assert!(msg.contains("does not fit"), "message was {msg:?}");
        assert!(msg.contains("signed 32-bit"), "message was {msg:?}");
        assert!(msg.contains(".rspl.401000.401020"), "message was {msg:?}");
        assert_eq!(&bin.data()[0x1000..0x1020], &[0xcc; 0x20], "target was patched");

        // An oversized one is relocated into the segment instead, and the jump
        // written at `begin` to reach it has the same 32-bit displacement.
        let (bin, result) = link_at("rlib_far_tramp", &build_rlib(SPLICE_VA + 8), FAR);
        let msg = format!("{:#}", result.expect_err("expected an error"));
        assert!(msg.contains("jmp rel32"), "message was {msg:?}");
        assert!(msg.contains("trampoline at 0x401000"), "message was {msg:?}");
        assert_eq!(&bin.data()[0x1000..0x1008], &[0xcc; 8], "target was patched");

        // A base that is within reach still links, both ways: the check must
        // not refuse the distances a real splice actually spans.
        for end in [SPLICE_VA + 0x20, SPLICE_VA + 8] {
            let (_, result) = link_at("rlib_near", &build_rlib(end), 0x1000_0000);
            assert_eq!(result.unwrap().len(), 1);
        }
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

    /// A `#[Splice]` on a `static` lands in a non-executable section and
    /// replaces its range byte-for-byte. Neither of the sizing moves available
    /// to code works for it: an oversized one would be trampolined, writing a
    /// branch instruction where the target expects a table, and an undersized
    /// one would have the tail of its range filled with NOP instructions.
    /// Both report success, so both have to be refused up front.
    #[test]
    fn test_link_rlib_rejects_missized_data_splice() {
        use object::write::Object;
        use object::{Architecture, BinaryFormat, Endianness, SectionKind};

        // An 8-byte range holding `bytes` as read-only data, not code.
        let build = |bytes: &[u8]| {
            let mut obj =
                Object::new(BinaryFormat::Elf, Architecture::X86_64, Endianness::Little);
            let name = format!(".rspl.{SPLICE_VA:x}.{:x}", SPLICE_VA + 8).into_bytes();
            let sec = obj.add_section(vec![], name, SectionKind::ReadOnlyData);
            obj.append_section_data(sec, bytes, 8);
            ar_wrap("splice.o", &obj.write().unwrap())
        };

        let msg = link_err("rlib_data_oversize", &build(&[0xaa; 16]));
        assert!(msg.contains("data splice"), "message was {msg:?}");
        assert!(msg.contains("16 bytes"), "message was {msg:?}");
        assert!(msg.contains("0x401000..0x401008"), "message was {msg:?}");

        let msg = link_err("rlib_data_undersize", &build(&[0xaa; 4]));
        assert!(msg.contains("data splice"), "message was {msg:?}");
        assert!(msg.contains("4 bytes"), "message was {msg:?}");

        // An exact fit is still patched straight over the range, and the byte
        // just past it is left alone.
        let data = [0xde, 0xad, 0xbe, 0xef, 0x11, 0x22, 0x33, 0x44];
        let (bin, applied) = run_link(&build(&data));
        assert_eq!(applied.len(), 1);
        assert!(!applied[0].trampoline);
        assert_eq!(&bin.data()[0x1000..0x1008], &data);
        assert_eq!(bin.data()[0x1008], 0xcc, "past the range, untouched");
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

    /// Two `#[Splice]` items given the same `begin`/`end` ask for the same
    /// `.rspl.<begin>.<end>` section, and rustc merges them when they land in
    /// the same codegen unit -- one section holding both functions, with a
    /// global symbol apiece. That is exactly what rustc produces for
    ///
    /// ```ignore
    /// #[Splice(begin = 0x401000, end = 0x401020)]
    /// pub extern "C" fn first() -> i32 { 42 }
    /// #[Splice(begin = 0x401000, end = 0x401020)]
    /// pub extern "C" fn second() -> i32 { 43 }
    /// ```
    ///
    /// which compiles without complaint. The concatenation was then spliced
    /// over the range as if it were one replacement: the target ran `first` and
    /// carried `second` as dead bytes behind it, and `resplice` reported one
    /// successful splice of twice the expected size. The same pair in different
    /// codegen units is caught as overlapping patches, so only the merged case
    /// slipped through.
    #[test]
    fn test_link_rlib_rejects_two_splices_sharing_a_section() {
        use object::write::{Object, Symbol, SymbolSection as WSection};
        use object::{
            Architecture, BinaryFormat, Endianness, SectionKind, SymbolFlags, SymbolKind,
            SymbolScope,
        };

        let mut obj = Object::new(BinaryFormat::Elf, Architecture::X86_64, Endianness::Little);
        let name = format!(".rspl.{SPLICE_VA:x}.{:x}", SPLICE_VA + 0x20).into_bytes();
        let sec = obj.add_section(vec![], name, SectionKind::Text);
        // What the merge leaves behind: both bodies end to end, each with its
        // own `no_mangle` global symbol.
        let bodies = [
            ("second", [0xb8, 43, 0, 0, 0, 0xc3]), // mov eax,43; ret
            ("first", [0xb8, 42, 0, 0, 0, 0xc3]),  // mov eax,42; ret
        ];
        for (fn_name, body) in bodies {
            let value = obj.append_section_data(sec, &body, 1);
            obj.add_symbol(Symbol {
                name: fn_name.as_bytes().to_vec(),
                value,
                size: body.len() as u64,
                kind: SymbolKind::Text,
                scope: SymbolScope::Dynamic,
                weak: false,
                section: WSection::Section(sec),
                flags: SymbolFlags::None,
            });
        }

        let msg = link_err("rlib_shared_section", &ar_wrap("splice.o", &obj.write().unwrap()));
        assert!(msg.contains("2 `#[Splice]` items"), "message was {msg:?}");
        // Both are named, in a fixed order rather than the symbol table's.
        assert!(msg.contains("first, second"), "message was {msg:?}");
        assert!(msg.contains("0x401000..0x401020"), "message was {msg:?}");
        assert!(msg.contains(".rspl.401000.401020"), "message was {msg:?}");

        // One item per section is still fine, symbol and all.
        let (_, applied) = run_link(&build_rlib(SPLICE_VA + 0x20));
        assert_eq!(applied.len(), 1);
    }

    /// A section whose contents `object` refuses to read must be an error.
    ///
    /// The read failure used to be swallowed, leaving the section with zero
    /// bytes: a splice with nothing to relocate was then patched as an empty
    /// replacement, so `patch_bytes` NOP-filled the whole of `[begin, end)` and
    /// the target fell straight through the function that used to be there --
    /// reported as a successful splice.
    #[test]
    fn test_link_rlib_rejects_unreadable_section_contents() {
        use object::write::Object;
        use object::{Architecture, BinaryFormat, Endianness, SectionKind};

        // A self-contained splice: no relocations, so nothing else would have
        // noticed that its bytes never arrived.
        let mut obj = Object::new(BinaryFormat::Elf, Architecture::X86_64, Endianness::Little);
        let name = format!(".rspl.{SPLICE_VA:x}.{:x}", SPLICE_VA + 0x20).into_bytes();
        let sec = obj.add_section(vec![], name, SectionKind::Text);
        obj.append_section_data(sec, &[0xb8, 42, 0, 0, 0, 0xc3], 1); // mov eax,42; ret
        let mut member = obj.write().unwrap();

        // Damage the section header so its contents run past the end of the
        // member, as a truncated rlib's would. The member still parses: only
        // reading that one section's bytes fails.
        let shoff = u64::from_le_bytes(member[0x28..0x30].try_into().unwrap()) as usize;
        let shentsize = u16::from_le_bytes(member[0x3a..0x3c].try_into().unwrap()) as usize;
        let index = {
            let parsed = object::File::parse(&*member).unwrap();
            parsed
                .sections()
                .find(|s| matches!(s.name(), Ok(n) if n.starts_with(".rspl.")))
                .unwrap()
                .index()
                .0
        };
        // Elf64_Shdr::sh_size sits at offset 32 of the entry.
        let sh_size = shoff + index * shentsize + 32;
        member[sh_size..sh_size + 8].copy_from_slice(&u64::MAX.to_le_bytes());

        let msg = link_err("rlib_unreadable", &ar_wrap("splice.o", &member));
        assert!(msg.contains("failed to read the contents"), "message was {msg:?}");
        assert!(msg.contains(".rspl.401000.401020"), "message was {msg:?}");
    }

    /// An archive member `object` cannot parse must be an error when it is one
    /// resplice was supposed to read, and skipped only when it is not an object
    /// file at all.
    ///
    /// Every unparseable member used to be skipped, which hid a damaged one:
    /// the splices compiled into it, and the helpers its siblings reference in
    /// it, were simply absent from the link. resplice patched whatever the
    /// members that did parse carried, reported exactly that many as applied,
    /// and exited successfully -- so a damaged rlib left the target partly
    /// unpatched with nothing anywhere saying so.
    #[test]
    fn test_link_rlib_rejects_unparseable_member() {
        // A member that really is an ELF object, with a splice in it, whose
        // section-header table points past its end: a truncated member's
        // damage. Its container format is still recognizable, so the failure is
        // in its contents rather than in what it claims to be.
        let mut damaged = build_obj_sym_value(0);
        damaged[0x28..0x30].copy_from_slice(&u64::MAX.to_le_bytes()); // e_shoff
        assert!(object::File::parse(&*damaged).is_err(), "member still parses");

        let msg = link_err("rlib_bad_member", &ar_wrap("splice.o", &damaged));
        assert!(msg.contains("failed to parse archive member"), "message was {msg:?}");
        assert!(msg.contains("\"splice.o\""), "message was {msg:?}");

        // The same member alongside one that links: refused all the same, since
        // a run that patches only the splices that survived is the thing being
        // refused.
        let good = build_obj_sym_value(0);
        let rlib = ar_wrap_many(&[("good.o", &good), ("splice.o", &damaged)]);
        let msg = link_err("rlib_bad_member_2", &rlib);
        assert!(msg.contains("\"splice.o\""), "message was {msg:?}");

        // LLVM bitcode in a `*.o` member -- what `-C linker-plugin-lto`
        // produces -- is a container `object` does not read at all, so only the
        // member's name says it was meant to hold compiled splices.
        let mut bitcode = Vec::from(&b"BC\xc0\xde"[..]);
        bitcode.extend_from_slice(&[0u8; 64]);
        let msg = link_err("rlib_bitcode", &ar_wrap("splice.o", &bitcode));
        assert!(msg.contains("LLVM bitcode"), "message was {msg:?}");
        assert!(msg.contains("linker-plugin-lto"), "message was {msg:?}");

        // A member that is neither an object file nor named like one is still
        // skipped: rustc's metadata member is exactly that, and refusing it
        // would refuse every rlib.
        let mut rmeta = Vec::from(&b"rust\0\0\0\0"[..]);
        rmeta.extend_from_slice(&[0xab; 64]);
        let rlib = ar_wrap_many(&[("lib.rmeta", &rmeta), ("splice.o", &good)]);
        let (bin, applied) = run_link_on(&crafted_target(), "rlib_rmeta", &rlib);
        assert_eq!(applied.len(), 1);
        assert_eq!(lea_target(&bin), INJECT_BASE);
    }

    /// An rlib whose splice reads a `TABLE` through
    /// `lea rax, [rip + TABLE]`, where `TABLE` is declared at `sym_value`
    /// bytes into the 16-byte `.rodata.table` section it is defined in.
    ///
    /// rustc writes 0 there. The parameter is what a damaged or crafted symbol
    /// table can say instead: `st_value` is an offset into the section, and
    /// nothing in the archive has vetted it.
    fn build_obj_sym_value(sym_value: u64) -> Vec<u8> {
        use object::write::{Object, Relocation, Symbol, SymbolSection as WSection};
        use object::{
            Architecture, BinaryFormat, Endianness, RelocationFlags, SectionKind, SymbolFlags,
            SymbolKind, SymbolScope,
        };

        let mut obj = Object::new(BinaryFormat::Elf, Architecture::X86_64, Endianness::Little);
        let table = table_data();
        let rodata = obj.add_section(vec![], b".rodata.table".to_vec(), SectionKind::ReadOnlyData);
        obj.append_section_data(rodata, &table, 1);
        let table_sym = obj.add_symbol(Symbol {
            name: b"TABLE".to_vec(),
            value: sym_value,
            size: 0,
            kind: SymbolKind::Data,
            scope: SymbolScope::Compilation,
            weak: false,
            section: WSection::Section(rodata),
            flags: SymbolFlags::None,
        });

        let name = format!(".rspl.{SPLICE_VA:x}.{:x}", SPLICE_VA + 0x20).into_bytes();
        let rspl = obj.add_section(vec![], name, SectionKind::Text);
        obj.append_section_data(rspl, &PTR_SPLICE_CODE, 16);
        obj.add_relocation(
            rspl,
            Relocation {
                offset: 3,
                symbol: table_sym,
                addend: -4,
                flags: RelocationFlags::Elf { r_type: object::elf::R_X86_64_PC32 },
            },
        )
        .unwrap();

        obj.write().unwrap()
    }

    /// [`build_obj_sym_value`]'s object, as a one-member rlib.
    fn build_rlib_sym_value(sym_value: u64) -> Vec<u8> {
        ar_wrap("splice.o", &build_obj_sym_value(sym_value))
    }

    /// Follow the `lea rax, [rip + d32]` of [`PTR_SPLICE_CODE`] patched at file
    /// offset 0x1000 and return the address it resolves to.
    fn lea_target(bin: &Binary) -> u64 {
        let d = bin.data();
        assert_eq!((d[0x1000], d[0x1002]), (0x48, 0x05), "lea opcode");
        let disp = i32::from_le_bytes(d[0x1003..0x1007].try_into().unwrap());
        (SPLICE_VA as i64 + 7 + disp as i64) as u64
    }

    /// A symbol's `st_value` is an offset into the section it is defined in, so
    /// one that runs past the end of that section is not a definition a
    /// reference can be pointed at and has to be refused.
    ///
    /// It used to be added to the section's injected address unchecked. The
    /// reference then pointed at whatever followed the section in the injected
    /// blob -- the next section's bytes, a synthesized GOT slot -- or, for a
    /// large enough value, past the end of the injected segment at an address
    /// nothing maps at all. `resplice` reported a successful splice in every
    /// case: the 16-byte table below with `st_value = 0x10000` resolved to
    /// 0x413000, 0x10000 bytes past a segment that is 16 bytes long.
    #[test]
    fn test_link_rlib_rejects_symbol_value_outside_its_section() {
        let rlib_path = temp_path("rlib_sym_oob");
        fs::write(&rlib_path, build_rlib_sym_value(0x1_0000)).unwrap();
        let mut bin = load(&crafted_target());
        let result = link_rlib(&mut bin, &rlib_path);
        fs::remove_file(&rlib_path).ok();

        let msg = format!("{:#}", result.expect_err("expected an error"));
        assert!(msg.contains("\"TABLE\" at offset 0x10000"), "message was {msg:?}");
        assert!(msg.contains(".rodata.table"), "message was {msg:?}");
        assert!(msg.contains("only 16 bytes"), "message was {msg:?}");
        // Refused before anything was written over the target's own bytes.
        assert_eq!(&bin.data()[0x1000..0x1020], &[0xcc; 0x20], "target was patched");

        // One byte past the end is the same mistake, and just as refused: the
        // interesting values are the ones a wrong offset lands on, not only the
        // wild ones.
        let msg = link_err("rlib_sym_oob_1", &build_rlib_sym_value(17));
        assert!(msg.contains("at offset 0x11"), "message was {msg:?}");

        // The boundary itself is legal -- an end marker names the first byte
        // past the section -- and so is every offset inside it. Each has to
        // still resolve, to exactly that many bytes into the injected section.
        for value in [0, 1, 8, 16] {
            let (bin, applied) =
                run_link_on(&crafted_target(), "rlib_sym_ok", &build_rlib_sym_value(value));
            assert_eq!(applied.len(), 1);
            assert_eq!(lea_target(&bin), INJECT_BASE + value, "st_value {value}");
        }
    }

    /// A common symbol (`SHN_COMMON`) is a request for space a linker is meant
    /// to allocate, not a definition with an address: `st_value` holds the
    /// alignment it wants. `resplice` allocates no such storage, so it has to
    /// refuse the reference.
    ///
    /// It used to fall through to the "absolute symbol" arm, which takes
    /// `st_value` as the final address. `object` reports a common symbol's
    /// address as 0 precisely so the alignment is not mistaken for one, so 0 is
    /// what got written: the patched binary dereferenced a null pointer, and
    /// `resplice` reported a successful splice.
    #[test]
    fn test_link_rlib_rejects_common_symbol() {
        use object::write::{Object, Relocation, Symbol, SymbolSection as WSection};
        use object::{
            Architecture, BinaryFormat, Endianness, RelocationFlags, SectionKind, SymbolFlags,
            SymbolKind, SymbolScope,
        };

        let mut obj = Object::new(BinaryFormat::Elf, Architecture::X86_64, Endianness::Little);
        let counter = obj.add_symbol(Symbol {
            name: b"shared_counter".to_vec(),
            value: 8, // a common symbol's value is its alignment
            size: 8,
            kind: SymbolKind::Data,
            scope: SymbolScope::Dynamic,
            weak: false,
            section: WSection::Common,
            flags: SymbolFlags::None,
        });
        let name = format!(".rspl.{SPLICE_VA:x}.{:x}", SPLICE_VA + 0x20).into_bytes();
        let rspl = obj.add_section(vec![], name, SectionKind::Text);
        obj.append_section_data(rspl, &PTR_SPLICE_CODE, 16);
        obj.add_relocation(
            rspl,
            Relocation {
                offset: 3,
                symbol: counter,
                addend: -4,
                flags: RelocationFlags::Elf { r_type: object::elf::R_X86_64_PC32 },
            },
        )
        .unwrap();

        let msg = link_err("rlib_common", &ar_wrap("splice.o", &obj.write().unwrap()));
        assert!(msg.contains("SHN_COMMON"), "message was {msg:?}");
        assert!(msg.contains("shared_counter"), "message was {msg:?}");
        assert!(msg.contains("address 0"), "message was {msg:?}");
    }

    /// The file offset of the `Elf64_Sym` entry for the symbol named `sym`.
    fn sym_entry_off(member: &[u8], sym: &str) -> usize {
        let obj = object::File::parse(member).unwrap();
        let symtab = obj
            .sections()
            .find(|s| matches!(s.name(), Ok(".symtab")))
            .expect(".symtab");
        let (off, _) = symtab.file_range().expect("symtab file range");
        let index = obj
            .symbols()
            .find(|s| matches!(s.name(), Ok(n) if n == sym))
            .expect("symbol")
            .index()
            .0;
        off as usize + index * 24 // Elf64_Sym is 24 bytes
    }

    /// Rewrite the `st_shndx` of the symbol named `sym` in an ELF64 object,
    /// leaving everything else alone. `object::write` cannot emit the reserved
    /// indices, and a real toolchain emits `SHN_XINDEX` only for an object with
    /// more than 65279 sections.
    fn set_st_shndx(member: &mut [u8], sym: &str, shndx: u16) {
        // st_shndx sits at offset 6 of the entry, after st_name/st_info/st_other.
        let at = sym_entry_off(member, sym) + 6;
        member[at..at + 2].copy_from_slice(&shndx.to_le_bytes());
    }

    /// `SHN_ABS` is the one section index with no section behind it whose
    /// `st_value` really is the answer: a link-time constant, which means the
    /// same wherever the image lands. Refusing the other such indices must not
    /// take it with them.
    #[test]
    fn test_link_rlib_resolves_an_absolute_symbol() {
        let mut member = build_obj_sym_value(0);
        set_st_shndx(&mut member, "TABLE", object::elf::SHN_ABS.0);
        // `st_value` is the whole address now, not an offset into a section, so
        // give it one inside the target rather than the 0 the builder wrote.
        let at = sym_entry_off(&member, "TABLE") + 8; // st_value
        member[at..at + 8].copy_from_slice(&(TARGET_BASE + 0x1800).to_le_bytes());

        let (bin, applied) =
            run_link_on(&crafted_target(), "rlib_abs", &ar_wrap("splice.o", &member));
        assert_eq!(applied.len(), 1);
        assert_eq!(lea_target(&bin), TARGET_BASE + 0x1800);
    }

    /// A section index `resplice` cannot resolve to a section -- a reserved
    /// one, or an `SHN_XINDEX` whose `.symtab_shndx` table is missing -- leaves
    /// a symbol with no section to inject and an `st_value` that is still a
    /// section-relative offset. That used to reach the "absolute symbol" arm
    /// too, which wrote the offset as if it were an address.
    #[test]
    fn test_link_rlib_rejects_unresolvable_section_index() {
        for shndx in [object::elf::SHN_LORESERVE, object::elf::SHN_XINDEX.0] {
            let mut member = build_obj_sym_value(0x40);
            set_st_shndx(&mut member, "TABLE", shndx);

            let msg = link_err("rlib_bad_shndx", &ar_wrap("splice.o", &member));
            assert!(msg.contains("cannot resolve it"), "message was {msg:?}");
            assert!(msg.contains("reserved or unresolvable"), "message was {msg:?}");
            assert!(msg.contains("TABLE"), "message was {msg:?}");
            assert!(msg.contains("0x40"), "message was {msg:?}");
        }
    }

    /// Linking the same rlib twice has to produce the same bytes.
    ///
    /// The sections are collected into a `HashMap` and the injected segment is
    /// laid out in the order they come back out of it, which Rust randomizes per
    /// process. With more than one splice that put a different address on every
    /// injected helper and constant from one run to the next, so the patched
    /// binary -- the artifact a user ships, checksums or compares against a
    /// previous build -- was a different file each time it was produced.
    #[test]
    fn test_link_rlib_is_deterministic() {
        use object::write::{Object, Relocation};
        use object::{
            Architecture, BinaryFormat, Endianness, RelocationFlags, SectionKind,
        };

        // Four splices, each calling a helper of its own, so a layout that
        // follows `HashMap` order has four orderings to disagree about.
        let rlib = {
            let mut obj =
                Object::new(BinaryFormat::Elf, Architecture::X86_64, Endianness::Little);
            for i in 0..4u64 {
                let begin = SPLICE_VA + i * 0x20;
                let helper = obj.add_section(
                    vec![],
                    format!(".text.helper{i}").into_bytes(),
                    SectionKind::Text,
                );
                // Distinct lengths, so a different order really does move the
                // later helpers to different addresses.
                obj.append_section_data(helper, &vec![0xc3; 1 + i as usize], 1);
                let helper_sym = obj.section_symbol(helper);

                let name = format!(".rspl.{begin:x}.{:x}", begin + 0x20).into_bytes();
                let rspl = obj.add_section(vec![], name, SectionKind::Text);
                obj.append_section_data(rspl, &[0xe8, 0, 0, 0, 0, 0xc3], 16);
                obj.add_relocation(
                    rspl,
                    Relocation {
                        offset: 1,
                        symbol: helper_sym,
                        addend: -4,
                        flags: RelocationFlags::Elf {
                            r_type: object::elf::R_X86_64_PLT32,
                        },
                    },
                )
                .unwrap();
            }
            ar_wrap("splice.o", &obj.write().unwrap())
        };

        let (first, applied) = run_link(&rlib);
        assert_eq!(applied.len(), 4);
        // Repeatedly, because the orders that disagree are drawn at random: a
        // single second run matching could be luck.
        for _ in 0..8 {
            let (again, _) = run_link(&rlib);
            assert_eq!(first.data(), again.data(), "patched binary is not reproducible");
        }
    }
}
