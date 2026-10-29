# Declarative binary patching with Rust

**resplice** takes "rewrite it in Rust" to a whole new level. It's a macro that
makes re-implementing sections of machine code in Rust (a little) more fun.

## Illustrative example

Take a trivial example that adds 1 + 1 in assembly:

```
0000000000001670 <main>:
    1670:       d10043ff        sub     sp, sp, #0x10
    1674:       b9000fff        str     wzr, [sp, #12]
    1678:       52800040        mov     w0, #0x2
    167c:       910043ff        add     sp, sp, #0x10
    1680:       d65f03c0        ret
```

Now let's reimplement it in Rust (probably with the help of a decompiler, in
practice):

```rust
use resplice_macros::Splice;

#[Splice(begin = 0x1670, end = 0x1684)]
fn add_one_plus_one() -> i32 {
    1 + 1
}
```

`begin`/`end` are **virtual addresses** and the range is half-open — `end` is
the address just past the last byte being replaced. Here the final `ret` sits at
`0x1680` and is four bytes long, so the whole function is `[0x1670, 0x1684)`.

What we get now is the original binary augmented with our custom function! If
we're adequately motivated, we can repeat this step iteratively until our entire
program is reverse engineered in Rust.

But, most likely, we only care about reversing a few specific sections.

## How it works

`Splice` compiles each annotated item into its own object-file section named
`.rspl.<begin>.<end>` (the `begin`/`end` **virtual addresses** in lowercase hex,
without the `0x`). The `resplice` tool reads those sections back out of the
compiled rlib and patches each one's bytes over `[begin, end)` in the target.

Real replacements are rarely self-contained — they call helpers, read `static`
data, or call libc. `resplice` resolves those relocations recursively:

- Referenced Rust code and read-only data are collected (transitively) and
  **injected into the target as a new segment**, then the references are fixed
  up to point at their injected addresses. The segment needs a program header:
  an unused `PT_NOTE` is converted into a `PT_LOAD` when the target has one,
  otherwise the program-header table is grown by one entry into the padding
  that follows it.
- Calls to symbols the target already **defines or imports** (e.g. a libc
  function reachable through the PLT) bind to the target's own addresses. Two
  kinds of definition are reported rather than bound, because their `st_value`
  is not an address: an ifunc (`STT_GNU_IFUNC`), which carries its resolver, and
  a thread-local (`STT_TLS`), which carries an offset into the TLS block. A
  statically linked glibc defines `memcpy`, `memmove`, `memset` and `strlen`
  the first way and `errno` the second.
- The rlib's own symbols are read the same way, and the same rule applies to
  them: a definition whose `st_value` is not an offset into a section the
  closure can inject is reported rather than resolved. That covers a value past
  the end of the section it claims to be in, a common symbol (`SHN_COMMON`,
  which asks a linker to allocate the storage — `resplice` allocates none, so
  the reference would resolve to address 0), and a reserved or unresolvable
  `st_shndx`.

That closure spans the whole rlib, not just the object file the splice itself
landed in. rustc compiles a crate into one object per codegen unit, partitioned
along module boundaries, so a `#[Splice]` in one module and the helper it calls
in another are in different members of the archive and reference each other by
name — splitting replacements across modules is fine.

What is *not* reachable is anything the rlib was not compiled from. Generic and
`#[inline]` code from `core` is instantiated into the rlib and works, but items
that only exist in a precompiled `core`/`std` (or any other crate's) rlib are
never read, so a splice cannot reach them. In practice that means the
`core::panicking` entry points, which is what a bounds check, an overflow check
or an `unwrap` compiles into:

```
error: relocation against "_RNvNtCs..._4core9panicking18panic_bounds_check" in
".rspl.4007e4.40084c": unresolved external symbol
"_RNvNtCs..._4core9panicking18panic_bounds_check": no member of the rlib defines
it, and the target neither defines nor imports it. [...]
```

So index with `get_unchecked`, do arithmetic with the `wrapping_*` methods, and
avoid `unwrap`. Build with `--release` (a debug build has `debug_assertions` and
overflow checks on, both of which panic), and add `panic = "abort"` to the
profile to drop the unwind guard rustc can otherwise put on the `extern "C"`
replacement — itself a call into `core`:

```toml
[profile.release]
panic = "abort"
```

Leave `lto` out of that profile, though. With link-time optimization on, rustc
stops compiling the crate and writes LLVM bitcode into the rlib instead, for the
linker's plugin to compile later — and `resplice` *is* the linker here, so there
is no machine code left for it to splice. It says so rather than reporting an
unpatched binary as a successful run:

```
error: archive member "mycrate-...-cgu.0.rcgu.o" holds LLVM bitcode rather than
machine code, so there are no compiled splices in it to read. [...] Build the
replacement crate with LTO off: `lto = false` in the Cargo profile you built it
with (`lto = true`, `"fat"` and `"thin"` all produce this), and no
`-C linker-plugin-lto` in `RUSTFLAGS`
```

When a replacement is **larger than `[begin, end)`**, it does not have to fit:
the full function is relocated into the injected segment and a jump (trampoline)
is written at `begin` to reach it, so `[begin, end)` only needs room for the
jump.

One run applies every `#[Splice]` in the rlib, so a crate can replace as many
ranges of the target as it likes. Two splices claiming *overlapping* ranges
cannot both survive — each patch NOP-fills the whole of its own region, so
whichever was written last would erase part of the other — and are refused
before anything is written:

```
error: splices 0x1000..0x1020 and 0x1010..0x1030 overlap; each one NOP-fills its
whole region, so they would overwrite each other
```

## Usage

`resplice` is a single binary; `cargo build --release` in this repository leaves
it at `target/release/resplice`.

Write your replacements in a library crate that depends on `resplice-macros`:

```rust
use resplice_macros::Splice;

#[Splice(begin = 0x1000, end = 0x1020)]
fn my_replacement_function() -> i32 {
    // Your Rust implementation here
    42
}
```

`#[Splice]` adds `pub` and `extern "C"` when they are absent, so a plain `fn`
works and the replacement still matches the calling convention the target's
caller expects. An explicitly written visibility or ABI is left alone.

Build it to an rlib and apply it to the target binary:

```sh
cargo build --release              # produces target/release/libyourcrate.rlib
resplice ./original-binary target/release/libyourcrate.rlib ./patched-binary
```

See `examples/adder` for a complete crate.

### Replacing data, not just code

`#[Splice]` also applies to a `static`, whose initializer bytes replace
`[begin, end)` byte-for-byte — handy for rewriting a table baked into the
original image without touching any code:

```rust
#[Splice(begin = 0x2ec70, end = 0x2ec78)]
static PRICES: [u32; 2] = [100, 200];
```

Give the static a `repr(C)` type so its layout is exactly what you wrote. Unlike
a function, a data splice has to match its range *exactly* — neither way out
that `resplice` has for code means anything for data, so a mismatch is an error
rather than a patch:

```
error: data splice 0x2ec70..0x2ec78 is 32 bytes but its range is 8; a `static`
splice replaces its range byte-for-byte, so the two have to match exactly --
adjust `begin`/`end` or the size of the `static`
```

An oversized one would be relocated into the injected segment and reached by a
jump written at `begin`, leaving a branch instruction where the target expects a
table; an undersized one would have the rest of its range filled with NOP
*instructions*. Any other kind of item is a compile error — `#[Splice]` accepts
only `fn` and `static`.

The range also has to be one the image actually holds bytes for. A
zero-initialized table lives in `.bss`, which the loader creates out of nothing
rather than reading — so its addresses are mapped (the RW segment's `p_memsz`
covers them, which is what `readelf -l` shows) but not file-backed, and there is
nothing there to replace:

```
error: 0x420040 is in the zero-filled tail (`.bss`) of the PT_LOAD at
0x41fdc8..0x420140: that segment is backed by the file only up to 0x420038, and
the loader zeroes the rest out of nothing, so there is no byte in the image at
0x420040 to patch. [...]
```

Give such a table its values from a code splice instead.

### Callers that rely on more than the calling convention

`extern "C"` gives a replacement the platform's calling convention, and that is
all a caller is normally entitled to assume. GCC's callers can assume more. With
`-fipa-ra`, which is on at `-O2`, a caller that can see the body of the function
it calls keeps values in the registers that body happens not to touch, even the
ones the convention says a call may overwrite. That applies whenever the callee
is defined in the same source file as its caller (or anywhere, under LTO). The
one exception is an exported function of a shared library, which can be
interposed. A replacement that uses one of those registers, as compiled
Rust is free to, overwrites a value its caller still needs. The splice applies
without an error, and the patched program crashes or misbehaves later in the
*caller*:

```
00400680 <main>:
  400680:  mov  x8, x1          // argv, kept in x8 across the call ...
  4006b0:  bl   400840 <sum>    // ... because the original `sum` never writes x8
  4006c0:  ldr  x0, [x8, #8]    // segfaults once the replacement has used x8
```

`resplice` cannot detect this, because it never sees the callers. If a splice
breaks its caller this way, check the caller's disassembly for registers that it
sets before the call and reads after it. Then put a naked wrapper at `begin` that
saves those registers around a call to the real implementation. The wrapper
leaves the argument registers alone, so the arguments reach `sum_impl` as they
arrived. Here, on aarch64:

```rust
use resplice_macros::Splice;

#[inline(never)]
extern "C" fn sum_impl(a: *const i32, n: i32) -> i32 { /* ... */ }

#[Splice(begin = 0x400840, end = 0x40089c)]
#[unsafe(naked)]
pub extern "C" fn sum() {
    core::arch::naked_asm!(
        "stp x29, x30, [sp, #-96]!",
        "mov x29, sp",
        "stp x8, x9, [sp, #16]",   // x8-x17: the registers a call may
        "stp x10, x11, [sp, #32]", // overwrite, other than the argument
        "stp x12, x13, [sp, #48]", // registers x0-x7
        "stp x14, x15, [sp, #64]",
        "stp x16, x17, [sp, #80]",
        "bl {imp}",
        "ldp x8, x9, [sp, #16]",
        "ldp x10, x11, [sp, #32]",
        "ldp x12, x13, [sp, #48]",
        "ldp x14, x15, [sp, #64]",
        "ldp x16, x17, [sp, #80]",
        "ldp x29, x30, [sp], #96",
        "ret",
        imp = sym sum_impl,
    )
}
```

The wrapper does not save the argument registers. Under `-fipa-ra` the caller
may still expect them back unchanged if the original never wrote them, so save
them too when the caller's disassembly reads them after the call.

### Placing the injected segment

By default the injected segment is mapped one page past the end of the target's
image. That is only safe when nothing else claims that address, and often
something does: a binary whose allocator hands out memory from the end of
`.bss` will overwrite the injected code and data on its first large allocation.
Pass a known-free address instead:

```sh
resplice --inject-base 0x1c0000 ./original-binary ./lib.rlib ./patched-binary
```

The address must be aligned to the target's page size — the largest `p_align`
among its `PT_LOAD` segments, which is 64K for a typical aarch64 image rather
than 4K. `resplice` rejects a misaligned base instead of emitting a binary the
loader would refuse, and likewise rejects a base that is aligned but not free —
one whose segment would be mapped over the target's own image, which the loader
either refuses outright or loads into a program with its code overwritten:

```
error: injected segment 0x400000..0x400010 would be mapped over the target's own
PT_LOAD at 0x400000..0x4009d4 (both round out to pages of 0x10000, so
0x400000..0x410000 overlaps 0x400000..0x410000); choose an --inject-base that is
free in the target's address space
```

The base also has to be within *reach* of the splices. A splice addresses its
injected helpers and constants relative to its own address, through relocation
fields narrower than a pointer — 32 bits on x86-64, so ±2GB, and only ±128MB
for an aarch64 `bl` — and the trampoline of an oversized splice is the same
width. A base further away than the field can express is rejected rather than
written with its high bits dropped:

```
error: trampoline at 0x401000 cannot reach its target 0x100000020; choose an
--inject-base within reach of the splice: PC-relative displacement 0xffbff01b
does not fit the signed 32-bit field this relocation writes; truncating it would
leave the reference pointing at an unrelated address. The symbol is too far from
the relocated site -- for a reference into the injected segment, choose an
--inject-base closer to the splice
```

Note that the segment is mapped read+execute, so referenced *writable* data
(mutable `static`s, `.bss`) is not supported yet and is reported as an error.

### Position-independent targets

A target the loader maps at a base of its own choosing — a PIE executable,
which is what `cc` and every other mainstream toolchain produce by default, or a
shared library — can be spliced, with one restriction.

The addresses in such an image, and so the `begin`/`end` you read out of a
disassembly of it, are link-time *offsets*; at run time everything sits at those
offsets plus a base picked per process. Everything `resplice` resolves is a
link-time value, and whether that survives depends on what the reference holds:

- A **PC-relative** reference survives unchanged. The splice, the injected
  segment and the target's own code are all shifted by the same amount, so the
  distance between any two of them is what it was. That covers calls into
  injected helpers, calls to functions the target imports through its PLT, the
  `adrp`/`lea` form of a data reference, and the trampoline of an oversized
  splice — the bulk of what a replacement compiles into.
- An **absolute** address does not. `resplice` emits no dynamic relocations, so
  nothing corrects it once the image is loaded. Such a reference is refused
  rather than written:

```
error: relocation against ".rodata._ZN4play1A17h..E" in
".data.rel.ro._ZN4play5NAMES17h..E": the target is position-independent
(ET_DYN), so the loader maps it at a base chosen at run time, but relocation
type 257 writes the absolute address 0x40010 -- a link-time offset that nothing
relocates once the image is loaded [...]
```

What needs an absolute address is a `static` that *stores* one — a reference, a
`&str`, a function pointer, or a table of any of those:

```rust
static A: [u8; 4] = *b"abcd";
static NAMES: [&[u8; 4]; 2] = [&A, &A];  // two absolute pointers
```

So either keep those out of the data a splice can reach (index a byte array
rather than a slice of references), or splice a target linked at fixed
addresses (`cc -no-pie`).

### Cross-compiling for another architecture

The replacement crate must be compiled for the **same** architecture as the
target binary. [`cross`](https://github.com/cross-rs/cross) makes this a
one-liner — it runs the Rust toolchain for the chosen target inside a container,
so no local cross toolchain is required:

```sh
cargo install cross
cross build --release --target aarch64-unknown-linux-gnu
# -> target/aarch64-unknown-linux-gnu/release/libyourcrate.rlib
resplice ./aarch64-binary \
    target/aarch64-unknown-linux-gnu/release/libyourcrate.rlib \
    ./patched
```

