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
  function reachable through the PLT) bind to the target's own addresses.

When a replacement is **larger than `[begin, end)`**, it does not have to fit:
the full function is relocated into the injected segment and a jump (trampoline)
is written at `begin` to reach it, so `[begin, end)` only needs room for the
jump.

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
a function, a data splice has to fit its range: the oversize path writes a jump
at `begin` to the relocated copy, which is meaningless for data. Any other kind
of item is a compile error — `#[Splice]` accepts only `fn` and `static`.

### Placing the injected segment

By default the injected segment is mapped one page past the end of the target's
image — one *target* page, which is 64K on a typical aarch64 image rather than
4K. That is only safe when nothing else claims that address, and often
something does: a binary whose allocator hands out memory from the end of
`.bss` will overwrite the injected code and data on its first large allocation.
Pass a known-free address instead:

```sh
resplice --inject-base 0x1c0000 ./original-binary ./lib.rlib ./patched-binary
```

The address must be aligned to the target's page size — the largest `p_align`
among its `PT_LOAD` segments, which is 64K for a typical aarch64 image rather
than 4K. `resplice` rejects a misaligned base instead of emitting a binary the
loader would refuse.

### What the injected segment can hold

The segment is mapped read+execute, so only data that is immutable at run time
can go in it. That is not the same as "not `SHF_WRITE`": an immutable `static`
holding references (`static NAMES: [&[u8; 4]; 2] = [&A, &B]`) is emitted into
`.data.rel.ro`, and so are the `core::panic::Location` records behind a bounds
check. Those are writable only for as long as a dynamic linker needs to
relocate them — `resplice` resolves their pointers statically — so they are
injected like any other `.rodata`.

Data that really is mutable at run time (`static mut`, `.bss`) is still
unsupported and reported as an error:

```
error: splice references writable section ".data._ZN4demo5COUNT17h…E.0";
injecting writable data (mutable statics/.bss) is not yet supported
```

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

`resplice` refuses an rlib whose architecture or byte order differs from the
target's rather than splicing in machine code the target cannot run:

```
error: rlib is aarch64 (LE, 64-bit) but target is x86-64 (LE, 64-bit)
```

## Supported targets

The target must be an **ELF** image; anything else is rejected while loading.
Within ELF, how far support goes varies by architecture:

| Architecture | Status |
| --- | --- |
| x86-64 | Exercised end-to-end by the test suite |
| AArch64 | Exercised end-to-end by the test suite |
| ARM (32-bit) | Relocation encoders written from the ABI, unit-tested only |
| MIPS / MIPS64 (LE and BE) | As ARM, plus the gaps below |
| x86 (32-bit) | Detected, but mis-linked — see below |

`tests/resplice.test` builds its throwaway target out of host assembly, so one
run covers whichever of x86-64 or AArch64 it runs on; covering both means
running it on both. AArch64 is worth running specifically because its 64K
`p_align` is what exercises the non-4K segment layout.

Known gaps:

- **MIPS `$gp`-relative GOT relocations** (`R_MIPS_GOT16` and friends) index
  the target's own GOT, which the injected segment cannot extend, so they are
  rejected rather than mis-resolved. Splices that only call target-*defined*
  symbols are the reliable path on MIPS.
- **32-bit x86** object files are decoded with the x86-64 relocation table,
  where the type numbers mean different things (`R_386_32` and `R_X86_64_64`
  are both type 1). An i386 rlib with absolute relocations is therefore
  mis-linked; treat i386 as unsupported until that is fixed.
