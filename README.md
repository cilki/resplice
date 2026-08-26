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

#[Splice(begin = 0x1670, end = 0x1680)]
fn add_one_plus_one() -> i32 {
    1 + 1
}
```

What we get now is the original binary augmented with our custom function! If
we're adequately motivated, we can repeat this step iteratively until our entire
program is reverse engineered in Rust.

But, most likely, we only care about reversing a few specific sections.

## How it works

`Splice` compiles each annotated function into its own object-file section named
`.rspl.<begin>.<end>` (the `begin`/`end` **virtual addresses** encoded in hex).
The `resplice` tool reads those sections back out of the compiled rlib and
patches each function's machine code over `[begin, end)` in the target.

Real replacements are rarely self-contained — they call helpers, read `static`
data, or call libc. `resplice` resolves those relocations recursively:

- Referenced Rust code and read-only data are collected (transitively) and
  **injected into the target as a new segment** (an unused `PT_NOTE` program
  header is converted into a `PT_LOAD`), then the references are fixed up to
  point at their injected addresses.
- Calls to symbols the target already **defines or imports** (e.g. a libc
  function reachable through the PLT) bind to the target's own addresses.

When a replacement is **larger than `[begin, end)`**, it does not have to fit:
the full function is relocated into the injected segment and a jump (trampoline)
is written at `begin` to reach it, so `[begin, end)` only needs room for the
jump.

## Usage

Write your replacements in a library crate that depends on `resplice-macros`:

```rust
use resplice_macros::Splice;

#[Splice(begin = 0x1000, end = 0x1020)]
fn my_replacement_function() -> i32 {
    // Your Rust implementation here
    42
}
```

Build it to an rlib and apply it to the target binary:

```sh
cargo build --release              # produces target/release/libyourcrate.rlib
resplice ./original-binary target/release/libyourcrate.rlib ./patched-binary
```

See `examples/adder` for a complete crate.

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

