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

As you may have guessed by the name, `Splice` forcibly inserts the machine code
for Rust functions into a target binary (either via direct substitution or via
trampolines).

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
