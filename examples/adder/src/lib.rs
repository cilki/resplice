//! Example reimplementation crate.
//!
//! Each `#[Splice]` function is compiled to machine code and patched over the
//! byte range `[begin, end)` — **virtual addresses** — of a target binary by the
//! `resplice` tool. Anything the function references is resolved automatically:
//! referenced Rust code and `static` data are injected into the target as a new
//! segment, and calls to functions the target already provides bind to the
//! target's own symbols. If the replacement is larger than `[begin, end)`, it is
//! placed in the injected segment and reached via a jump written at `begin`.

use resplice_macros::Splice;

/// A helper the spliced function calls. Because a splice references it, its
/// compiled code is pulled in and injected into the target automatically.
#[inline(never)]
fn scale(x: i32) -> i32 {
    x.wrapping_mul(3)
}

/// Read-only data referenced by the splice; injected alongside `scale`.
static TABLE: [i32; 4] = [10, 20, 30, 40];

/// Reimplement the code at `[0x401160, 0x4011c6)` of some target binary.
///
/// The addresses are illustrative (the end-to-end test discovers real ones from
/// a throwaway target rather than touching this file). `pub extern "C"` is added
/// automatically when omitted, so a plain `fn` works too.
#[Splice(begin = 0x401160, end = 0x4011c6)]
pub extern "C" fn compute(i: usize) -> i32 {
    scale(TABLE[i & 3])
}
