use resplice_macros::Splice;

/// Reimplement the code at `[0x1670, 0x1680)` in a target binary as Rust.
///
/// The addresses here are illustrative (they match the README example). The
/// end-to-end test builds its own throwaway crate with addresses discovered
/// from a real target rather than touching this file.
#[Splice(begin = 0x1670, end = 0x1680)]
fn add_one_plus_one() -> i32 {
    1 + 1
}
