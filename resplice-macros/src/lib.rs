use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Expr, Item, Lit, Meta};

/// Marks a function or static as a replacement for a section of the binary.
///
/// The item's machine code (or, for a `static`, its bytes) is emitted into a
/// dedicated section named `.rspl.<begin>.<end>` (addresses in lowercase hex).
/// The `resplice` tool reads this section back out of the compiled rlib, using
/// the section name to recover the target address range and the section's
/// bytes as the replacement code.
///
/// # Arguments
///
/// Both are **virtual addresses** in the target image, and the range is
/// half-open: `[begin, end)`.
///
/// * `begin` - Address of the first byte to replace
/// * `end` - Address just past the last byte to replace
///
/// # Example
///
/// The function is exported (`pub`) and given the C ABI (`extern "C"`) so the
/// replacement matches the calling convention the target's caller expects. Both
/// are added automatically when absent, so the annotated function can be written
/// as a plain `fn`; an explicitly written visibility or ABI is left untouched.
///
/// Replacing a function whose last instruction is a four-byte `ret` at `0x1680`
/// means an `end` of `0x1684`, not `0x1680`:
///
/// ```ignore
/// #[Splice(begin = 0x1670, end = 0x1684)]
/// fn add_one_plus_one() -> i32 {
///     1 + 1
/// }
/// ```
///
/// A `static` works the same way, replacing a data range byte-for-byte with
/// the static's initializer (its type should be `repr(C)` so the layout is
/// exact). It is exported like a function is, so the bytes survive as a
/// distinct section in the rlib:
///
/// ```ignore
/// #[Splice(begin = 0x2ec70, end = 0x2ec78)]
/// static PRICES: [u32; 2] = [100, 200];
/// ```
#[proc_macro_attribute]
#[allow(non_snake_case)]
pub fn Splice(args: TokenStream, input: TokenStream) -> TokenStream {
    let item = parse_macro_input!(input as Item);

    let meta_list = parse_macro_input!(args with syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated);

    // The range is encoded into the section name and recovered by the tool,
    // which relies on `end - begin` as the length of the region being replaced.
    // A missing or reversed range has to be refused here, where the diagnostic
    // can point at the attribute that got it wrong.
    let (begin, end) = match parse_range(meta_list) {
        (Some(begin), Some(end)) if begin < end => (begin, end),
        (None, _) => return range_error(&item, "#[Splice] requires a `begin` address"),
        (_, None) => return range_error(&item, "#[Splice] requires an `end` address"),
        (Some(begin), Some(end)) => {
            return range_error(
                &item,
                &format!(
                    "#[Splice] range {begin:#x}..{end:#x} is empty or reversed: `end` is \
                     the address just past the last byte being replaced, so it must be \
                     greater than `begin`"
                ),
            )
        }
    };

    let section = format!(".rspl.{:x}.{:x}", begin, end);

    let expanded = match item {
        // Export the function and give it the C ABI unless the author already
        // specified them, so the spliced code matches the target's calling
        // convention and is emitted as an external symbol.
        Item::Fn(mut input_fn) => {
            if matches!(input_fn.vis, syn::Visibility::Inherited) {
                input_fn.vis = syn::parse_quote!(pub);
            }
            if input_fn.sig.abi.is_none() {
                input_fn.sig.abi = Some(syn::parse_quote!(extern "C"));
            }

            let vis = &input_fn.vis;
            let sig = &input_fn.sig;
            let block = &input_fn.block;

            quote! {
                #[unsafe(no_mangle)]
                #[unsafe(link_section = #section)]
                #vis #sig #block
            }
        }
        // A static is exported the same way so its initializer bytes land in
        // the splice section as an external symbol the linker keeps.
        Item::Static(mut input_static) => {
            if matches!(input_static.vis, syn::Visibility::Inherited) {
                input_static.vis = syn::parse_quote!(pub);
            }
            quote! {
                #[unsafe(no_mangle)]
                #[unsafe(link_section = #section)]
                #input_static
            }
        }
        other => {
            return syn::Error::new_spanned(
                other,
                "#[Splice] supports only `fn` and `static` items",
            )
            .to_compile_error()
            .into();
        }
    };

    TokenStream::from(expanded)
}

/// A compile error about the `#[Splice]` range, spanned on the annotated item.
fn range_error(item: &Item, message: &str) -> TokenStream {
    syn::Error::new_spanned(item, message)
        .to_compile_error()
        .into()
}

/// Extract the `begin`/`end` integer addresses from a `#[Splice(..)]` argument
/// list. Either may be missing (reported to the caller as `None`).
fn parse_range(meta_list: impl IntoIterator<Item = Meta>) -> (Option<u64>, Option<u64>) {
    let mut begin = None;
    let mut end = None;
    for meta in meta_list {
        let Meta::NameValue(nv) = meta else { continue };
        let slot = match nv.path.get_ident().map(|i| i.to_string()).as_deref() {
            Some("begin") => &mut begin,
            Some("end") => &mut end,
            _ => continue,
        };
        if let Expr::Lit(expr_lit) = nv.value
            && let Lit::Int(lit_int) = expr_lit.lit
        {
            *slot = lit_int.base10_parse().ok();
        }
    }
    (begin, end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;
    use syn::punctuated::Punctuated;

    fn parse(input: Punctuated<Meta, syn::Token![,]>) -> (Option<u64>, Option<u64>) {
        parse_range(input)
    }

    #[test]
    fn parses_begin_and_end() {
        assert_eq!(
            parse(parse_quote!(begin = 0x1000, end = 0x2000)),
            (Some(0x1000), Some(0x2000))
        );
    }

    #[test]
    fn missing_or_unknown_keys_yield_none() {
        assert_eq!(
            parse(parse_quote!(end = 0x2000, extra = 1)),
            (None, Some(0x2000))
        );
        assert_eq!(parse(parse_quote!()), (None, None));
    }
}
