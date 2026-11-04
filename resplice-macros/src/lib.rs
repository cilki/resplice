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
/// * `begin` - The starting address of the code to replace
/// * `end` - The ending address of the code to replace
///
/// # Example
///
/// The function is exported (`pub`) and given the C ABI (`extern "C"`) so the
/// replacement matches the calling convention the target's caller expects. Both
/// are added automatically when absent, so the annotated function can be written
/// as a plain `fn`; an explicitly written visibility or ABI is left untouched.
/// Any other attribute on the item is kept as written.
///
/// ```ignore
/// #[Splice(begin = 0x1670, end = 0x1680)]
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
    let range = match parse_range(meta_list) {
        Ok(range) => range,
        Err(e) => return e.to_compile_error().into(),
    };
    let (begin, end) = match range {
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
            // A function generic over types or consts is not code until it is
            // instantiated, and `#[no_mangle]` cannot name an instantiation:
            // rustc emits nothing for it, so the splice used to vanish from the
            // rlib with no diagnostic at all, leaving the tool to report "no
            // splices found". Lifetime parameters are erased and are fine.
            if let Some(param) = input_fn
                .sig
                .generics
                .params
                .iter()
                .find(|p| !matches!(p, syn::GenericParam::Lifetime(_)))
            {
                return syn::Error::new_spanned(
                    param,
                    "#[Splice] cannot apply to a function generic over types or consts: \
                     rustc emits no code for it until it is instantiated, so there would be \
                     nothing to splice. Write the replacement for the concrete types the \
                     target uses",
                )
                .to_compile_error()
                .into();
            }
            if matches!(input_fn.vis, syn::Visibility::Inherited) {
                input_fn.vis = syn::parse_quote!(pub);
            }
            if input_fn.sig.abi.is_none() {
                input_fn.sig.abi = Some(syn::parse_quote!(extern "C"));
            }

            // The function's own attributes are kept, like a `static`'s are.
            // Rebuilding it from its visibility, signature and body alone used
            // to drop every one of them without a word: `#[unsafe(naked)]`
            // failed to compile ("`naked_asm!` can only be used in functions
            // marked with `#[unsafe(naked)]`"), while `#[instruction_set]`,
            // `#[target_feature]`, `#[inline(never)]` and doc comments
            // compiled to a function without them.
            quote! {
                #[unsafe(no_mangle)]
                #[unsafe(link_section = #section)]
                #input_fn
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
///
/// Anything else in the list is an error spanned on the token at fault, not
/// skipped. The address is the whole of what the macro is told, and every
/// argument it used to pass over was a range the author meant and did not get.
/// An unknown key or a non-literal value came out as "requires a `begin`
/// address". A repeated key silently took its last value. A suffix was
/// dropped, and Rust lexes `0X1000`, which has no `0X` prefix, as the literal
/// `0` with suffix `X1000`, so that splice was aimed at address 0.
fn parse_range(
    meta_list: impl IntoIterator<Item = Meta>,
) -> syn::Result<(Option<u64>, Option<u64>)> {
    let mut begin = None;
    let mut end = None;
    for meta in meta_list {
        let Meta::NameValue(nv) = meta else {
            return Err(syn::Error::new_spanned(
                meta,
                "expected `begin = <address>` or `end = <address>`",
            ));
        };
        let key = nv.path.get_ident().map(|i| i.to_string());
        let slot = match key.as_deref() {
            Some("begin") => &mut begin,
            Some("end") => &mut end,
            _ => {
                return Err(syn::Error::new_spanned(
                    &nv.path,
                    "unknown #[Splice] argument; expected `begin` or `end`",
                ));
            }
        };
        let key = key.unwrap();
        if slot.is_some() {
            return Err(syn::Error::new_spanned(
                &nv.path,
                format!("`{key}` is given more than once"),
            ));
        }
        let Expr::Lit(syn::ExprLit {
            lit: Lit::Int(lit_int),
            ..
        }) = &nv.value
        else {
            return Err(syn::Error::new_spanned(
                &nv.value,
                format!("`{key}` must be an integer literal address, like `0x1000`"),
            ));
        };
        match lit_int.suffix() {
            "" | "u64" | "usize" => {}
            s if lit_int.base10_digits() == "0" && s.starts_with(['X', 'B', 'O']) => {
                return Err(syn::Error::new_spanned(
                    lit_int,
                    format!(
                        "`0{s}` is the literal `0` with suffix `{s}`, not an address: \
                         Rust's radix prefixes are lowercase -- write `0{}`",
                        s.to_ascii_lowercase()
                    ),
                ));
            }
            s => {
                return Err(syn::Error::new_spanned(
                    lit_int,
                    format!("unexpected suffix `{s}` on `{key}`; an address takes no suffix"),
                ));
            }
        }
        *slot = Some(lit_int.base10_parse::<u64>()?);
    }
    Ok((begin, end))
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;
    use syn::punctuated::Punctuated;

    fn parse(input: Punctuated<Meta, syn::Token![,]>) -> (Option<u64>, Option<u64>) {
        parse_range(input).unwrap()
    }

    /// The message `parse_range` refuses `input` with.
    fn refusal(input: Punctuated<Meta, syn::Token![,]>) -> String {
        parse_range(input).unwrap_err().to_string()
    }

    #[test]
    fn parses_begin_and_end() {
        assert_eq!(
            parse(parse_quote!(begin = 0x1000, end = 0x2000)),
            (Some(0x1000), Some(0x2000))
        );
        assert_eq!(
            parse(parse_quote!(end = 0x40_2000u64, begin = 4096usize)),
            (Some(0x1000), Some(0x40_2000))
        );
    }

    #[test]
    fn missing_keys_yield_none() {
        assert_eq!(parse(parse_quote!(end = 0x2000)), (None, Some(0x2000)));
        assert_eq!(parse(parse_quote!()), (None, None));
    }

    /// `0X1000` is the literal `0` with suffix `X1000`; reading its digits
    /// alone aimed the splice at address 0.
    #[test]
    fn uppercase_radix_prefix_is_refused_not_read_as_zero() {
        let msg = refusal(parse_quote!(begin = 0X1000, end = 0x2000));
        assert!(msg.contains("write `0x1000`"), "{msg}");
        let msg = refusal(parse_quote!(begin = 0x1000, end = 0B11));
        assert!(msg.contains("write `0b11`"), "{msg}");
        let msg = refusal(parse_quote!(begin = 0x1000u8, end = 0x2000));
        assert!(msg.contains("unexpected suffix `u8`"), "{msg}");
    }

    #[test]
    fn unknown_repeated_and_non_literal_arguments_are_refused() {
        let msg = refusal(parse_quote!(begin = 0x1000, ends = 0x2000));
        assert!(msg.contains("unknown #[Splice] argument"), "{msg}");
        let msg = refusal(parse_quote!(begin = 0x1000, end = 0x2000, begin = 0x10));
        assert!(msg.contains("`begin` is given more than once"), "{msg}");
        let msg = refusal(parse_quote!(begin = BASE, end = 0x2000));
        assert!(msg.contains("must be an integer literal"), "{msg}");
        let msg = refusal(parse_quote!(begin, end = 0x2000));
        assert!(msg.contains("expected `begin = <address>`"), "{msg}");
        let msg = refusal(parse_quote!(begin = 0x1_0000_0000_0000_0000, end = 0x2000));
        assert!(msg.contains("too large"), "{msg}");
    }
}
