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

    let mut begin_addr: Option<u64> = None;
    let mut end_addr: Option<u64> = None;

    for meta in meta_list {
        if let Meta::NameValue(nv) = meta {
            let name = nv.path.get_ident().map(|i| i.to_string());

            match name.as_deref() {
                Some("begin") => {
                    if let Expr::Lit(expr_lit) = nv.value
                        && let Lit::Int(lit_int) = expr_lit.lit {
                            begin_addr = lit_int.base10_parse().ok();
                        }
                }
                Some("end") => {
                    if let Expr::Lit(expr_lit) = nv.value
                        && let Lit::Int(lit_int) = expr_lit.lit {
                            end_addr = lit_int.base10_parse().ok();
                        }
                }
                _ => {}
            }
        }
    }

    let begin = begin_addr.expect("Splice attribute requires 'begin' parameter");
    let end = end_addr.expect("Splice attribute requires 'end' parameter");

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

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;
    use syn::punctuated::Punctuated;

    #[test]
    fn test_section_name_formatting() {
        let begin: u64 = 0x1670;
        let end: u64 = 0x1680;
        assert_eq!(format!(".rspl.{:x}.{:x}", begin, end), ".rspl.1670.1680");
    }

    #[test]
    fn test_address_parsing_logic() {
        let meta_list: Punctuated<Meta, syn::Token![,]> =
            parse_quote!(begin = 0x1000, end = 0x2000);

        let mut begin_addr: Option<u64> = None;
        let mut end_addr: Option<u64> = None;

        for meta in meta_list {
            if let Meta::NameValue(nv) = meta {
                let name = nv.path.get_ident().map(|i| i.to_string());

                match name.as_deref() {
                    Some("begin") => {
                        if let Expr::Lit(expr_lit) = nv.value
                            && let Lit::Int(lit_int) = expr_lit.lit {
                                begin_addr = lit_int.base10_parse().ok();
                            }
                    }
                    Some("end") => {
                        if let Expr::Lit(expr_lit) = nv.value
                            && let Lit::Int(lit_int) = expr_lit.lit {
                                end_addr = lit_int.base10_parse().ok();
                            }
                    }
                    _ => {}
                }
            }
        }

        assert_eq!(begin_addr, Some(0x1000));
        assert_eq!(end_addr, Some(0x2000));
    }
}
