use anyhow::Result;
use resplice::{link_rlib, Binary};
use std::env;
use std::path::Path;
use std::process::ExitCode;

fn run() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!(
            "usage: {} <original-binary> <reimpl.rlib> <output-binary>",
            args.first().map(String::as_str).unwrap_or("resplice")
        );
        std::process::exit(2);
    }

    let original = &args[1];
    let rlib = &args[2];
    let output = &args[3];

    let mut binary = Binary::load(original)?;
    let applied = link_rlib(&mut binary, Path::new(rlib))?;
    if applied.is_empty() {
        eprintln!("warning: no splices found in {rlib}");
    }
    binary.save(output)?;

    println!("applied {} splice(s) to {output}:", applied.len());
    for splice in &applied {
        let how = if splice.trampoline {
            " (trampoline to injected segment)"
        } else {
            ""
        };
        println!(
            "  {:#x}..{:#x} <- {} bytes{how}",
            splice.begin, splice.end, splice.code_len
        );
    }

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}
