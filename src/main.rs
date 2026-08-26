use anyhow::Result;
use resplice::{apply_splices, read_splices_from_rlib, Binary};
use std::env;
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

    let splices = read_splices_from_rlib(rlib)?;
    if splices.is_empty() {
        eprintln!("warning: no splices found in {rlib}");
    }

    let mut binary = Binary::load(original)?;
    apply_splices(&mut binary, &splices)?;
    binary.save(output)?;

    println!("applied {} splice(s) to {output}:", splices.len());
    for splice in &splices {
        println!(
            "  {:#x}..{:#x} <- {} bytes",
            splice.begin,
            splice.end,
            splice.code.len()
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
