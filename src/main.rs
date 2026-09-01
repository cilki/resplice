use anyhow::Result;
use clap::Parser;
use resplice::{link_rlib, Binary};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// Splice Rust reimplementations into an existing binary.
#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    /// The original binary to patch.
    original_binary: PathBuf,
    /// The reimplementation rlib containing splices.
    rlib: PathBuf,
    /// Where to write the patched binary.
    output_binary: PathBuf,
}

fn run(cli: Cli) -> Result<()> {
    let mut binary = Binary::load(&cli.original_binary)?;
    let applied = link_rlib(&mut binary, Path::new(&cli.rlib))?;
    if applied.is_empty() {
        eprintln!("warning: no splices found in {}", cli.rlib.display());
    }
    binary.save(&cli.output_binary)?;

    let output = cli.output_binary.display();
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
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}
