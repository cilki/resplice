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

    /// Virtual address for the injected segment, e.g. `0x1c0000`.
    ///
    /// By default the segment goes one page past the end of the image. That is
    /// unsafe when something else already owns that address -- console binaries
    /// commonly allocate from the end of `.bss`, so the default lands in the
    /// heap and is overwritten during play. Pass a known-free, page-aligned
    /// address to place it deliberately.
    #[arg(long, value_parser = parse_addr)]
    inject_base: Option<u64>,
}

/// Parse a `0x`-prefixed hex or plain decimal address.
fn parse_addr(s: &str) -> Result<u64, String> {
    let t = s.trim();
    let r = match t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        Some(hex) => u64::from_str_radix(hex, 16),
        None => t.parse(),
    };
    r.map_err(|e| format!("invalid address {s:?}: {e}"))
}

fn run(cli: Cli) -> Result<()> {
    let mut binary = Binary::load(&cli.original_binary)?;
    binary.set_injected_base(cli.inject_base);
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
