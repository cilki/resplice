use anyhow::Result;
use clap::Parser;
use resplice::{link_rlib, Binary};
use std::io::{BufWriter, Write};
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
    /// heap and is overwritten during play. Pass a known-free address to place
    /// it deliberately. Three things are required of it, each reported as an
    /// error rather than written into a binary that misbehaves later.
    ///
    /// It must be aligned to the target's page size -- the largest `p_align` of
    /// its `PT_LOAD` segments, which is 64K on a typical aarch64 image rather
    /// than 4K -- or the loader refuses the image outright.
    ///
    /// It must be free: a base whose segment would share a page with one of the
    /// target's own `PT_LOAD` segments replaces that mapping instead of adding
    /// to it, leaving the target running with its own code overwritten.
    ///
    /// It must be within reach of the splices. References into the injected
    /// segment, and the trampoline of an oversized splice, are PC-relative
    /// through fields narrower than a pointer -- 32 bits on x86-64, so 2GB, and
    /// only 128MB for an aarch64 `bl`.
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

    // One line per splice, and a crate that reimplements a few thousand
    // functions has a few thousand of them. `println!` takes the lock and
    // flushes once per line -- stdout is line-buffered whenever it is not a
    // terminal too -- so reporting the work cost a write syscall per splice,
    // which was a measurable fraction of the whole run. Buffer the report and
    // flush it once.
    let output = cli.output_binary.display();
    let stdout = std::io::stdout();
    let mut report = BufWriter::new(stdout.lock());
    writeln!(report, "applied {} splice(s) to {output}:", applied.len())?;
    for splice in &applied {
        let how = if splice.trampoline {
            " (trampoline to injected segment)"
        } else {
            ""
        };
        writeln!(
            report,
            "  {:#x}..{:#x} <- {} bytes{how}",
            splice.begin, splice.end, splice.code_len
        )?;
    }
    report.flush()?;

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
