//! Architecture-specific pieces of the linker: endianness-aware integer
//! encoding, relocation application, and the small facts (GOT need, HI16/LO16
//! pairing) that the generic driver in [`crate::link`] needs to stay
//! architecture-agnostic.
//!
//! Only x86-64 is exercised end-to-end in this repository; the AArch64, ARM and
//! MIPS encoders are implemented from their ABI documents and pinned by unit
//! tests that assert exact instruction-field encodings. MIPS in particular is
//! spec-derived and has known gaps (see [`apply_mips`]).

use crate::Architecture;
use anyhow::{anyhow, bail, Result};
use object::elf::{
    R_AARCH64_ABS32, R_AARCH64_ABS64, R_AARCH64_ADD_ABS_LO12_NC, R_AARCH64_ADR_GOT_PAGE,
    R_AARCH64_ADR_PREL_PG_HI21, R_AARCH64_CALL26, R_AARCH64_JUMP26, R_AARCH64_LD64_GOT_LO12_NC,
    R_AARCH64_LDST128_ABS_LO12_NC, R_AARCH64_LDST16_ABS_LO12_NC, R_AARCH64_LDST32_ABS_LO12_NC,
    R_AARCH64_LDST64_ABS_LO12_NC, R_AARCH64_LDST8_ABS_LO12_NC, R_AARCH64_PREL32, R_AARCH64_PREL64,
    R_ARM_ABS32, R_ARM_CALL, R_ARM_GOT_PREL, R_ARM_JUMP24, R_ARM_MOVT_ABS, R_ARM_MOVW_ABS_NC,
    R_ARM_PC24, R_ARM_REL32, R_MIPS_26, R_MIPS_32, R_MIPS_64, R_MIPS_HI16, R_MIPS_LO16, R_MIPS_PC16,
    R_X86_64_32, R_X86_64_32S, R_X86_64_64, R_X86_64_GOTPCREL, R_X86_64_GOTPCRELX, R_X86_64_PC32,
    R_X86_64_PC64, R_X86_64_PLT32, R_X86_64_REX_GOTPCRELX,
};
use object::elf::RelocationType;

/// `R_MIPS_PC32` is absent from `object` 0.40's constant list.
const R_MIPS_PC32: RelocationType = RelocationType(248);

/// Byte order of a target binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endian {
    Little,
    Big,
}

impl Endian {
    /// Read a `width`-byte (1/2/4/8) unsigned integer from `buf` at `off`.
    pub fn read_uint(self, buf: &[u8], off: usize, width: usize) -> u64 {
        let mut v = 0u64;
        match self {
            Endian::Little => {
                for i in 0..width {
                    v |= (buf[off + i] as u64) << (8 * i);
                }
            }
            Endian::Big => {
                for i in 0..width {
                    v = (v << 8) | buf[off + i] as u64;
                }
            }
        }
        v
    }

    /// Write the low `width` bytes (1/2/4/8) of `val` to `buf` at `off`.
    pub fn write_uint(self, buf: &mut [u8], off: usize, width: usize, val: u64) {
        let le = val.to_le_bytes();
        match self {
            Endian::Little => buf[off..off + width].copy_from_slice(&le[..width]),
            Endian::Big => {
                for i in 0..width {
                    buf[off + i] = le[width - 1 - i];
                }
            }
        }
    }
}

/// GOT-slot width for a target of the given pointer size.
pub fn got_slot_width(is_64: bool) -> usize {
    if is_64 {
        8
    } else {
        4
    }
}

/// Whether a relocation of this type is resolved through a synthesized GOT slot
/// holding the symbol's address (so the driver knows to allocate one).
pub fn needs_got(arch: Architecture, r_type: RelocationType) -> bool {
    match arch {
        Architecture::X86 | Architecture::X86_64 => matches!(
            r_type,
            R_X86_64_GOTPCREL | R_X86_64_GOTPCRELX | R_X86_64_REX_GOTPCRELX
        ),
        Architecture::Arm64 => matches!(r_type, R_AARCH64_ADR_GOT_PAGE | R_AARCH64_LD64_GOT_LO12_NC),
        Architecture::Arm => r_type == R_ARM_GOT_PREL,
        // MIPS GOT relocations are $gp-relative into the target's own GOT and
        // cannot be satisfied by an injected segment (see `apply_mips`).
        Architecture::Mips | Architecture::Mips64 => false,
    }
}

/// Whether the field this relocation writes holds an absolute address, as
/// opposed to a displacement (or a slice of one) that a uniform shift of the
/// whole image leaves unchanged.
///
/// The distinction only matters for a position-independent target, where a
/// link-time absolute address is wrong by the load base and nothing fixes it up
/// (see [`crate::Binary::is_position_independent`]).
///
/// The low-half companions of a PC-relative high half are deliberately *not*
/// here: aarch64's `*_ABS_LO12_NC` family is named for taking the low 12 bits
/// of `S + A`, but it completes an `ADRP` that is PC-relative, and a load base
/// is page-aligned — so the bits it writes are the same wherever the image
/// lands. MIPS `R_MIPS_LO16` is listed because its partner `R_MIPS_HI16` builds
/// a genuinely absolute 32-bit address rather than a page displacement, so the
/// pair stands or falls together.
pub fn writes_absolute_address(arch: Architecture, r_type: RelocationType) -> bool {
    match arch {
        Architecture::X86 | Architecture::X86_64 => {
            matches!(r_type, R_X86_64_64 | R_X86_64_32 | R_X86_64_32S)
        }
        Architecture::Arm64 => matches!(r_type, R_AARCH64_ABS64 | R_AARCH64_ABS32),
        Architecture::Arm => matches!(r_type, R_ARM_ABS32 | R_ARM_MOVW_ABS_NC | R_ARM_MOVT_ABS),
        Architecture::Mips | Architecture::Mips64 => matches!(
            r_type,
            R_MIPS_32 | R_MIPS_64 | R_MIPS_26 | R_MIPS_HI16 | R_MIPS_LO16
        ),
    }
}

/// True for `R_MIPS_HI16`, whose resolved value depends on the low half from a
/// paired `R_MIPS_LO16` (only under the REL scheme, where the addend is stored
/// in the instruction rather than carried explicitly).
pub fn is_mips_hi16(arch: Architecture, r_type: RelocationType) -> bool {
    matches!(arch, Architecture::Mips | Architecture::Mips64) && r_type == R_MIPS_HI16
}

/// True for `R_MIPS_LO16`, the partner of an `R_MIPS_HI16`.
pub fn is_mips_lo16(arch: Architecture, r_type: RelocationType) -> bool {
    matches!(arch, Architecture::Mips | Architecture::Mips64) && r_type == R_MIPS_LO16
}

/// Sign-extend a 16-bit MIPS instruction immediate stored at `off` (the low
/// half-word of the 32-bit instruction) into an addend.
pub fn mips_lo16_addend(endian: Endian, buf: &[u8], off: usize) -> i64 {
    let insn = endian.read_uint(buf, off, 4) as u32;
    (insn & 0xffff) as i16 as i64
}

/// Refuse a resolved value that the relocation field about to be written cannot
/// hold.
///
/// Every narrow field below is written with a truncating cast, so a value that
/// does not fit is silently reduced to its low bits: the patched binary calls,
/// jumps to, or loads from an address that has nothing to do with the symbol it
/// was relocated against, and `resplice` reports a successful splice. Nothing
/// malformed is needed to get there -- on a 64-bit target an `--inject-base`
/// more than 2GB from the splice site puts every PC-relative reference into the
/// injected segment out of range of its field.
///
/// The aarch64 branch and ADRP encoders checked their own ranges from the
/// start; this is the same refusal for the fields that did not.
fn does_not_fit(kind: &str, value: i128, field: &str) -> anyhow::Error {
    anyhow!(
        "{kind} {value:#x} does not fit the {field} field this relocation writes; \
         truncating it would leave the reference pointing at an unrelated \
         address. The symbol is too far from the relocated site -- for a \
         reference into the injected segment, choose an --inject-base closer to \
         the splice"
    )
}

/// Width of the field the encoder for `r_type` reads and writes at the
/// relocated offset: eight bytes for the handful of 64-bit relocations, and a
/// four-byte instruction (or word) for everything else, including the types
/// [`apply`] goes on to reject.
fn field_width(arch: Architecture, r_type: RelocationType) -> usize {
    let wide = match arch {
        Architecture::X86 | Architecture::X86_64 => matches!(r_type, R_X86_64_64 | R_X86_64_PC64),
        Architecture::Arm64 => matches!(r_type, R_AARCH64_ABS64 | R_AARCH64_PREL64),
        Architecture::Arm => false,
        Architecture::Mips | Architecture::Mips64 => r_type == R_MIPS_64,
    };
    if wide { 8 } else { 4 }
}

/// Apply one relocation, writing the resolved field into `buf` at `off`.
///
/// `buf` is the section the relocation belongs to, and nothing beyond it: the
/// field at `off` is required to lie inside it, so a malformed object cannot
/// reach past the section it is relocating.
///
/// - `s` is the resolved symbol virtual address, `a` the explicit RELA addend
///   (0 for REL — those addends live in the instruction and are recovered here),
///   `p` the virtual address of the relocated field.
/// - `got_va` is the synthesized GOT slot's address when [`needs_got`] holds.
/// - `has_implicit` marks REL relocations (ARM, MIPS o32), whose addend must be
///   read back out of the field.
/// - `pair_lo` carries the paired `R_MIPS_LO16` addend for an `R_MIPS_HI16`.
#[allow(clippy::too_many_arguments)]
pub fn apply(
    arch: Architecture,
    endian: Endian,
    r_type: RelocationType,
    buf: &mut [u8],
    off: usize,
    s: u64,
    a: i64,
    p: u64,
    got_va: Option<u64>,
    has_implicit: bool,
    pair_lo: Option<i64>,
) -> Result<()> {
    let width = field_width(arch, r_type);
    if off.checked_add(width).is_none_or(|end| end > buf.len()) {
        bail!(
            "relocation at offset {off:#x} has a {width}-byte field that runs past the \
             end of its {}-byte section",
            buf.len()
        );
    }

    match arch {
        Architecture::X86 | Architecture::X86_64 => {
            apply_x86_64(endian, r_type, buf, off, s, a, p, got_va, arch.is_64())
        }
        Architecture::Arm64 => apply_aarch64(endian, r_type, buf, off, s, a, p, got_va),
        Architecture::Arm => apply_arm(endian, r_type, buf, off, s, p, got_va),
        Architecture::Mips | Architecture::Mips64 => {
            apply_mips(
                endian,
                r_type,
                buf,
                off,
                s,
                a,
                p,
                has_implicit,
                pair_lo,
                arch.is_64(),
            )
        }
    }
}

fn unsupported(arch: &str, r_type: RelocationType) -> anyhow::Error {
    anyhow!("unsupported {arch} relocation type {}", r_type.0)
}

#[allow(clippy::too_many_arguments)]
fn apply_x86_64(
    endian: Endian,
    r_type: RelocationType,
    buf: &mut [u8],
    off: usize,
    s: u64,
    a: i64,
    p: u64,
    got_va: Option<u64>,
    is_64: bool,
) -> Result<()> {
    // A 32-bit field cannot hold every value a reference can take on a 64-bit
    // target, so each one is range-checked before it is written. On 32-bit x86
    // every address fits by construction and the truncation *is* the right
    // modular arithmetic (a backwards reference wraps), so checking there would
    // reject legitimate relocations.
    let pc32 = |v: i64| -> Result<u64> {
        if is_64 {
            i32::try_from(v)
                .map_err(|_| does_not_fit("PC-relative displacement", v as i128, "signed 32-bit"))?;
        }
        Ok((v as i32) as u32 as u64)
    };
    match r_type {
        R_X86_64_64 => endian.write_uint(buf, off, 8, s.wrapping_add(a as u64)),
        R_X86_64_PC64 => endian.write_uint(buf, off, 8, (s as i64 + a - p as i64) as u64),
        R_X86_64_PC32 | R_X86_64_PLT32 => {
            let v = pc32(s as i64 + a - p as i64)?;
            endian.write_uint(buf, off, 4, v)
        }
        R_X86_64_32 => {
            let v = s.wrapping_add(a as u64);
            if is_64 {
                u32::try_from(v).map_err(|_| {
                    does_not_fit("absolute address", v as i128, "unsigned 32-bit")
                })?;
            }
            endian.write_uint(buf, off, 4, v as u32 as u64)
        }
        R_X86_64_32S => {
            let v = s.wrapping_add(a as u64) as i64;
            if is_64 {
                i32::try_from(v)
                    .map_err(|_| does_not_fit("absolute address", v as i128, "signed 32-bit"))?;
            }
            endian.write_uint(buf, off, 4, v as i32 as u32 as u64)
        }
        R_X86_64_GOTPCREL | R_X86_64_GOTPCRELX | R_X86_64_REX_GOTPCRELX => {
            let g = got_va.ok_or_else(|| anyhow!("missing GOT slot for GOTPCREL"))?;
            let v = pc32(g as i64 + a - p as i64)?;
            endian.write_uint(buf, off, 4, v);
        }
        other => return Err(unsupported("x86-64", other)),
    }
    Ok(())
}

/// Replace the bits selected by `mask` in the 32-bit instruction at `off` with
/// `value` (already positioned), preserving the rest of the opcode.
fn patch_insn(endian: Endian, buf: &mut [u8], off: usize, mask: u32, value: u32) {
    let insn = endian.read_uint(buf, off, 4) as u32;
    let insn = (insn & !mask) | (value & mask);
    endian.write_uint(buf, off, 4, insn as u64);
}

#[allow(clippy::too_many_arguments)]
fn apply_aarch64(
    endian: Endian,
    r_type: RelocationType,
    buf: &mut [u8],
    off: usize,
    s: u64,
    a: i64,
    p: u64,
    got_va: Option<u64>,
) -> Result<()> {
    let page = |x: i64| x & !0xfff;
    // The ABI's range for both 32-bit word relocations is -2^31 <= X < 2^32:
    // the field holds either representation of a 32-bit value, and nothing
    // wider. aarch64 is always 64-bit, so unlike x86-64 this is unconditional.
    let word32 = |v: i64, kind: &str| -> Result<u64> {
        if i32::try_from(v).is_err() && u32::try_from(v).is_err() {
            return Err(does_not_fit(kind, v as i128, "32-bit"));
        }
        Ok(v as u32 as u64)
    };
    match r_type {
        R_AARCH64_ABS64 => endian.write_uint(buf, off, 8, s.wrapping_add(a as u64)),
        R_AARCH64_ABS32 => {
            let v = word32(s.wrapping_add(a as u64) as i64, "absolute address")?;
            endian.write_uint(buf, off, 4, v)
        }
        R_AARCH64_PREL64 => endian.write_uint(buf, off, 8, (s as i64 + a - p as i64) as u64),
        R_AARCH64_PREL32 => {
            let v = word32(s as i64 + a - p as i64, "PC-relative displacement")?;
            endian.write_uint(buf, off, 4, v)
        }
        R_AARCH64_CALL26 | R_AARCH64_JUMP26 => {
            let x = s as i64 + a - p as i64;
            if x & 3 != 0 || !(-(1 << 27)..(1 << 27)).contains(&x) {
                bail!("aarch64 branch target {x:#x} out of range");
            }
            let imm26 = ((x >> 2) as u32) & 0x03ff_ffff;
            patch_insn(endian, buf, off, 0x03ff_ffff, imm26);
        }
        // ADRP: page offset of (S + A) relative to the page of P, imm21 split
        // into immlo (bits 30:29) and immhi (bits 23:5).
        R_AARCH64_ADR_PREL_PG_HI21 => {
            let x = page(s as i64 + a) - page(p as i64);
            adrp(endian, buf, off, x)?;
        }
        R_AARCH64_ADD_ABS_LO12_NC => {
            let imm = ((s as i64 + a) & 0xfff) as u32;
            patch_insn(endian, buf, off, 0xfff << 10, imm << 10);
        }
        R_AARCH64_LDST8_ABS_LO12_NC
        | R_AARCH64_LDST16_ABS_LO12_NC
        | R_AARCH64_LDST32_ABS_LO12_NC
        | R_AARCH64_LDST64_ABS_LO12_NC
        | R_AARCH64_LDST128_ABS_LO12_NC => {
            let scale = match r_type {
                R_AARCH64_LDST8_ABS_LO12_NC => 0,
                R_AARCH64_LDST16_ABS_LO12_NC => 1,
                R_AARCH64_LDST32_ABS_LO12_NC => 2,
                R_AARCH64_LDST64_ABS_LO12_NC => 3,
                _ => 4,
            };
            let imm = (((s as i64 + a) & 0xfff) as u32) >> scale;
            patch_insn(endian, buf, off, 0xfff << 10, imm << 10);
        }
        // GOT: the ADRP/LDR pair addresses the synthesized slot holding S.
        R_AARCH64_ADR_GOT_PAGE => {
            let g = got_va.ok_or_else(|| anyhow!("missing GOT slot for ADR_GOT_PAGE"))?;
            let x = page(g as i64 + a) - page(p as i64);
            adrp(endian, buf, off, x)?;
        }
        R_AARCH64_LD64_GOT_LO12_NC => {
            let g = got_va.ok_or_else(|| anyhow!("missing GOT slot for LD64_GOT_LO12"))?;
            let imm = (((g as i64 + a) & 0xff8) as u32) >> 3;
            patch_insn(endian, buf, off, 0xfff << 10, imm << 10);
        }
        other => return Err(unsupported("aarch64", other)),
    }
    Ok(())
}

/// Encode a page displacement `x` (a multiple of 4096) into an ADRP at `off`.
fn adrp(endian: Endian, buf: &mut [u8], off: usize, x: i64) -> Result<()> {
    if x & 0xfff != 0 || !(-(1 << 32)..(1 << 32)).contains(&x) {
        bail!("aarch64 ADRP displacement {x:#x} out of range");
    }
    let imm = (x >> 12) as u32 & 0x1f_ffff; // 21-bit
    let immlo = imm & 0x3;
    let immhi = (imm >> 2) & 0x7ffff;
    patch_insn(endian, buf, off, (0x3 << 29) | (0x7ffff << 5), (immlo << 29) | (immhi << 5));
    Ok(())
}

fn apply_arm(
    endian: Endian,
    r_type: RelocationType,
    buf: &mut [u8],
    off: usize,
    s: u64,
    p: u64,
    got_va: Option<u64>,
) -> Result<()> {
    // ARM ELF uses the REL scheme: the addend is stored in the field.
    let insn = endian.read_uint(buf, off, 4) as u32;
    match r_type {
        R_ARM_ABS32 => {
            let a = insn as i32 as i64;
            endian.write_uint(buf, off, 4, (s as i64 + a) as u32 as u64);
        }
        R_ARM_REL32 => {
            let a = insn as i32 as i64;
            endian.write_uint(buf, off, 4, (s as i64 + a - p as i64) as u32 as u64);
        }
        R_ARM_CALL | R_ARM_JUMP24 | R_ARM_PC24 => {
            let a = (((insn & 0x00ff_ffff) << 8) as i32 >> 8) as i64 * 4; // sign-extend imm24<<2
            let x = s as i64 + a - p as i64;
            if x & 3 != 0 || !(-(1 << 25)..(1 << 25)).contains(&x) {
                bail!("arm branch target {x:#x} out of range");
            }
            let imm24 = ((x >> 2) as u32) & 0x00ff_ffff;
            patch_insn(endian, buf, off, 0x00ff_ffff, imm24);
        }
        R_ARM_MOVW_ABS_NC | R_ARM_MOVT_ABS => {
            let a = (((insn & 0xf0000) >> 4) | (insn & 0xfff)) as i16 as i64;
            let full = s as i64 + a;
            let val = if r_type == R_ARM_MOVT_ABS {
                ((full >> 16) & 0xffff) as u32
            } else {
                (full & 0xffff) as u32
            };
            let field = ((val & 0xf000) << 4) | (val & 0x0fff);
            patch_insn(endian, buf, off, 0x000f_0fff, field);
        }
        R_ARM_GOT_PREL => {
            let g = got_va.ok_or_else(|| anyhow!("missing GOT slot for R_ARM_GOT_PREL"))?;
            let a = insn as i32 as i64;
            endian.write_uint(buf, off, 4, (g as i64 + a - p as i64) as u32 as u64);
        }
        other => return Err(unsupported("arm", other)),
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn apply_mips(
    endian: Endian,
    r_type: RelocationType,
    buf: &mut [u8],
    off: usize,
    s: u64,
    a: i64,
    p: u64,
    has_implicit: bool,
    pair_lo: Option<i64>,
    is_64: bool,
) -> Result<()> {
    let insn = endian.read_uint(buf, off, 4) as u32;
    // A 32-bit field holds any mips32 address, so only a mips64 target can put
    // a value out of its reach.
    let word32 = |v: i64, kind: &str| -> Result<u64> {
        if is_64 && i32::try_from(v).is_err() && u32::try_from(v).is_err() {
            return Err(does_not_fit(kind, v as i128, "32-bit"));
        }
        Ok(v as u32 as u64)
    };
    match r_type {
        R_MIPS_32 => {
            let a = if has_implicit { insn as i64 } else { a };
            let v = word32(s as i64 + a, "absolute address")?;
            endian.write_uint(buf, off, 4, v);
        }
        R_MIPS_64 => {
            let a = if has_implicit {
                endian.read_uint(buf, off, 8) as i64
            } else {
                a
            };
            endian.write_uint(buf, off, 8, s.wrapping_add(a as u64));
        }
        R_MIPS_26 => {
            let a = if has_implicit {
                ((insn & 0x03ff_ffff) << 2) as i64
            } else {
                a
            };
            let target = (s as i64 + a) as u64;
            // `j`/`jal` encode only bits 27:2 of the target; the top bits come
            // from the delay slot's own PC, so the two have to share a 256MB
            // region and the target has to be 4-byte aligned. Truncating
            // instead jumps into whatever sits at that offset of the *calling*
            // region. `Binary::jump_bytes` refuses the same thing for the
            // trampoline it encodes by hand.
            if target & 3 != 0 {
                bail!("mips jump target {target:#x} is not 4-byte aligned");
            }
            let delay_pc = p.wrapping_add(4);
            if (delay_pc >> 28) != (target >> 28) {
                bail!(
                    "mips jump from {p:#x} to {target:#x} crosses a 256MB region \
                     boundary and cannot be encoded as `j`/`jal`"
                );
            }
            let field = ((target >> 2) as u32) & 0x03ff_ffff;
            patch_insn(endian, buf, off, 0x03ff_ffff, field);
        }
        R_MIPS_HI16 => {
            // Combine this instruction's high half with the paired LO16 low half
            // so the sign of the low half carries into the high half correctly.
            let ahi = if has_implicit {
                ((insn & 0xffff) << 16) as i64
            } else {
                a << 16
            };
            let alo = pair_lo.unwrap_or(0);
            let full = ahi + alo + s as i64;
            // A HI16/LO16 pair spans 32 bits and no more. Reaching further on
            // mips64 takes the HIGHEST/HIGHER sequence, which `apply` rejects
            // as unsupported rather than mis-encoding -- so a pair that cannot
            // express the address has to be rejected here too.
            word32(full, "absolute address")?;
            let value = ((full + 0x8000) >> 16) & 0xffff;
            patch_insn(endian, buf, off, 0xffff, value as u32);
        }
        R_MIPS_LO16 => {
            let a = if has_implicit { (insn & 0xffff) as i16 as i64 } else { a };
            let value = (s as i64 + a) & 0xffff;
            patch_insn(endian, buf, off, 0xffff, value as u32);
        }
        R_MIPS_PC16 => {
            let a = if has_implicit {
                ((insn & 0xffff) as i16 as i64) << 2
            } else {
                a
            };
            let x = s as i64 + a - p as i64;
            // A signed 16-bit field scaled by four: ±128KB of instructions.
            if x & 3 != 0 || !(-(1 << 17)..(1 << 17)).contains(&x) {
                bail!("mips branch target {x:#x} out of range");
            }
            let field = ((x >> 2) as u32) & 0xffff;
            patch_insn(endian, buf, off, 0xffff, field);
        }
        R_MIPS_PC32 => {
            let a = if has_implicit { insn as i64 } else { a };
            let v = word32(s as i64 + a - p as i64, "PC-relative displacement")?;
            endian.write_uint(buf, off, 4, v);
        }
        other => {
            return Err(anyhow!(
                "unsupported mips relocation type {other} (note: $gp-relative GOT \
                 relocations that bind new symbols are not supported by segment injection)"
            ))
        }
    }
    Ok(())
}

/// Render an architecture and byte order for diagnostics, used when reporting
/// that an rlib member does not match the target it is being spliced into.
pub fn describe(arch: Architecture, endian: Endian) -> String {
    let a = match arch {
        Architecture::X86 => "x86",
        Architecture::X86_64 => "x86-64",
        Architecture::Arm => "arm",
        Architecture::Arm64 => "aarch64",
        Architecture::Mips => "mips",
        Architecture::Mips64 => "mips64",
    };
    let e = match endian {
        Endian::Little => "LE",
        Endian::Big => "BE",
    };
    let c = if arch.is_64() { "64-bit" } else { "32-bit" };
    format!("{a} ({e}, {c})")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Apply one relocation to a fresh 4-byte instruction word `base` and return
    /// the resulting word (decoded in `endian`).
    #[allow(clippy::too_many_arguments)]
    fn insn(
        arch: Architecture,
        endian: Endian,
        r_type: RelocationType,
        base: u32,
        s: u64,
        a: i64,
        p: u64,
        got_va: Option<u64>,
        has_implicit: bool,
        pair_lo: Option<i64>,
    ) -> u32 {
        let mut buf = [0u8; 4];
        endian.write_uint(&mut buf, 0, 4, base as u64);
        apply(
            arch, endian, r_type, &mut buf, 0, s, a, p, got_va, has_implicit, pair_lo,
        )
        .unwrap();
        endian.read_uint(&buf, 0, 4) as u32
    }

    #[test]
    fn endian_roundtrip() {
        let mut b = [0u8; 8];
        Endian::Big.write_uint(&mut b, 0, 4, 0x1122_3344);
        assert_eq!(&b[..4], &[0x11, 0x22, 0x33, 0x44]);
        assert_eq!(Endian::Big.read_uint(&b, 0, 4), 0x1122_3344);
        Endian::Little.write_uint(&mut b, 0, 8, 0x1122_3344_5566_7788);
        assert_eq!(Endian::Little.read_uint(&b, 0, 8), 0x1122_3344_5566_7788);
    }

    /// `buf` is the relocation's own section, so a field that does not fit
    /// inside it is an error -- never a write into the bytes that follow.
    #[test]
    fn field_outside_its_section_is_rejected() {
        let (a, le) = (Architecture::X86_64, Endian::Little);
        let mut buf = [0u8; 8];

        // An 8-byte ABS64 field needs the whole buffer; one byte in, it does not
        // fit, even though the 4-byte relocation types still would.
        let err = apply(a, le, R_X86_64_64, &mut buf, 1, 0x1234, 0, 0, None, false, None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("runs past the end"), "message was {err:?}");
        assert!(buf.iter().all(|&b| b == 0), "buffer was written to");

        apply(a, le, R_X86_64_32, &mut buf, 4, 0x1234, 0, 0, None, false, None).unwrap();
        assert!(apply(a, le, R_X86_64_32, &mut buf, 5, 0x1234, 0, 0, None, false, None).is_err());
        // An offset at or past the end is rejected rather than wrapping.
        assert!(apply(a, le, R_X86_64_32, &mut buf, usize::MAX, 0, 0, 0, None, false, None).is_err());
    }

    /// A resolved value that does not fit the field being written is refused,
    /// not truncated. Writing its low bits leaves the reference pointing at an
    /// unrelated address, and nothing downstream can tell that happened.
    #[test]
    fn out_of_range_fields_are_rejected() {
        let le = Endian::Little;
        let mut buf = [0u8; 8];
        // 4GB away: the branch/load would have to reach it through a 32-bit
        // field. Reachable in practice through `--inject-base`.
        let far = 0x1_0000_0000u64;

        let err = |r_type, s, a, p, arch| {
            let mut buf = [0u8; 8];
            let e = apply(arch, le, r_type, &mut buf, 0, s, a, p, None, false, None)
                .expect_err("value does not fit its field")
                .to_string();
            assert!(buf.iter().all(|&b| b == 0), "field was written anyway");
            e
        };

        // x86-64: a call 4GB away does not fit a PC32/PLT32 displacement.
        let e = err(R_X86_64_PLT32, far, -4, 0x1000, Architecture::X86_64);
        assert!(e.contains("does not fit"), "message was {e:?}");
        assert!(e.contains("signed 32-bit"), "message was {e:?}");
        // Nor does its address fit an absolute 32-bit field, signed or not.
        err(R_X86_64_32, far, 0, 0, Architecture::X86_64);
        err(R_X86_64_32S, 0x8000_0000, 0, 0, Architecture::X86_64); // fits u32, not i32
        // In range, both still write the field.
        apply(Architecture::X86_64, le, R_X86_64_32, &mut buf, 0, 0xffff_f000, 0, 0, None, false, None)
            .unwrap();
        assert_eq!(le.read_uint(&buf, 0, 4), 0xffff_f000);

        // 32-bit x86 has no such limit: every address fits the field, and a
        // backwards reference truncating to its low 32 bits is the correct
        // modular arithmetic rather than a dropped high half.
        apply(Architecture::X86, le, R_X86_64_PC32, &mut buf, 0, 0x1000, -4, 0xf000_0000, None, false, None)
            .unwrap();
        assert_eq!(le.read_uint(&buf, 0, 4), 0x1000_0ffc);

        // aarch64's 32-bit word relocations: -2^31 <= X < 2^32.
        err(R_AARCH64_ABS32, far, 0, 0, Architecture::Arm64);
        err(R_AARCH64_PREL32, far, 0, 0, Architecture::Arm64);
        apply(Architecture::Arm64, le, R_AARCH64_ABS32, &mut buf, 0, 0xffff_f000, 0, 0, None, false, None)
            .unwrap();

        // mips `j`/`jal` encode bits 27:2 of the target, taking the rest from
        // the delay slot's own PC, so the two must share a 256MB region.
        let e = err(R_MIPS_26, 0x1000_0000, 0, 0, Architecture::Mips);
        assert!(e.contains("256MB"), "message was {e:?}");
        let e = err(R_MIPS_26, 0x0040_0102, 0, 0, Architecture::Mips);
        assert!(e.contains("4-byte aligned"), "message was {e:?}");
        // A 16-bit branch field reaches ±128KB.
        let e = err(R_MIPS_PC16, 0x0040_0000, 0, 0, Architecture::Mips);
        assert!(e.contains("out of range"), "message was {e:?}");
        // On mips64 a 32-bit field, and a HI16/LO16 pair, still span 32 bits.
        err(R_MIPS_32, far, 0, 0, Architecture::Mips64);
        err(R_MIPS_HI16, far, 0, 0, Architecture::Mips64);
    }

    #[test]
    fn aarch64_encodings() {
        let a = Architecture::Arm64;
        let le = Endian::Little;
        // BL: (S - P) >> 2 into imm26.
        assert_eq!(
            insn(a, le, R_AARCH64_CALL26, 0x9400_0000, 0x2000, 0, 0x1000, None, false, None),
            0x9400_0400
        );
        // ADRP: page(S) - page(P) into imm21 (immlo/immhi).
        assert_eq!(
            insn(a, le, R_AARCH64_ADR_PREL_PG_HI21, 0x9000_0000, 0x3000, 0, 0x1000, None, false, None),
            0xd000_0000
        );
        // ADD lo12: (S) & 0xfff into imm12.
        assert_eq!(
            insn(a, le, R_AARCH64_ADD_ABS_LO12_NC, 0x9100_0000, 0x3456, 0, 0, None, false, None),
            0x9111_5800
        );
        // GOT: ADRP + LD64 addressing a synthesized slot at 0x5040.
        assert_eq!(
            insn(a, le, R_AARCH64_ADR_GOT_PAGE, 0x9000_0000, 0, 0, 0x1000, Some(0x5040), false, None),
            0x9000_0020
        );
        assert_eq!(
            insn(a, le, R_AARCH64_LD64_GOT_LO12_NC, 0xf940_0000, 0, 0, 0, Some(0x5040), false, None),
            0xf940_2000
        );
        // ABS64 writes the full pointer.
        let mut buf = [0u8; 8];
        apply(a, le, R_AARCH64_ABS64, &mut buf, 0, 0x1234, 8, 0, None, false, None).unwrap();
        assert_eq!(le.read_uint(&buf, 0, 8), 0x123c);
    }

    #[test]
    fn arm_encodings() {
        let a = Architecture::Arm;
        let le = Endian::Little;
        // BL uses the REL scheme: addend read from the (zero) branch field.
        assert_eq!(
            insn(a, le, R_ARM_CALL, 0xeb00_0000, 0x2000, 0, 0x1000, None, true, None),
            0xeb00_0400
        );
        assert_eq!(
            insn(a, le, R_ARM_MOVW_ABS_NC, 0xe300_0000, 0x1234, 0, 0, None, true, None),
            0xe301_0234
        );
        assert_eq!(
            insn(a, le, R_ARM_MOVT_ABS, 0xe340_0000, 0x1234_5678, 0, 0, None, true, None),
            0xe341_0234
        );
    }

    #[test]
    fn mips_encodings() {
        let a = Architecture::Mips;
        let be = Endian::Big;
        // JAL: (S) >> 2 into the 26-bit region field.
        assert_eq!(
            insn(a, be, R_MIPS_26, 0x0c00_0000, 0x0040_0128, 0, 0, None, true, None),
            0x0c10_004a
        );
        // HI16 carries the sign of the paired LO16 low half (here 0).
        assert_eq!(
            insn(a, be, R_MIPS_HI16, 0x3c01_0000, 0x0041_8000, 0, 0, None, true, Some(0)),
            0x3c01_0042
        );
        assert_eq!(
            insn(a, be, R_MIPS_LO16, 0x8c22_0000, 0x0041_8000, 0, 0, None, true, None),
            0x8c22_8000
        );
        assert_eq!(
            insn(a, be, R_MIPS_PC16, 0x1000_0000, 0x0040_0100, 0, 0x0040_0000, None, true, None),
            0x1000_0040
        );
        // Little-endian MIPS encodes the same instruction word in LE byte order.
        let mut buf = [0u8; 4];
        Endian::Little.write_uint(&mut buf, 0, 4, 0x0c00_0000);
        apply(a, Endian::Little, R_MIPS_26, &mut buf, 0, 0x0040_0128, 0, 0, None, true, None).unwrap();
        assert_eq!(buf, 0x0c10_004au32.to_le_bytes());
    }

    #[test]
    fn mips_got_relocs_are_rejected() {
        let mut buf = [0u8; 4];
        let err = apply(
            Architecture::Mips,
            Endian::Big,
            object::elf::R_MIPS_GOT16,
            &mut buf,
            0,
            0x1000,
            0,
            0,
            None,
            true,
            None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("mips relocation"));
        assert!(!needs_got(Architecture::Mips, object::elf::R_MIPS_GOT16));
    }

    /// Which relocations hold an absolute address decides which references
    /// `link_objects` can resolve against a position-independent target, so the
    /// split has to follow what the encoders above actually write.
    ///
    /// The near-misses are the point: an aarch64 `*_ABS_LO12_NC` is named for
    /// an absolute address but writes only the low 12 bits of one, and it
    /// completes a PC-relative `ADRP` -- a page-aligned load base leaves both
    /// halves correct. Classifying it as absolute would refuse the form every
    /// Rust data reference compiles into on aarch64.
    #[test]
    fn absolute_and_pc_relative_relocations_are_told_apart() {
        use Architecture::{Arm, Arm64, Mips, Mips64, X86, X86_64};

        for (arch, absolute, relative) in [
            (
                X86_64,
                &[R_X86_64_64, R_X86_64_32, R_X86_64_32S][..],
                &[R_X86_64_PC64, R_X86_64_PC32, R_X86_64_PLT32, R_X86_64_GOTPCREL][..],
            ),
            (
                Arm64,
                &[R_AARCH64_ABS64, R_AARCH64_ABS32][..],
                &[
                    R_AARCH64_PREL64,
                    R_AARCH64_PREL32,
                    R_AARCH64_CALL26,
                    R_AARCH64_JUMP26,
                    R_AARCH64_ADR_PREL_PG_HI21,
                    R_AARCH64_ADD_ABS_LO12_NC,
                    R_AARCH64_LDST8_ABS_LO12_NC,
                    R_AARCH64_LDST32_ABS_LO12_NC,
                    R_AARCH64_LDST64_ABS_LO12_NC,
                    R_AARCH64_LDST128_ABS_LO12_NC,
                    R_AARCH64_ADR_GOT_PAGE,
                    R_AARCH64_LD64_GOT_LO12_NC,
                ][..],
            ),
            (
                Arm,
                &[R_ARM_ABS32, R_ARM_MOVW_ABS_NC, R_ARM_MOVT_ABS][..],
                &[R_ARM_REL32, R_ARM_CALL, R_ARM_JUMP24, R_ARM_GOT_PREL][..],
            ),
            (
                Mips64,
                &[R_MIPS_32, R_MIPS_64, R_MIPS_26, R_MIPS_HI16, R_MIPS_LO16][..],
                &[R_MIPS_PC16, R_MIPS_PC32][..],
            ),
        ] {
            for &r in absolute {
                assert!(
                    writes_absolute_address(arch, r),
                    "{arch:?} relocation {} should be absolute",
                    r.0
                );
            }
            for &r in relative {
                assert!(
                    !writes_absolute_address(arch, r),
                    "{arch:?} relocation {} should not be absolute",
                    r.0
                );
            }
        }

        // The 32-bit variants answer exactly as their 64-bit siblings do, which
        // is how [`apply`] dispatches them: same encoders, same classification.
        for r in [R_X86_64_64, R_X86_64_32, R_X86_64_32S, R_X86_64_PC32] {
            assert_eq!(
                writes_absolute_address(X86, r),
                writes_absolute_address(X86_64, r),
                "relocation {} classified differently for x86 and x86-64",
                r.0
            );
        }
        for r in [R_MIPS_32, R_MIPS_HI16, R_MIPS_LO16, R_MIPS_PC16] {
            assert_eq!(
                writes_absolute_address(Mips, r),
                writes_absolute_address(Mips64, r),
                "relocation {} classified differently for mips and mips64",
                r.0
            );
        }
    }
}
