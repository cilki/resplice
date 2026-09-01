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

/// Apply one relocation, writing the resolved field into `buf` at `off`.
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
    match arch {
        Architecture::X86 | Architecture::X86_64 => {
            apply_x86_64(endian, r_type, buf, off, s, a, p, got_va)
        }
        Architecture::Arm64 => apply_aarch64(endian, r_type, buf, off, s, a, p, got_va),
        Architecture::Arm => apply_arm(endian, r_type, buf, off, s, p, got_va),
        Architecture::Mips | Architecture::Mips64 => {
            apply_mips(endian, r_type, buf, off, s, a, p, has_implicit, pair_lo)
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
) -> Result<()> {
    let pc32 = |v: i64| (v as i32) as u32 as u64;
    match r_type {
        R_X86_64_64 => endian.write_uint(buf, off, 8, s.wrapping_add(a as u64)),
        R_X86_64_PC64 => endian.write_uint(buf, off, 8, (s as i64 + a - p as i64) as u64),
        R_X86_64_PC32 | R_X86_64_PLT32 => {
            endian.write_uint(buf, off, 4, pc32(s as i64 + a - p as i64))
        }
        R_X86_64_32 => endian.write_uint(buf, off, 4, s.wrapping_add(a as u64) as u32 as u64),
        R_X86_64_32S => endian.write_uint(buf, off, 4, s.wrapping_add(a as u64) as i32 as u32 as u64),
        R_X86_64_GOTPCREL | R_X86_64_GOTPCRELX | R_X86_64_REX_GOTPCRELX => {
            let g = got_va.ok_or_else(|| anyhow!("missing GOT slot for GOTPCREL"))?;
            endian.write_uint(buf, off, 4, pc32(g as i64 + a - p as i64));
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
    match r_type {
        R_AARCH64_ABS64 => endian.write_uint(buf, off, 8, s.wrapping_add(a as u64)),
        R_AARCH64_ABS32 => endian.write_uint(buf, off, 4, s.wrapping_add(a as u64) as u32 as u64),
        R_AARCH64_PREL64 => endian.write_uint(buf, off, 8, (s as i64 + a - p as i64) as u64),
        R_AARCH64_PREL32 => {
            endian.write_uint(buf, off, 4, (s as i64 + a - p as i64) as i32 as u32 as u64)
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
) -> Result<()> {
    let insn = endian.read_uint(buf, off, 4) as u32;
    match r_type {
        R_MIPS_32 => {
            let a = if has_implicit { insn as i64 } else { a };
            endian.write_uint(buf, off, 4, (s as i64 + a) as u32 as u64);
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
            let value = ((ahi + alo + s as i64 + 0x8000) >> 16) & 0xffff;
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
            let field = ((x >> 2) as u32) & 0xffff;
            patch_insn(endian, buf, off, 0xffff, field);
        }
        R_MIPS_PC32 => {
            let a = if has_implicit { insn as i64 } else { a };
            endian.write_uint(buf, off, 4, (s as i64 + a - p as i64) as u32 as u64);
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

/// Map an `object` crate architecture/endianness/class to ours, for verifying
/// that an rlib member matches the target it is being spliced into.
pub fn describe(arch: Architecture, endian: Endian, is_64: bool) -> String {
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
    let c = if is_64 { "64-bit" } else { "32-bit" };
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
}
