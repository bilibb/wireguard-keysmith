// The CPU search (search() and its field helpers) is unused when built for the GPU.
#![cfg_attr(feature = "cuda", allow(dead_code))]

use base64::{Engine as _, engine::general_purpose::STANDARD};
use fiat_crypto::curve25519_64::*;
use rand::{RngCore, rngs::OsRng};
use rayon::prelude::*;
use std::env;
use std::time::Instant;
use x25519_dalek::{PublicKey, StaticSecret, X25519_BASEPOINT_BYTES, x25519};

type Fe = fiat_25519_tight_field_element;
type Point = (Fe, Fe);

/// Half-width of a batch: per center C the keys C and C ± i·G (i = 1..=H) are checked, G = 8·B.
const H: usize = 512;
const SPAN: u64 = 2 * H as u64 + 1;
/// Centers per random start scalar (each start costs two full scalar mults).
const CENTERS: u64 = 64;
/// G = 8·B in affine Montgomery coordinates (u, v). Stepping by 8·B keeps private keys clamped.
const U_8B: [u8; 32] = [
    0x68, 0x99, 0xc9, 0x48, 0xed, 0x6f, 0x71, 0xf1, 0x1e, 0xa9, 0x74, 0xad, 0xca, 0xed, 0x78, 0x7a,
    0xa1, 0x5c, 0xb6, 0xc8, 0x4d, 0x88, 0x3b, 0xd8, 0x2d, 0x5c, 0xd6, 0xaa, 0xd7, 0xa6, 0x5f, 0x27,
];
const V_8B: [u8; 32] = [
    0xd5, 0x78, 0x8d, 0x60, 0x91, 0x92, 0xd6, 0x3a, 0x06, 0x6e, 0x3a, 0x8f, 0x98, 0x63, 0xb8, 0xb0,
    0x73, 0x4f, 0x99, 0xbf, 0x96, 0xbb, 0x30, 0x72, 0x1b, 0xc6, 0x46, 0x70, 0x45, 0x44, 0x7a, 0x1f,
];
const ONE: Fe = fiat_25519_tight_field_element([1, 0, 0, 0, 0]);
const A: Fe = fiat_25519_tight_field_element([486662, 0, 0, 0, 0]); // curve v² = u³ + A·u² + u

fn fe(bytes: &[u8; 32]) -> Fe {
    let mut out = ONE;
    fiat_25519_from_bytes(&mut out, bytes);
    out
}

fn mul(a: &Fe, b: &Fe) -> Fe {
    let (mut la, mut lb, mut out) = (
        fiat_25519_loose_field_element([0; 5]),
        fiat_25519_loose_field_element([0; 5]),
        ONE,
    );
    fiat_25519_relax(&mut la, a);
    fiat_25519_relax(&mut lb, b);
    fiat_25519_carry_mul(&mut out, &la, &lb);
    out
}

fn square(a: &Fe) -> Fe {
    let (mut la, mut out) = (fiat_25519_loose_field_element([0; 5]), ONE);
    fiat_25519_relax(&mut la, a);
    fiat_25519_carry_square(&mut out, &la);
    out
}

fn add(a: &Fe, b: &Fe) -> Fe {
    let (mut l, mut out) = (fiat_25519_loose_field_element([0; 5]), ONE);
    fiat_25519_add(&mut l, a, b);
    fiat_25519_carry(&mut out, &l);
    out
}

fn sub(a: &Fe, b: &Fe) -> Fe {
    let (mut l, mut out) = (fiat_25519_loose_field_element([0; 5]), ONE);
    fiat_25519_sub(&mut l, a, b);
    fiat_25519_carry(&mut out, &l);
    out
}

fn invert(a: &Fe) -> Fe {
    // ponytail: plain square-and-multiply a^(p-2), ~2x the cost of an addition chain; amortized over SPAN keys.
    // p-2 = 2^255-21: bits 0..=254 all set except bits 2 and 4.
    let mut r = ONE;
    for bit in (0..255).rev() {
        r = square(&r);
        if bit != 2 && bit != 4 {
            r = mul(&r, a);
        }
    }
    r
}

/// Affine P + Q for P != ±Q, given λ = (yQ - yP)/(xQ - xP).
fn add_with_slope(p: &Point, q: &Point, l: &Fe) -> Point {
    let x = sub(&square(l), &add(&add(&A, &p.0), &q.0));
    (x, sub(&mul(l, &sub(&p.0, &x)), &p.1))
}

fn affine_add(p: &Point, q: &Point) -> Point {
    add_with_slope(p, q, &mul(&sub(&q.1, &p.1), &invert(&sub(&q.0, &p.0))))
}

/// T[i] = (i+1)·G for i < h, T[h] = (2h+1)·G (the jump to the next center).
fn table(h: usize) -> Vec<Point> {
    let g = (fe(&U_8B), fe(&V_8B));
    // 2G: λ = (3x² + 2Ax + 1) / 2y
    let three_x2 = mul(
        &square(&g.0),
        &fiat_25519_tight_field_element([3, 0, 0, 0, 0]),
    );
    let l = mul(
        &add(&add(&three_x2, &mul(&add(&A, &A), &g.0)), &ONE),
        &invert(&add(&g.1, &g.1)),
    );
    let mut t = vec![g, add_with_slope(&g, &g, &l)];
    while t.len() <= h {
        t.push(affine_add(t.last().unwrap(), &g));
    }
    t[h] = affine_add(&t[h - 1], &t[h]);
    t
}

/// Private scalar of key `i` in the sequence starting at `s`: s + 8·i (little-endian).
fn nth_key(mut s: [u8; 32], i: u64) -> [u8; 32] {
    let mut carry = i as u128 * 8;
    for b in s.iter_mut() {
        carry += *b as u128;
        *b = carry as u8;
        carry >>= 8;
    }
    s
}

/// Affine point of key `i` (its private scalar is nth_key(s, i)). x25519 gives only u, so v is recovered
/// from u(P) and u(P + G) (Okeya–Sakurai): v = ((u·uG + 1)(u + uG + 2A) - 2A - (u - uG)²·u(P+G)) / 2vG.
fn point(s: [u8; 32], i: u64) -> Point {
    let u = fe(&x25519(nth_key(s, i), X25519_BASEPOINT_BYTES));
    let u1 = fe(&x25519(nth_key(s, i + 1), X25519_BASEPOINT_BYTES));
    let (ug, vg) = (fe(&U_8B), fe(&V_8B));
    let a2 = add(&A, &A);
    let t = mul(&add(&mul(&u, &ug), &ONE), &add(&add(&u, &ug), &a2));
    let num = sub(&sub(&t, &a2), &mul(&square(&sub(&u, &ug)), &u1));
    (u, mul(&num, &invert(&add(&vg, &vg))))
}

fn to_bytes(x: &Fe) -> [u8; 32] {
    let mut out = [0u8; 32];
    fiat_25519_to_bytes(&mut out, x);
    out
}

fn matches(public: &[u8; 32], prefix: &[u8]) -> bool {
    // Cheap reject on the first char (top 6 bits of byte 0) before base64-encoding everything.
    const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    if !B64[(public[0] >> 2) as usize].eq_ignore_ascii_case(&prefix[0]) {
        return false;
    }
    let mut b64 = [0u8; 44];
    STANDARD.encode_slice(public, &mut b64).unwrap();
    b64[..prefix.len()].eq_ignore_ascii_case(prefix)
}

/// Checks keys s, s+8, s+16, … (CENTERS·SPAN of them) without a scalar mult per key. Around each
/// center C: C ± i·G share the denominator xT - xC, so one batch inversion (Montgomery's trick)
/// yields 2H affine additions plus the jump C += (2H+1)·G; each costs ~1 mul + 1 square.
fn search(s: [u8; 32], prefix: &[u8], t: &[Point]) -> Option<[u8; 32]> {
    let mut c = point(s, H as u64);
    let mut acc = vec![ONE; H + 1];
    let check = |x: &Fe, i: u64| matches(&to_bytes(x), prefix).then(|| nth_key(s, i));

    for k in 0..CENTERS {
        let (xc, yc) = c;
        let center = k * SPAN + H as u64;
        if let Some(key) = check(&xc, center) {
            return Some(key);
        }
        let mut prod = ONE;
        for (a, ti) in acc.iter_mut().zip(t) {
            prod = mul(&prod, &sub(&ti.0, &xc));
            *a = prod;
        }
        let mut inv = invert(&prod);
        // Jump first: inv currently is 1/prod, peel off T[H].
        inv = {
            let d_inv = mul(&inv, &acc[H - 1]);
            c = add_with_slope(&(xc, yc), &t[H], &mul(&sub(&t[H].1, &yc), &d_inv));
            mul(&inv, &sub(&t[H].0, &xc))
        };
        let a_xc = add(&A, &xc);
        for i in (0..H).rev() {
            let d_inv = if i == 0 { inv } else { mul(&inv, &acc[i - 1]) };
            inv = mul(&inv, &sub(&t[i].0, &xc));
            let s_x = add(&a_xc, &t[i].0);
            // C + T[i]: λ = (yT - yC)/d; C - T[i]: λ = -(yT + yC)/d (sign vanishes in λ²)
            let x_plus = sub(&square(&mul(&sub(&t[i].1, &yc), &d_inv)), &s_x);
            let x_minus = sub(&square(&mul(&add(&t[i].1, &yc), &d_inv)), &s_x);
            let off = i as u64 + 1;
            if let Some(key) =
                check(&x_plus, center + off).or_else(|| check(&x_minus, center - off))
            {
                return Some(key);
            }
        }
    }
    None
}

fn random_scalar() -> [u8; 32] {
    let mut s = [0u8; 32];
    OsRng.fill_bytes(&mut s);
    s[0] &= 248;
    s[31] = (s[31] & 127) | 64;
    s
}

#[cfg(not(feature = "cuda"))]
fn find(prefix: &[u8]) -> [u8; 32] {
    let t = table(H);
    (0..u64::MAX)
        .into_par_iter()
        .find_map_any(|_| search(random_scalar(), prefix, &t))
        .expect("Search failed")
}

/// Same algorithm on the GPU (src/gpu.cu): one key sequence per CUDA thread.
#[cfg(feature = "cuda")]
fn find(prefix: &[u8]) -> [u8; 32] {
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Hit {
        thread: u32,
        step: u64,
    }
    unsafe extern "C" {
        fn ks_half_width() -> u32;
        fn ks_suggested_threads() -> u32;
        fn ks_init(
            centers: *const u8,
            n: u32,
            table: *const u8,
            prefix: *const u8,
            len: u32,
        ) -> i32;
        fn ks_run(centers: u32, hits: *mut Hit, max_hits: i32) -> i32;
    }
    const GPU_CENTERS: u32 = 32; // centers per thread per kernel launch
    let bytes = |p: &Point| [to_bytes(&p.0), to_bytes(&p.1)].concat();

    let (h, n) = unsafe { (ks_half_width() as usize, ks_suggested_threads()) };
    assert!(n > 0, "No CUDA device found");
    let table: Vec<u8> = table(h).iter().flat_map(bytes).collect();
    // First center of every GPU thread: two scalar mults each, once, on the CPU.
    let seeds: Vec<[u8; 32]> = (0..n).map(|_| random_scalar()).collect();
    let centers: Vec<u8> = seeds
        .par_iter()
        .flat_map_iter(|&s| bytes(&point(s, h as u64)))
        .collect();
    let lower = prefix.to_ascii_lowercase();
    let len = lower.len() as u32;
    let init = unsafe { ks_init(centers.as_ptr(), n, table.as_ptr(), lower.as_ptr(), len) };
    assert_eq!(init, 0, "CUDA init failed");

    let mut hit = Hit { thread: 0, step: 0 };
    loop {
        match unsafe { ks_run(GPU_CENTERS, &mut hit, 1) } {
            0 => {}
            1 => return nth_key(seeds[hit.thread as usize], hit.step),
            _ => panic!("CUDA error"),
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <WireGuard Public Key Prefix>", args[0]);
        return;
    }

    let prefix = args[1].as_bytes();
    if prefix.is_empty()
        || prefix.len() > 43
        || !prefix
            .iter()
            .all(|c| c.is_ascii_alphanumeric() || b"+/".contains(c))
    {
        eprintln!("Prefix must be 1 to 43 base64 characters (A-Z a-z 0-9 + /)");
        std::process::exit(1);
    }

    let start = Instant::now();

    let private_key = find(prefix);

    // Independent check with x25519-dalek, so a math bug can never print a wrong pair.
    let public_key = PublicKey::from(&StaticSecret::from(private_key)).to_bytes();
    assert!(
        matches(&public_key, prefix),
        "Internal error: key does not match prefix"
    );

    let duration = start.elapsed();

    println!("\n### Found Matching Key Pair ###");
    println!("Private Key: {}", STANDARD.encode(private_key));
    println!("Public Key:  {}", STANDARD.encode(public_key));
    println!("Time taken: {:?}", duration);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_matches_x25519() {
        let s = nth_key([0x40; 32], 0); // arbitrary clamped scalar (low 3 bits 0, bit 254 set)
        let t = table(H);
        let h = H as u64;
        for i in [
            0,
            1,
            h - 1,
            h,
            h + 1,
            2 * h,
            SPAN,
            SPAN + h,
            5 * SPAN + 3,
            CENTERS * SPAN - 1,
        ] {
            let key = nth_key(s, i);
            let b64 = STANDARD.encode(x25519(key, X25519_BASEPOINT_BYTES));
            assert_eq!(search(s, &b64.as_bytes()[..43], &t), Some(key), "key {i}");
        }
    }
}
