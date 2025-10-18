use base64::{Engine as _, engine::general_purpose::STANDARD};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use std::env;
use std::time::Instant;
use x25519_dalek::{PublicKey, StaticSecret};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <WireGuard Public Key Prefix>", args[0]);
        return;
    }

    let prefix = &args[1].clone();
    let prefix_len = prefix.len();
    let prefix_bytes = prefix.as_bytes().to_vec();

    let start = Instant::now();

    let result = (0..u64::MAX)
        .into_par_iter()
        .map_init(
            || ChaCha20Rng::from_entropy(),
            |rng, _| {
                let private_key: StaticSecret = StaticSecret::random_from_rng(rng);
                let public_key: PublicKey = PublicKey::from(&private_key);

                let public_key_bytes: [u8; 32] = public_key.to_bytes();

                // 32 Bytes to Base64 = 44 Bytes --> ((4 * n / 3) + 3) & ~3
                let mut b64_buffer = [0u8; 44];
                STANDARD
                    .encode_slice(public_key_bytes, &mut b64_buffer)
                    .unwrap();

                if b64_buffer[..prefix_len].eq_ignore_ascii_case(&prefix_bytes) {
                    let public_key_b64 = std::str::from_utf8(&b64_buffer).unwrap().to_string();

                    // Encode private key to Base64
                    let private_key_bytes = private_key.to_bytes();
                    let private_key_b64 = STANDARD.encode(private_key_bytes);

                    Some((private_key_b64, public_key_b64))
                } else {
                    None
                }
            },
        )
        .find_any(|result| result.is_some())
        .expect("Search failed")
        .expect("Internal error: Some was None");

    let duration = start.elapsed();

    println!("\n### Found Matching Key Pair ###");
    println!("Private Key: {}", result.0);
    println!("Public Key:  {}", result.1);
    println!("Time taken: {:?}", duration);
}
