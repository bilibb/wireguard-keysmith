# wireguard-keysmith

This tool brute-forces WireGuard keys to find a public key that matches a given prefix. The primary purpose is to associate keys with a human-readable identifier (e.g., a person's name), similar to how [eschalot](https://github.com/ReclaimYourPrivacy/eschalot) works for Tor onion addresses.

When managing a WireGuard server with many peers, the output of the `wg` command doesn't provide any identifying information, which can be inconvenient:

```sh
peer: tWsaglw7mdtV8u1w0l+/sEfkluQFRp+JHREryLSS1EY=
  endpoint: 131.171.237.179:21247
  allowed ips: 172.18.200.5/32
  latest handshake: 10 days, 7 hours, 42 minutes, 9 seconds ago
  transfer: 1.36 MiB received, 7.85 MiB sent
```

## Prerequisites

You will need to have the Rust toolchain (including Cargo) installed.

## Usage

Run the program with your desired prefix as an argument. The tool will use all available CPU cores, and the prefix matching is case-insensitive.

You can run the application in one of the following ways:

```sh
# Using Cargo
cargo run --release -- <Prefix>

# Using just
just run <Prefix>

# On an NVIDIA GPU (needs the CUDA toolkit; CUDA_PATH defaults to /opt/cuda)
cargo run --release --features cuda -- <Prefix>
just run-cuda <Prefix>
```

## How it is fast

Instead of a full scalar multiplication per candidate, each search walks the keys `s, s+8, s+16, …` from a random
start `s` (stepping by 8 keeps the private key clamped). Around a center point `C` the candidates `C ± i·G`
(`G = 8·B`, `i = 1..H`) share the denominator of the affine addition, so one batched field inversion gives
`2H+1` public keys at roughly one multiplication and one squaring each.
Every match is re-checked with `x25519-dalek` before it is printed.

Rough throughput (Ryzen 7 1700X / RTX 2080 Ti): original approach 0.35 M keys/s, CPU 64 M keys/s, GPU 5 G keys/s.
Each case-insensitive letter in the prefix multiplies the expected work by 32, other characters by 64.
