# wireGuard-keysmith

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

# Using the Makefile
make run ARGS=<Prefix>
```
