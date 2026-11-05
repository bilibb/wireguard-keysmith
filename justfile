# List recipes
default:
    @just --list

# Build the CPU version
build:
    cargo build --release

# Search for a public key starting with PREFIX on the CPU
run prefix:
    cargo run --release -- {{prefix}}

# Build the CUDA version (needs nvcc; CUDA_PATH defaults to /opt/cuda)
build-cuda:
    cargo build --release --features cuda

# Search for a public key starting with PREFIX on the GPU
run-cuda prefix:
    cargo run --release --features cuda -- {{prefix}}

# Run the tests
test:
    cargo test --release
