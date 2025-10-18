.PHONY: build run

build:
	cargo build --release

run:
	cargo run --release -- $(ARGS)
