// Only with `--features cuda`: compile src/gpu.cu with nvcc and link it plus the CUDA runtime.
fn main() {
    println!("cargo:rerun-if-changed=src/gpu.cu");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let out = std::env::var("OUT_DIR").unwrap();
    let cuda = std::env::var("CUDA_PATH").unwrap_or_else(|_| "/opt/cuda".into());
    let status = std::process::Command::new(format!("{cuda}/bin/nvcc"))
        .args(["-O3", "-arch=native", "-lib", "src/gpu.cu", "-o"])
        .arg(format!("{out}/libkeysmith_gpu.a"))
        .status()
        .expect("nvcc not found, set CUDA_PATH");
    assert!(status.success(), "nvcc failed");
    println!("cargo:rustc-link-search=native={out}");
    println!("cargo:rustc-link-search=native={cuda}/lib64");
    println!("cargo:rustc-link-lib=static=keysmith_gpu");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=stdc++");
}
