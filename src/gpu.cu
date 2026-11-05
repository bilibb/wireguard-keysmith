// GPU version of the CPU search in main.rs. Every thread owns a sequence of keys s, s+8, s+16, ...
// walked center by center: for a center C (affine) the points C ± i·G (G = 8·B, i = 1..H) share the
// denominator xT - xC, so one batch inversion per center yields 2H+1 keys plus the jump C += (2H+1)·G.
// Field elements: 8 x 32-bit limbs, kept in [0, 2^256), reduced mod p = 2^255-19.
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#ifndef H
#define H 512 // half-width of a batch; acc[] lives in local memory ((H+1)*32 bytes per thread)
#endif
#define SPAN (2 * H + 1)

typedef struct { uint32_t v[8]; } fe;
typedef struct { uint32_t thread; uint64_t step; } ks_hit;

__constant__ fe C_TX[H + 1], C_TY[H + 1]; // T[i] = (i+1)·G for i < H, T[H] = (2H+1)·G
__constant__ uint8_t C_ACCEPT[2][43];     // per prefix position: accepted 6-bit base64 values (upper/lower case)
__constant__ uint32_t C_LEN;

__device__ __forceinline__ fe fe_add(const fe &a, const fe &b) {
    fe r; uint64_t c = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) { c += (uint64_t)a.v[i] + b.v[i]; r.v[i] = (uint32_t)c; c >>= 32; }
    c *= 38; // 2^256 = 38 (mod p)
    #pragma unroll
    for (int i = 0; i < 8; i++) { c += r.v[i]; r.v[i] = (uint32_t)c; c >>= 32; }
    r.v[0] += (uint32_t)c * 38; // wrapped again => r < 38, cannot overflow
    return r;
}

__device__ __forceinline__ fe fe_sub(const fe &a, const fe &b) {
    fe r; int64_t c = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) { c += (int64_t)a.v[i] - b.v[i]; r.v[i] = (uint32_t)c; c >>= 32; }
    c *= 38; // borrowed 2^256 => subtract 38
    #pragma unroll
    for (int i = 0; i < 8; i++) { c += r.v[i]; r.v[i] = (uint32_t)c; c >>= 32; }
    r.v[0] -= (uint32_t)(-c) * 38; // borrowed again => r >= 2^256-38, cannot underflow
    return r;
}

__device__ __forceinline__ fe fe_reduce(const uint32_t t[16]) {
    fe r; uint64_t c = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) { c += (uint64_t)t[i + 8] * 38 + t[i]; r.v[i] = (uint32_t)c; c >>= 32; }
    c *= 38;
    #pragma unroll
    for (int i = 0; i < 8; i++) { c += r.v[i]; r.v[i] = (uint32_t)c; c >>= 32; }
    r.v[0] += (uint32_t)c * 38;
    return r;
}

__device__ __forceinline__ fe fe_mul(const fe &a, const fe &b) {
    uint32_t t[16] = {0};
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t c = 0;
        #pragma unroll
        for (int j = 0; j < 8; j++) { c += (uint64_t)a.v[i] * b.v[j] + t[i + j]; t[i + j] = (uint32_t)c; c >>= 32; }
        t[i + 8] = (uint32_t)c;
    }
    return fe_reduce(t);
}

__device__ __forceinline__ fe fe_sq(const fe &a) {
    // Off-diagonal products once, doubled, plus the diagonal: 36 instead of 64 multiplies.
    uint32_t t[16] = {0};
    #pragma unroll
    for (int i = 0; i < 7; i++) {
        uint64_t c = 0;
        #pragma unroll
        for (int j = i + 1; j < 8; j++) { c += (uint64_t)a.v[i] * a.v[j] + t[i + j]; t[i + j] = (uint32_t)c; c >>= 32; }
        t[i + 8] = (uint32_t)c;
    }
    uint32_t hi = 0;
    #pragma unroll
    for (int i = 0; i < 16; i++) { uint32_t n = t[i] >> 31; t[i] = (t[i] << 1) | hi; hi = n; }
    uint64_t c = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t d = (uint64_t)a.v[i] * a.v[i];
        c += (uint64_t)t[2 * i] + (uint32_t)d; t[2 * i] = (uint32_t)c; c >>= 32;
        c += (uint64_t)t[2 * i + 1] + (d >> 32); t[2 * i + 1] = (uint32_t)c; c >>= 32;
    }
    return fe_reduce(t);
}

__device__ __forceinline__ fe fe_sqn(fe a, int n) { for (int i = 0; i < n; i++) a = fe_sq(a); return a; }

__device__ fe fe_inv(const fe &z) { // z^(p-2), ref10 addition chain: 254 squarings, 11 multiplies
    fe t0 = fe_sq(z), t1 = fe_mul(z, fe_sqn(t0, 2)), t2, t3;
    t0 = fe_mul(t0, t1);                   // z^11
    t1 = fe_mul(t1, fe_sq(t0));            // z^(2^5-1)
    t1 = fe_mul(fe_sqn(t1, 5), t1);        // 2^10-1
    t2 = fe_mul(fe_sqn(t1, 10), t1);       // 2^20-1
    t2 = fe_mul(fe_sqn(t2, 20), t2);       // 2^40-1
    t1 = fe_mul(fe_sqn(t2, 10), t1);       // 2^50-1
    t2 = fe_mul(fe_sqn(t1, 50), t1);       // 2^100-1
    t3 = fe_mul(fe_sqn(t2, 100), t2);      // 2^200-1
    t1 = fe_mul(fe_sqn(t3, 50), t1);       // 2^250-1
    return fe_mul(fe_sqn(t1, 5), t0);      // 2^255-21
}

__device__ __forceinline__ void fe_canon(fe &r) { // [0, 2^256) -> [0, p)
    uint64_t c = (uint64_t)(r.v[7] >> 31) * 19; r.v[7] &= 0x7fffffff;
    #pragma unroll
    for (int i = 0; i < 8; i++) { c += r.v[i]; r.v[i] = (uint32_t)c; c >>= 32; }
    fe t; c = 19; // r < 2^255+19 now; r >= p  <=>  r+19 >= 2^255
    #pragma unroll
    for (int i = 0; i < 8; i++) { c += r.v[i]; t.v[i] = (uint32_t)c; c >>= 32; }
    if (t.v[7] >> 31) { t.v[7] &= 0x7fffffff; r = t; }
}

__device__ __forceinline__ uint32_t byte_at(const fe &u, uint32_t i) { return i < 32 ? (u.v[i >> 2] >> ((i & 3) * 8)) & 0xff : 0; }

__device__ __forceinline__ bool matches(const fe &u) {
    for (uint32_t k = 0; k < C_LEN; k++) {
        uint32_t bit = 6 * k, i = bit >> 3, w = (byte_at(u, i) << 8) | byte_at(u, i + 1);
        uint32_t v = (w >> (10 - (bit & 7))) & 63;
        if (v != C_ACCEPT[0][k] && v != C_ACCEPT[1][k]) return false;
    }
    return true;
}

__device__ __forceinline__ void check(fe u, uint32_t t, uint64_t j, ks_hit *hits, uint32_t *nhits, uint32_t max_hits) {
    fe_canon(u);
    if (matches(u)) {
        uint32_t h = atomicAdd(nhits, 1);
        if (h < max_hits) hits[h] = ks_hit{t, j};
    }
}

__global__ void __launch_bounds__(256, 1) ks_kernel(fe *cx, fe *cy, uint32_t n, uint32_t centers, uint64_t base,
                                                 ks_hit *hits, uint32_t *nhits, uint32_t max_hits) {
    uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const fe A = {{486662}};
    fe xc = cx[t], yc = cy[t];
    fe acc[H + 1]; // acc[i] = d_0 * ... * d_i, d_i = xT_i - xC
    for (uint32_t k = 0; k < centers; k++) {
        uint64_t jc = H + (uint64_t)SPAN * (base + k); // key index of the center
        check(xc, t, jc, hits, nhits, max_hits);
        fe prod = {{1}};
        for (int i = 0; i <= H; i++) { prod = fe_mul(prod, fe_sub(C_TX[i], xc)); acc[i] = prod; }
        fe inv = fe_inv(prod);
        // Jump (T[H]) first: C' = C + (2H+1)G
        fe l = fe_mul(fe_sub(C_TY[H], yc), fe_mul(inv, acc[H - 1]));
        inv = fe_mul(inv, fe_sub(C_TX[H], xc));
        fe a_xc = fe_add(A, xc);
        fe nx = fe_sub(fe_sq(l), fe_add(a_xc, C_TX[H]));
        fe ny = fe_sub(fe_mul(l, fe_sub(xc, nx)), yc);
        for (int i = H - 1; i >= 0; i--) {
            fe d_inv = i ? fe_mul(inv, acc[i - 1]) : inv;
            inv = fe_mul(inv, fe_sub(C_TX[i], xc));
            fe s = fe_add(a_xc, C_TX[i]);
            // C + T: λ = (yT - yC)/d; C - T: λ = -(yT + yC)/d (sign vanishes in λ²)
            check(fe_sub(fe_sq(fe_mul(fe_sub(C_TY[i], yc), d_inv)), s), t, jc + i + 1, hits, nhits, max_hits);
            check(fe_sub(fe_sq(fe_mul(fe_add(C_TY[i], yc), d_inv)), s), t, jc - i - 1, hits, nhits, max_hits);
        }
        xc = nx; yc = ny;
    }
    cx[t] = xc; cy[t] = yc;
}

static fe *d_cx, *d_cy; static ks_hit *d_hits; static uint32_t *d_nhits;
static uint32_t g_n; static uint64_t g_total;
#define MAX_HITS 64
#define TRY(x) do { if ((x) != cudaSuccess) return -1; } while (0)

static fe fe_bytes(const uint8_t *b) { fe r; for (int i = 0; i < 8; i++) r.v[i] = b[4*i] | b[4*i+1] << 8 | b[4*i+2] << 16 | (uint32_t)b[4*i+3] << 24; return r; }

extern "C" uint32_t ks_suggested_threads(void) {
    int dev = 0, sms = 0, blocks = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, ks_kernel, 256, 0);
    return (uint32_t)(sms * (blocks > 0 ? blocks : 1) * 256);
}

extern "C" uint32_t ks_half_width(void) { return H; }

// centers: n * 64 bytes, per thread affine (x, y) of its first center, canonical little-endian.
// table: (H+1) * 64 bytes, (x, y) of T[0..=H]. prefix: lowercase base64 chars, 1 <= len <= 43.
extern "C" int ks_init(const uint8_t *centers, uint32_t n, const uint8_t *table, const char *prefix, uint32_t len) {
    static const char B64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    uint8_t accept[2][43] = {{0}};
    if (len < 1 || len > 43) return -1;
    for (uint32_t k = 0; k < len; k++) {
        char lo = prefix[k], up = lo >= 'a' && lo <= 'z' ? lo - 32 : lo;
        for (int v = 0; v < 64; v++) {
            if (B64[v] == lo) accept[0][k] = v;
            if (B64[v] == up) accept[1][k] = v;
        }
    }
    fe tx[H + 1], ty[H + 1];
    for (int i = 0; i <= H; i++) { tx[i] = fe_bytes(table + 64 * i); ty[i] = fe_bytes(table + 64 * i + 32); }
    fe *hx = (fe *)malloc(sizeof(fe) * n), *hy = (fe *)malloc(sizeof(fe) * n);
    if (!hx || !hy) { free(hx); free(hy); return -1; }
    for (uint32_t t = 0; t < n; t++) { hx[t] = fe_bytes(centers + 64 * t); hy[t] = fe_bytes(centers + 64 * t + 32); }
    g_n = n; g_total = 0;
    int err = cudaMalloc(&d_cx, sizeof(fe) * n) != cudaSuccess || cudaMalloc(&d_cy, sizeof(fe) * n) != cudaSuccess
           || cudaMemcpy(d_cx, hx, sizeof(fe) * n, cudaMemcpyHostToDevice) != cudaSuccess
           || cudaMemcpy(d_cy, hy, sizeof(fe) * n, cudaMemcpyHostToDevice) != cudaSuccess;
    free(hx); free(hy);
    if (err) return -1;
    TRY(cudaMalloc(&d_hits, sizeof(ks_hit) * MAX_HITS));
    TRY(cudaMalloc(&d_nhits, sizeof(uint32_t)));
    TRY(cudaMemcpyToSymbol(C_TX, tx, sizeof(tx)));
    TRY(cudaMemcpyToSymbol(C_TY, ty, sizeof(ty)));
    TRY(cudaMemcpyToSymbol(C_ACCEPT, accept, sizeof(accept)));
    TRY(cudaMemcpyToSymbol(C_LEN, &len, sizeof(len)));
    return 0;
}

// Advance every thread by `centers` centers (SPAN keys each). Hit step = key index j: key = s + 8j.
extern "C" int ks_run(uint32_t centers, ks_hit *hits, int max_hits) {
    if (max_hits < 0 || max_hits > MAX_HITS) max_hits = max_hits < 0 ? 0 : MAX_HITS;
    uint32_t found = 0;
    TRY(cudaMemset(d_nhits, 0, sizeof(uint32_t)));
    ks_kernel<<<(g_n + 255) / 256, 256>>>(d_cx, d_cy, g_n, centers, g_total, d_hits, d_nhits, max_hits);
    TRY(cudaGetLastError());
    TRY(cudaMemcpy(&found, d_nhits, sizeof(found), cudaMemcpyDeviceToHost));
    g_total += centers;
    if (found > (uint32_t)max_hits) found = max_hits;
    TRY(cudaMemcpy(hits, d_hits, sizeof(ks_hit) * found, cudaMemcpyDeviceToHost));
    return (int)found;
}
