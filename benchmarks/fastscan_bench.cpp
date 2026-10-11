// Speed of a NEON fast-scan kernel (4-bit codes, uint8 tables, 32-vector
// blocks) against the current scalar ADC scan, on the same number of vectors
// as one IVF-PQ query scans on SIFT1M (16 lists of ~1000). ARM only:
//
//   clang++ -std=c++20 -O3 -mcpu=apple-m1 benchmarks/fastscan_bench.cpp -o fastscan_bench
#include <arm_neon.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using clk = std::chrono::steady_clock;

// Current qann kernel (PQCodebook::distances_adc), 4 codes at a time.
static void scan_float(const float* table, const uint8_t* codes, size_t n, size_t m, size_t K, float* out) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const uint8_t* c = codes + i * m;
        float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
        for (size_t j = 0; j < m; ++j) {
            const float* row = table + j * K;
            d0 += row[c[j]]; d1 += row[c[m + j]]; d2 += row[c[2 * m + j]]; d3 += row[c[3 * m + j]];
        }
        out[i] = d0; out[i + 1] = d1; out[i + 2] = d2; out[i + 3] = d3;
    }
}

// Fast-scan: per block of 32 vectors and per subspace, 16 bytes hold the 32
// 4-bit codes (byte j: vector j in the low nibble, vector j+16 in the high).
// One table lookup instruction scores 16 vectors.
static void scan_fast(const uint8_t* lut, const uint8_t* blocks, size_t nblocks, size_t m, uint16_t* out) {
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    for (size_t b = 0; b < nblocks; ++b) {
        const uint8_t* blk = blocks + b * m * 16;
        uint16x8_t a0 = vdupq_n_u16(0), a1 = a0, a2 = a0, a3 = a0;
        for (size_t s = 0; s < m; ++s) {
            const uint8x16_t t = vld1q_u8(lut + s * 16);
            const uint8x16_t c = vld1q_u8(blk + s * 16);
            const uint8x16_t lo = vqtbl1q_u8(t, vandq_u8(c, mask));
            const uint8x16_t hi = vqtbl1q_u8(t, vshrq_n_u8(c, 4));
            a0 = vaddw_u8(a0, vget_low_u8(lo));
            a1 = vaddw_high_u8(a1, lo);
            a2 = vaddw_u8(a2, vget_low_u8(hi));
            a3 = vaddw_high_u8(a3, hi);
        }
        uint16_t* o = out + b * 32;
        vst1q_u16(o, a0); vst1q_u16(o + 8, a1); vst1q_u16(o + 16, a2); vst1q_u16(o + 24, a3);
    }
}

template <class F>
static double best_ns_per_vec(F&& f, size_t n) {
    double best = 1e30;
    for (int r = 0; r < 7; ++r) {
        const int reps = 200;
        auto t0 = clk::now();
        for (int i = 0; i < reps; ++i) f();
        const double ns = std::chrono::duration<double, std::nano>(clk::now() - t0).count() / reps / n;
        best = std::min(best, ns);
    }
    return best;
}

int main() {
    const size_t n = 16 * 1024;   // vectors scanned per query (nprobe 16 x ~1000)
    std::mt19937 rng(1);
    volatile float sinkf = 0;
    volatile uint16_t sinku = 0;

    std::printf("%-26s %8s %10s\n", "kernel", "ns/vec", "us/query");
    for (size_t m : {16, 32}) {   // today's PQ16x8 and PQ32x8
        std::vector<uint8_t> codes(n * m);
        for (auto& c : codes) c = rng() & 0xFF;
        std::vector<float> table(m * 256);
        for (auto& t : table) t = static_cast<float>(rng() % 1000);
        std::vector<float> out(n);
        double ns = best_ns_per_vec([&] { scan_float(table.data(), codes.data(), n, m, 256, out.data()); sinkf = sinkf + out[n - 1]; }, n);
        std::printf("scalar PQ%zux8 (current)    %8.2f %10.1f\n", m, ns, ns * n / 1000);
    }
    for (size_t m : {32, 64}) {   // fast-scan at the same 16 and 32 bytes per vector
        std::vector<uint8_t> blocks(n / 32 * m * 16);
        for (auto& c : blocks) c = rng() & 0xFF;
        std::vector<uint8_t> lut(m * 16);
        for (auto& t : lut) t = rng() & 0xFF;
        std::vector<uint16_t> out(n);
        double ns = best_ns_per_vec([&] { scan_fast(lut.data(), blocks.data(), n / 32, m, out.data()); sinku = sinku + out[n - 1]; }, n);
        std::printf("NEON fast-scan PQ%zux4      %8.2f %10.1f\n", m, ns, ns * n / 1000);
    }
}
