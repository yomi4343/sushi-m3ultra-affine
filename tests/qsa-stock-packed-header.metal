#define QSA_PACKED 1
template <typename T, int BITS, int GS>
inline uint4 sushi_qsa_unpack8(const device uint32_t* wq, const device T* sc, const device T* bi, int c8) {
  constexpr int VPW = 32 / BITS;
  constexpr int NW = 8 / VPW;
  constexpr uint MASKB = (1u << BITS) - 1u;
  const int g = (c8 * 8) / GS;
  const float sj = float(sc[g]);
  const float bj = float(bi[g]);
  uint4 out = uint4(0);
  thread T* e = (thread T*)&out;
  for (int wj = 0; wj < NW; ++wj) {
    const uint w = wq[c8 * NW + wj];
    for (int u = 0; u < VPW; ++u) e[wj * VPW + u] = T(float((w >> (u * BITS)) & MASKB) * sj + bj);
  }
  return out;
}
#include <metal_simdgroup_matrix>

// Fragment layout mirrors MLX steel BaseMMAFrag<float,8,8>: each thread
// of a simdgroup holds 2 adjacent elements of an 8x8 tile; the hardware
// mma runs on simdgroup_float8x8 built from those elements. The 4 threads
// holding one row differ in lane bits 0 and 3 (see sushi_coord).
inline short2 sushi_coord(ushort lane) {
  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;
  return short2(fn, fm);
}

inline void sushi_mma(thread float2 &d, float2 a, float2 b) {
  metal::simdgroup_float8x8 D, A, B, C;
  A.thread_elements()[0] = a.x;
  A.thread_elements()[1] = a.y;
  B.thread_elements()[0] = b.x;
  B.thread_elements()[1] = b.y;
  C.thread_elements()[0] = d.x;
  C.thread_elements()[1] = d.y;
  simdgroup_multiply_accumulate(D, A, B, C);
  d.x = D.thread_elements()[0];
  d.y = D.thread_elements()[1];
}

inline float sushi_row_max(float2 v) {
  float t = metal::max(v.x, v.y);
  t = metal::max(t, metal::simd_shuffle_xor(t, ushort(1)));
  t = metal::max(t, metal::simd_shuffle_xor(t, ushort(8)));
  return t;
}

inline float sushi_row_sum(float2 v) {
  float t = v.x + v.y;
  t += metal::simd_shuffle_xor(t, ushort(1));
  t += metal::simd_shuffle_xor(t, ushort(8));
  return t;
}
inline int sushi_qsa_pos(const device int* blk, int vi, int sel_len, int tail_start, int ratio) {
  const int b = vi / ratio;
  return (vi < sel_len) ? (blk[b] * ratio + (vi - b * ratio)) : (tail_start + (vi - sel_len));
}
