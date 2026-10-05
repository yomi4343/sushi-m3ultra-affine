const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const uint vec0 = threadgroup_position_in_grid.x * NV;
const uint out_row = threadgroup_position_in_grid.y * 8u + sg * 4u;
const int K = int(K_size);
const int N = int(N_size);
constexpr int VPT = FAST ? 8 : 4;
constexpr int BLOCK = VPT * 32;
const int groups = K / GS;
const device uchar* ws = (const device uchar*)w + size_t(out_row) * K + lane * VPT;
const device T* sp = sc + size_t(out_row) * groups + lane / (GS / VPT);
const device T* bp = bi + size_t(out_row) * groups + lane / (GS / VPT);
const device T* xp[NV];
for (int v = 0; v < NV; ++v) xp[v] = x + size_t(min(vec0 + uint(v), uint(M - 1))) * K + lane * VPT;
float result[NV][4] = {};
int k = 0;
for (; k < (FAST ? K : K - BLOCK); k += BLOCK) {
  float xt[NV][VPT];
  float sums[NV] = {};
  for (int v = 0; v < NV; ++v) {
    for (int i = 0; i < VPT; ++i) { sums[v] += xp[v][i]; xt[v][i] = xp[v][i]; }
  }
  for (int row = 0; row < 4; ++row) {
    uchar codes[VPT];
    for (int i = 0; i < VPT; ++i) codes[i] = ws[row * K + i];
    const float scale = sp[row * groups];
    const float bias = bp[row * groups];
    for (int v = 0; v < NV; ++v) {
      float accum = 0.0f;
      for (int i = 0; i < VPT; ++i) accum += xt[v][i] * codes[i];
      result[v][row] += scale * accum + sums[v] * bias;
    }
  }
  ws += BLOCK;
  sp += BLOCK / GS;
  bp += BLOCK / GS;
  for (int v = 0; v < NV; ++v) xp[v] += BLOCK;
}
if (!FAST) {
  const int remaining = clamp(K - k - int(lane) * VPT, 0, VPT);
  if (remaining > 0) {
    float xt[NV][VPT];
    float sums[NV] = {};
    for (int v = 0; v < NV; ++v) {
      for (int i = 0; i < remaining; ++i) { sums[v] += xp[v][i]; xt[v][i] = xp[v][i]; }
    }
    for (int row = 0; row < 4; ++row) {
      uchar codes[VPT];
      for (int i = 0; i < remaining; ++i) codes[i] = ws[row * K + i];
      const float scale = sp[row * groups];
      const float bias = bp[row * groups];
      for (int v = 0; v < NV; ++v) {
        float accum = 0.0f;
        for (int i = 0; i < remaining; ++i) accum += xt[v][i] * codes[i];
        result[v][row] += scale * accum + sums[v] * bias;
      }
    }
  }
}
for (int v = 0; v < NV; ++v) {
  for (int row = 0; row < 4; ++row) {
    const float value = simd_sum(result[v][row]);
    if (lane == 0 && vec0 + uint(v) < uint(M)) y[size_t(vec0 + uint(v)) * N + out_row + row] = T(value);
  }
}