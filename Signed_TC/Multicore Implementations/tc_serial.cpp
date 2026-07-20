#include "graph_io.h"
#include "timer.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>

struct GlobalTriCounts {
  uint64_t t0, t1, t2, t3;

  uint64_t balanced() const { return t1 + t3; }
  uint64_t unbalanced() const { return t0 + t2; }
  uint64_t total() const { return t0 + t1 + t2 + t3; }

  void print() const {
    std::printf("\nTriangle counts\n");
    std::printf("  T3 (+++ balanced)   : %llu\n", (unsigned long long)t3);
    std::printf("  T1 (+-- balanced)   : %llu\n", (unsigned long long)t1);
    std::printf("  T2 (++- unbalanced) : %llu\n", (unsigned long long)t2);
    std::printf("  T0 (--- unbalanced) : %llu\n", (unsigned long long)t0);
    std::printf("  --------------------------------\n");
    std::printf("  balanced total      : %llu\n",
                (unsigned long long)balanced());
    std::printf("  unbalanced total    : %llu\n",
                (unsigned long long)unbalanced());
    std::printf("  grand total         : %llu\n", (unsigned long long)total());
  }
};

static void sort_adj_lists(ForwardCSR &csr) {
  const uint32_t n = csr.n;

  uint32_t max_fdeg = 0;
  for (uint32_t v = 0; v < n; v++)
    if (csr.fdeg[v] > max_fdeg)
      max_fdeg = csr.fdeg[v];

  if (max_fdeg == 0)
    return;

#pragma omp parallel
  {
    uint32_t *tmp_e = new uint32_t[max_fdeg + 1];
    int8_t *tmp_s = new int8_t[max_fdeg + 1];

#pragma omp for schedule(dynamic, 64)
    for (uint32_t v = 0; v < n; v++) {
      uint32_t deg = csr.fdeg[v];
      if (deg < 2)
        continue;

      uint64_t beg = csr.offsets[v];
      std::memcpy(tmp_e, csr.edges + beg, deg * sizeof(uint32_t));
      std::memcpy(tmp_s, csr.signs + beg, deg * sizeof(int8_t));

      if (deg <= 32) {

        for (uint32_t i = 1; i < deg; i++) {
          uint32_t ke = tmp_e[i];
          int8_t ks = tmp_s[i];
          int32_t j = (int32_t)i - 1;
          while (j >= 0 && tmp_e[j] > ke) {
            tmp_e[j + 1] = tmp_e[j];
            tmp_s[j + 1] = tmp_s[j];
            j--;
          }
          tmp_e[j + 1] = ke;
          tmp_s[j + 1] = ks;
        }
        std::memcpy(csr.edges + beg, tmp_e, deg * sizeof(uint32_t));
        std::memcpy(csr.signs + beg, tmp_s, deg * sizeof(int8_t));
      } else {

        uint32_t *idx = new uint32_t[deg];
        for (uint32_t i = 0; i < deg; i++)
          idx[i] = i;
        std::sort(idx, idx + deg,
                  [&](uint32_t a, uint32_t b) { return tmp_e[a] < tmp_e[b]; });
        for (uint32_t i = 0; i < deg; i++) {
          csr.edges[beg + i] = tmp_e[idx[i]];
          csr.signs[beg + i] = tmp_s[idx[i]];
        }
        delete[] idx;
      }
    }

    delete[] tmp_e;
    delete[] tmp_s;
  }
}

static inline void classify(int8_t s_uv, int8_t s_uw, int8_t s_vw, uint64_t &t0,
                            uint64_t &t1, uint64_t &t2, uint64_t &t3) {
  int neg = ((1 - s_uv) >> 1) + ((1 - s_uw) >> 1) + ((1 - s_vw) >> 1);
  switch (neg) {
  case 0:
    t3++;
    break;
  case 1:
    t2++;
    break;
  case 2:
    t1++;
    break;
  case 3:
    t0++;
    break;
  }
}

static inline void merge_intersect(const ForwardCSR &csr, uint32_t u,
                                   uint32_t v, int8_t s_uv, uint64_t &t0,
                                   uint64_t &t1, uint64_t &t2, uint64_t &t3) {
  uint64_t i = csr.offsets[u], i_end = csr.offsets[u + 1];
  uint64_t j = csr.offsets[v], j_end = csr.offsets[v + 1];
  const uint32_t rank_u = csr.rank_[u];
  const uint32_t rank_v = csr.rank_[v];

  while (i < i_end && j < j_end) {
    uint32_t wu = csr.edges[i];
    uint32_t wv = csr.edges[j];
    if (wu == wv) {
      if (csr.rank_[wu] > rank_u && csr.rank_[wu] > rank_v)
        classify(s_uv, csr.signs[i], csr.signs[j], t0, t1, t2, t3);
      i++;
      j++;
    } else if (wu < wv) {
      i++;
    } else {
      j++;
    }
  }
}

static GlobalTriCounts count_triangles(const ForwardCSR &csr) {
  const uint32_t n = csr.n;

  uint64_t *cnt = new uint64_t[4 * (uint64_t)n]();

#pragma omp parallel for schedule(dynamic, 64)
  for (uint32_t u = 0; u < n; u++) {
    const uint64_t u_beg = csr.offsets[u];
    const uint64_t u_end = csr.offsets[u + 1];
    if (u_beg == u_end)
      continue;

    uint64_t t0 = 0, t1 = 0, t2 = 0, t3 = 0;
    for (uint64_t ei = u_beg; ei < u_end; ei++)
      merge_intersect(csr, u, csr.edges[ei], csr.signs[ei], t0, t1, t2, t3);

    cnt[4 * (uint64_t)u + 0] = t0;
    cnt[4 * (uint64_t)u + 1] = t1;
    cnt[4 * (uint64_t)u + 2] = t2;
    cnt[4 * (uint64_t)u + 3] = t3;
  }

  GlobalTriCounts g{0, 0, 0, 0};
  for (uint32_t u = 0; u < n; u++) {
    g.t0 += cnt[4 * (uint64_t)u + 0];
    g.t1 += cnt[4 * (uint64_t)u + 1];
    g.t2 += cnt[4 * (uint64_t)u + 2];
    g.t3 += cnt[4 * (uint64_t)u + 3];
  }

  delete[] cnt;
  return g;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <graph_file>\n", argv[0]);
    return 1;
  }

  Timer total_time;

  EdgeList el;
  {
    BlockTimer bt("Graph Load (I/O + parse)");
    el = load_graph(argv[1]);
  }
  std::printf("  n=%u  raw_edges=%u\n", el.n, el.m);

  ForwardCSR csr;
  {
    BlockTimer bt("Build Forward CSR");
    csr = build_forward_csr(el);
  }
  free_edge_list(el);

  {
    BlockTimer bt("Algorithm-Specific Preprocessing (Adjacency List Sort)");
    sort_adj_lists(csr);
  }
  std::printf("  forward_edges=%llu\n", (unsigned long long)csr.m);

  GlobalTriCounts g;
  {
    BlockTimer bt("Triangle Counting Execution");
    g = count_triangles(csr);
  }
  free_forward_csr(csr);

  g.print();
  total_time.print("Total execution time");
  return 0;
}