#include "graph_io.h"
#include "timer.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>

using namespace std;

struct SplitForwardCSR {
  uint32_t n;
  uint64_t *pos_offsets;
  uint32_t *pos_edges;
  uint64_t *neg_offsets;
  uint32_t *neg_edges;
};

static SplitForwardCSR build_split_forward_csr(const ForwardCSR &csr) {
  const uint32_t n = csr.n;
  const uint64_t *offsets = csr.offsets;
  const uint32_t *edges = csr.edges;
  const int8_t *signs = csr.signs;
  const uint32_t *fdeg = csr.fdeg;

  uint32_t max_deg = 0;
#pragma omp parallel for reduction(max : max_deg) schedule(static)
  for (uint32_t u = 0; u < n; ++u)
    if (fdeg[u] > max_deg)
      max_deg = fdeg[u];

  auto *pos_deg = new uint32_t[n]();
  auto *neg_deg = new uint32_t[n]();

#pragma omp parallel
  {
    uint32_t *tmp_e = new uint32_t[max_deg + 1];
    int8_t *tmp_s = new int8_t[max_deg + 1];
    uint32_t *idx = new uint32_t[max_deg + 1];

#pragma omp for schedule(dynamic, 64)
    for (uint32_t u = 0; u < n; ++u) {
      const uint32_t deg = fdeg[u];
      if (deg == 0)
        continue;

      const uint64_t beg = offsets[u];

      if (deg == 1) {
        (signs[beg] > 0) ? ++pos_deg[u] : ++neg_deg[u];
        continue;
      }

      memcpy(tmp_e, edges + beg, deg * sizeof(uint32_t));
      memcpy(tmp_s, signs + beg, deg * sizeof(int8_t));

      if (deg <= 32) {
        for (uint32_t i = 1; i < deg; ++i) {
          uint32_t ke = tmp_e[i];
          int8_t ks = tmp_s[i];
          int32_t j = (int32_t)i - 1;
          while (j >= 0 && tmp_e[j] > ke) {
            tmp_e[j + 1] = tmp_e[j];
            tmp_s[j + 1] = tmp_s[j];
            --j;
          }
          tmp_e[j + 1] = ke;
          tmp_s[j + 1] = ks;
        }
      } else {
        for (uint32_t i = 0; i < deg; ++i)
          idx[i] = i;
        std::sort(idx, idx + deg,
                  [&](uint32_t a, uint32_t b) { return tmp_e[a] < tmp_e[b]; });

        uint32_t *te2 = new uint32_t[deg];
        int8_t *ts2 = new int8_t[deg];
        for (uint32_t i = 0; i < deg; ++i) {
          te2[i] = tmp_e[idx[i]];
          ts2[i] = tmp_s[idx[i]];
        }
        memcpy(tmp_e, te2, deg * sizeof(uint32_t));
        memcpy(tmp_s, ts2, deg * sizeof(int8_t));
        delete[] te2;
        delete[] ts2;
      }

      memcpy((uint32_t *)(edges + beg), tmp_e, deg * sizeof(uint32_t));
      memcpy((int8_t *)(signs + beg), tmp_s, deg * sizeof(int8_t));

      uint32_t pc = 0, nc = 0;
      for (uint32_t i = 0; i < deg; ++i) {
        if (tmp_s[i] > 0)
          ++pc;
        else
          ++nc;
      }
      pos_deg[u] = pc;
      neg_deg[u] = nc;
    }

    delete[] tmp_e;
    delete[] tmp_s;
    delete[] idx;
  }

  auto *pos_offsets = new uint64_t[n + 1];
  auto *neg_offsets = new uint64_t[n + 1];
  pos_offsets[0] = neg_offsets[0] = 0;
  for (uint32_t u = 0; u < n; ++u) {
    pos_offsets[u + 1] = pos_offsets[u] + pos_deg[u];
    neg_offsets[u + 1] = neg_offsets[u] + neg_deg[u];
  }
  delete[] pos_deg;
  delete[] neg_deg;

  auto *pos_edges_out = new uint32_t[pos_offsets[n]];
  auto *neg_edges_out = new uint32_t[neg_offsets[n]];

#pragma omp parallel for schedule(static)
  for (uint32_t u = 0; u < n; ++u) {
    const uint32_t deg = fdeg[u];
    if (deg == 0)
      continue;

    const uint64_t beg = offsets[u];
    uint64_t pcur = pos_offsets[u];
    uint64_t ncur = neg_offsets[u];

    for (uint32_t i = 0; i < deg; ++i) {
      if (signs[beg + i] > 0)
        pos_edges_out[pcur++] = edges[beg + i];
      else
        neg_edges_out[ncur++] = edges[beg + i];
    }
  }

  SplitForwardCSR out;
  out.n = n;
  out.pos_offsets = pos_offsets;
  out.pos_edges = pos_edges_out;
  out.neg_offsets = neg_offsets;
  out.neg_edges = neg_edges_out;
  return out;
}

static void free_split_forward_csr(SplitForwardCSR &s) {
  delete[] s.pos_offsets;
  delete[] s.pos_edges;
  delete[] s.neg_offsets;
  delete[] s.neg_edges;
  s.n = 0;
}

static inline uint64_t intersect_count_ranked(const uint32_t *A, uint64_t a0,
                                              uint64_t a1, const uint32_t *B,
                                              uint64_t b0, uint64_t b1,
                                              const uint32_t *rank_, uint32_t u,
                                              uint32_t v) {
  uint64_t cnt = 0;
  uint32_t rank_u = rank_[u];
  uint32_t rank_v = rank_[v];

  while (a0 < a1 && b0 < b1) {
    uint32_t x = A[a0];
    uint32_t y = B[b0];

    if (x == y) {
      if (rank_[x] > rank_u && rank_[x] > rank_v) {
        ++cnt;
      }
      ++a0;
      ++b0;
    } else if (x < y) {
      ++a0;
    } else {
      ++b0;
    }
  }
  return cnt;
}

static void count_signed_triangles(const SplitForwardCSR &csr,
                                   const uint32_t *rank_,
                                   uint64_t &total_balanced,
                                   uint64_t &total_unbalanced) {
  total_balanced = total_unbalanced = 0;

#pragma omp parallel reduction(+ : total_balanced, total_unbalanced)
  {
    uint64_t lbal = 0, lunbal = 0;

#pragma omp for schedule(dynamic, 16) nowait
    for (uint32_t u = 0; u < csr.n; ++u) {
      const uint64_t pu0 = csr.pos_offsets[u], pu1 = csr.pos_offsets[u + 1];
      const uint64_t nu0 = csr.neg_offsets[u], nu1 = csr.neg_offsets[u + 1];

      for (uint64_t e = pu0; e < pu1; ++e) {
        const uint32_t v = csr.pos_edges[e];
        const uint64_t pv0 = csr.pos_offsets[v], pv1 = csr.pos_offsets[v + 1];
        const uint64_t nv0 = csr.neg_offsets[v], nv1 = csr.neg_offsets[v + 1];

        lbal += intersect_count_ranked(csr.pos_edges, pu0, pu1, csr.pos_edges,
                                       pv0, pv1, rank_, u, v);
        lbal += intersect_count_ranked(csr.neg_edges, nu0, nu1, csr.neg_edges,
                                       nv0, nv1, rank_, u, v);
        lunbal += intersect_count_ranked(csr.pos_edges, pu0, pu1, csr.neg_edges,
                                         nv0, nv1, rank_, u, v);
        lunbal += intersect_count_ranked(csr.neg_edges, nu0, nu1, csr.pos_edges,
                                         pv0, pv1, rank_, u, v);
      }

      for (uint64_t e = nu0; e < nu1; ++e) {
        const uint32_t v = csr.neg_edges[e];
        const uint64_t pv0 = csr.pos_offsets[v], pv1 = csr.pos_offsets[v + 1];
        const uint64_t nv0 = csr.neg_offsets[v], nv1 = csr.neg_offsets[v + 1];

        lbal += intersect_count_ranked(csr.pos_edges, pu0, pu1, csr.neg_edges,
                                       nv0, nv1, rank_, u, v);
        lbal += intersect_count_ranked(csr.neg_edges, nu0, nu1, csr.pos_edges,
                                       pv0, pv1, rank_, u, v);
        lunbal += intersect_count_ranked(csr.pos_edges, pu0, pu1, csr.pos_edges,
                                         pv0, pv1, rank_, u, v);
        lunbal += intersect_count_ranked(csr.neg_edges, nu0, nu1, csr.neg_edges,
                                         nv0, nv1, rank_, u, v);
      }
    }

    total_balanced += lbal;
    total_unbalanced += lunbal;
  }
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: %s <graph_file>\n", argv[0]);
    return 1;
  }

  Timer total_time;

  Timer io_time;
  EdgeList el = load_graph(argv[1]);
  io_time.print("Graph Load (I/O + parse)");

  Timer csr_time;
  ForwardCSR csr = build_forward_csr(el);
  csr_time.print("Build Forward CSR");
  free_edge_list(el);

  Timer build_time;
  SplitForwardCSR split = build_split_forward_csr(csr);
  build_time.print(
      "Algorithm-Specific Preprocessing (Split CSR Build, incl. sort)");

  Timer algo_time;
  uint64_t balanced, unbalanced;
  count_signed_triangles(split, csr.rank_, balanced, unbalanced);

  algo_time.print("Triangle Counting Execution");
  printf("Balanced   = %llu\n", (unsigned long long)balanced);
  printf("Unbalanced = %llu\n", (unsigned long long)unbalanced);
  printf("Total      = %llu\n", (unsigned long long)(balanced + unbalanced));

  free_split_forward_csr(split);
  free_forward_csr(csr);
  total_time.print("Total execution time");
  return 0;
}