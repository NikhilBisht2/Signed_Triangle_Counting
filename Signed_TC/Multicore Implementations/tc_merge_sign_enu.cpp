#include "graph_io.h"
#include "timer.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>

using namespace std;

static inline void
intersect_signed_ranked(const uint32_t *edges, const int8_t *signs,
                        const uint32_t *rank_, uint64_t a_begin, uint64_t a_end,
                        uint64_t b_begin, uint64_t b_end, uint32_t v,
                        uint32_t w, int8_t sign_vw, uint64_t &balanced,
                        uint64_t &unbalanced) {
  uint64_t local_bal = 0;
  uint64_t local_unbal = 0;
  uint32_t rank_v = rank_[v];
  uint32_t rank_w = rank_[w];

  while (a_begin < a_end && b_begin < b_end) {
    uint32_t x = edges[a_begin];
    uint32_t y = edges[b_begin];
    if (x == y) {

      if (rank_[x] > rank_v && rank_[x] > rank_w) {
        int prod = sign_vw * signs[a_begin] * signs[b_begin];
        if (prod > 0)
          local_bal++;
        else
          local_unbal++;
      }
      a_begin++;
      b_begin++;
    } else if (x < y)
      a_begin++;
    else
      b_begin++;
  }
  balanced += local_bal;
  unbalanced += local_unbal;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: %s <graph_file_path>\n", argv[0]);
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

  const uint32_t n = csr.n;
  const uint64_t *f_offsets = csr.offsets;
  const uint32_t *f_edges = csr.edges;
  const int8_t *f_signs = csr.signs;
  const uint32_t *fdeg = csr.fdeg;
  const uint32_t *rank_ = csr.rank_;

  Timer sort_time;
  uint32_t max_fdeg_global = 0;
#pragma omp parallel for reduction(max : max_fdeg_global)
  for (uint32_t v = 0; v < n; v++) {
    if (fdeg[v] > max_fdeg_global)
      max_fdeg_global = fdeg[v];
  }

#pragma omp parallel
  {
    uint32_t *tmp_edges = new uint32_t[max_fdeg_global + 1];
    int8_t *tmp_signs = new int8_t[max_fdeg_global + 1];

#pragma omp for schedule(dynamic, 64)
    for (uint32_t v = 0; v < n; v++) {
      uint64_t begin = f_offsets[v];
      uint32_t deg = fdeg[v];
      if (deg < 2)
        continue;

      std::memcpy(tmp_edges, f_edges + begin, deg * sizeof(uint32_t));
      std::memcpy(tmp_signs, f_signs + begin, deg * sizeof(int8_t));

      if (deg <= 32) {
        for (uint32_t i = 1; i < deg; i++) {
          uint32_t ke = tmp_edges[i];
          int8_t ks = tmp_signs[i];
          int32_t j = (int32_t)i - 1;
          while (j >= 0 && tmp_edges[j] > ke) {
            tmp_edges[j + 1] = tmp_edges[j];
            tmp_signs[j + 1] = tmp_signs[j];
            j--;
          }
          tmp_edges[j + 1] = ke;
          tmp_signs[j + 1] = ks;
        }
        std::memcpy((void *)(f_edges + begin), tmp_edges,
                    deg * sizeof(uint32_t));
        std::memcpy((void *)(f_signs + begin), tmp_signs, deg * sizeof(int8_t));
      } else {
        uint32_t *idx = new uint32_t[deg];
        for (uint32_t i = 0; i < deg; i++)
          idx[i] = i;
        std::sort(idx, idx + deg, [&](uint32_t a, uint32_t b) {
          return tmp_edges[a] < tmp_edges[b];
        });
        for (uint32_t i = 0; i < deg; i++) {
          ((uint32_t *)f_edges)[begin + i] = tmp_edges[idx[i]];
          ((int8_t *)f_signs)[begin + i] = tmp_signs[idx[i]];
        }
        delete[] idx;
      }
    }
    delete[] tmp_edges;
    delete[] tmp_signs;
  }
  sort_time.print("Algorithm-Specific Preprocessing (Adjacency List Sort)");

  uint64_t total_balanced = 0;
  uint64_t total_unbalanced = 0;

  Timer algo_time;
#pragma omp parallel for schedule(dynamic, 16)                                 \
    reduction(+ : total_balanced, total_unbalanced)
  for (uint32_t v = 0; v < n; v++) {
    uint64_t begin_v = f_offsets[v];
    uint64_t end_v = f_offsets[v + 1];

    for (uint64_t e = begin_v; e < end_v; e++) {
      uint32_t w = f_edges[e];
      intersect_signed_ranked(f_edges, f_signs, rank_, begin_v, end_v,
                              f_offsets[w], f_offsets[w + 1], v, w, f_signs[e],
                              total_balanced, total_unbalanced);
    }
  }
  algo_time.print("Triangle Counting Execution");
  printf("Balanced   = %llu\n", (unsigned long long)total_balanced);
  printf("Unbalanced = %llu\n", (unsigned long long)total_unbalanced);
  printf("Total      = %llu\n",
         (unsigned long long)(total_balanced + total_unbalanced));

  free_forward_csr(csr);
  total_time.print("Total execution time");
  return 0;
}
