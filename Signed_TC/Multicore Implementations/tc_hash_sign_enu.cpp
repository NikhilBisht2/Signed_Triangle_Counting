#include "graph_io.h"
#include "timer.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>

using namespace std;

struct HTEntry {
  uint32_t key;
  int8_t sign;
};

struct AdjHashTable {
  HTEntry *table = nullptr;
  uint32_t mask = 0;
  static constexpr uint32_t EMPTY = UINT32_MAX;

  void build(HTEntry *buf, uint32_t cap, const uint32_t *edges,
             const int8_t *signs, uint32_t deg) {
    table = buf;
    mask = cap - 1;
    for (uint32_t i = 0; i < cap; i++)
      table[i].key = EMPTY;
    for (uint32_t i = 0; i < deg; i++) {
      uint32_t pos = hash32(edges[i]) & mask;
      while (table[pos].key != EMPTY)
        pos = (pos + 1) & mask;
      table[pos].key = edges[i];
      table[pos].sign = signs[i];
    }
  }

  inline int8_t query_found(uint32_t v, bool &found) const {
    uint32_t pos = hash32(v) & mask;
    while (table[pos].key != EMPTY) {
      if (table[pos].key == v) {
        found = true;
        return table[pos].sign;
      }
      pos = (pos + 1) & mask;
    }
    found = false;
    return 0;
  }

  inline int8_t query(uint32_t v) const {
    uint32_t pos = hash32(v) & mask;
    while (table[pos].key != EMPTY) {
      if (table[pos].key == v)
        return table[pos].sign;
      pos = (pos + 1) & mask;
    }
    return 0;
  }
  static inline uint32_t hash32(uint32_t x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    return (x >> 16) ^ x;
  }
  static inline uint32_t next_pow2(uint32_t x) {
    if (x == 0)
      return 1;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
  }
};

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

  Timer build_time;
  uint64_t *ht_offsets = new uint64_t[n + 1];
  ht_offsets[0] = 0;
  for (uint32_t v = 0; v < n; v++) {
    uint32_t cap = AdjHashTable::next_pow2(fdeg[v] * 2 + 1);
    ht_offsets[v + 1] = ht_offsets[v] + cap;
  }

  const uint64_t arena_size = ht_offsets[n];
  HTEntry *arena = new HTEntry[arena_size];
  AdjHashTable *ht = new AdjHashTable[n];

#pragma omp parallel for schedule(dynamic, 64)
  for (uint32_t v = 0; v < n; v++) {
    uint32_t cap = (uint32_t)(ht_offsets[v + 1] - ht_offsets[v]);
    ht[v].build(arena + ht_offsets[v], cap, f_edges + f_offsets[v],
                f_signs + f_offsets[v], fdeg[v]);
  }
  delete[] ht_offsets;
  build_time.print("Algorithm-Specific Preprocessing (Hash Table Build)");

  uint64_t total_balanced = 0;
  uint64_t total_unbalanced = 0;

  Timer algo_time;
#pragma omp parallel for schedule(dynamic, 16)                                 \
    reduction(+ : total_balanced, total_unbalanced)
  for (uint32_t v = 0; v < n; v++) {
    uint64_t begin_v = f_offsets[v];
    uint64_t end_v = f_offsets[v + 1];

    for (uint64_t wi = begin_v; wi < end_v; wi++) {
      uint32_t w = f_edges[wi];
      if (rank_[v] >= rank_[w])
        continue;

      int8_t sign_vw = f_signs[wi];

      bool v_smaller = fdeg[v] <= fdeg[w];
      const uint32_t *iter_edges =
          v_smaller ? f_edges + f_offsets[v] : f_edges + f_offsets[w];
      const int8_t *iter_signs =
          v_smaller ? f_signs + f_offsets[v] : f_signs + f_offsets[w];
      uint32_t iter_deg = v_smaller ? fdeg[v] : fdeg[w];
      const AdjHashTable &probe = v_smaller ? ht[w] : ht[v];

      for (uint32_t i = 0; i < iter_deg; i++) {
        uint32_t u = iter_edges[i];
        int8_t sign_u = iter_signs[i];

        if (rank_[u] <= rank_[v] || rank_[u] <= rank_[w])
          continue;

        bool found = false;
        int8_t sign_wu = probe.query_found(u, found);
        if (!found)
          continue;

        int prod = sign_vw * sign_u * sign_wu;
        if (prod > 0)
          total_balanced++;
        else
          total_unbalanced++;
      }
    }
  }
  algo_time.print("Triangle Counting Execution");
  printf("Balanced   = %llu\n", (unsigned long long)total_balanced);
  printf("Unbalanced = %llu\n", (unsigned long long)total_unbalanced);
  printf("Total      = %llu\n",
         (unsigned long long)(total_balanced + total_unbalanced));

  delete[] arena;
  delete[] ht;
  free_forward_csr(csr);
  total_time.print("Total execution time");
  return 0;
}
