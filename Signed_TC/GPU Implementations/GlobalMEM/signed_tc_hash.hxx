/**
 * @file signed_tc_hash.hxx
 * @brief Hash-Based Signed Triangle Counting on GPU via Gunrock/Essentials.
 *
 * Mirrors the CPU pipeline from tc_hash_sign_enu.cpp:
 * 1. Build a degree-ordered forward DAG.
 * Rank = ascending degree; ties broken by ascending vertex ID
 * (matches graph_io.h counting-sort stable order).
 * 2. Allocate and build a flattened open-addressed hash table arena for forward adjacent edges.
 * 3. Match edge queries via a GPU-optimized 32-bit hash function using a 2-pointer degree optimization step.
 *
 * Inherited Fixes from signed_tc.hxx:
 * [A] Step A: source vertex only increments to avoid double counting degrees in symmetric CSR.
 * [B] Offset arrays use int64_t to avoid overflow at scale.
 * [C] cudaDeviceSynchronize() before reading .back() from device vectors.
 * [E] Capture rank stable member pointers directly to avoid dangling host local variables in device lambdas.
 */
#pragma once

#include <gunrock/algorithms/algorithms.hxx>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <limits>

namespace gunrock {
namespace signed_tc_hash {

// ============================================================
// param / result
// ============================================================

struct param_t {
  // No user-facing parameters needed.
};

struct result_t {
  std::size_t* total_balanced;
  std::size_t* total_unbalanced;

  result_t(std::size_t* b, std::size_t* u)
      : total_balanced(b), total_unbalanced(u) {}
};

// ============================================================
// Hash Entry Struct
// ============================================================
template <typename vertex_t, typename weight_t>
struct HTEntry {
  vertex_t key;
  weight_t sign;
};

// ============================================================
// problem_t  —  device storage
// ============================================================

template <typename graph_t, typename param_type, typename result_type>
struct problem_t : gunrock::problem_t<graph_t> {
  param_type  param;
  result_type result;

  using vertex_t = typename graph_t::vertex_type;
  using edge_t   = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

  static constexpr vertex_t EMPTY = std::numeric_limits<vertex_t>::max();

  // Rank array: rank_[v] = DAG rank of vertex v (ascending out-degree).
  thrust::device_vector<vertex_t> rank_;

  // Forward-CSR (Unsplit) built in init() using int64_t offsets [B]
  thrust::device_vector<int64_t>  f_offsets;  // [n+1]
  thrust::device_vector<vertex_t> f_edges;
  thrust::device_vector<weight_t> f_signs;

  // Global Arena Open-Addressed Hash Tables
  thrust::device_vector<int64_t>  ht_offsets; // [n+1] Base index into arena for vertex v
  thrust::device_vector<HTEntry<vertex_t, weight_t>> ht_arena;

  // Global accumulators (unified memory)
  unsigned long long* d_balanced   = nullptr;
  unsigned long long* d_unbalanced = nullptr;

  problem_t(graph_t& G,
            param_type& _param,
            result_type& _result,
            std::shared_ptr<gcuda::multi_context_t> _context)
      : gunrock::problem_t<graph_t>(G, _context),
        param(_param),
        result(_result) {}

  // 32-bit Murmur-esque mixing hash function matching CPU logic
  __device__ static inline uint32_t hash32(uint32_t x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    return (x >> 16) ^ x;
  }

  // Next power of 2 utility to ensure hash table power of 2 masking works
  __device__ static inline uint32_t next_pow2(uint32_t x) {
    if (x == 0) return 1;
    x--;
    x |= x >> 1;  x |= x >> 2;
    x |= x >> 4;  x |= x >> 8;
    x |= x >> 16;
    return x + 1;
  }

  void init() override {
    auto G     = this->get_graph();
    vertex_t n = G.get_number_of_vertices();
    edge_t   m = G.get_number_of_edges();   // all directed edges in symmetric CSR

    // ----------------------------------------------------------------
    // Step A: compute out-degree per vertex from symmetric directed CSR [A]
    // ----------------------------------------------------------------
    thrust::device_vector<vertex_t> total_deg(n, vertex_t{0});
    auto* tdeg_ptr = total_deg.data().get();
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator<edge_t>(0),
        thrust::make_counting_iterator<edge_t>(m),
        [G, tdeg_ptr] __device__(edge_t eid) {
          vertex_t u = G.get_source_vertex(eid);
          vertex_t v = G.get_destination_vertex(eid);
          if (u == v) return;
          math::atomic::add(&tdeg_ptr[u], vertex_t{1});
        });

    // ----------------------------------------------------------------
    // Step B: build rank array
    // ----------------------------------------------------------------
    thrust::device_vector<vertex_t> rank_vec(n);
    thrust::sequence(thrust::device, rank_vec.begin(), rank_vec.end());
    auto* rank_ptr = rank_vec.data().get();
    thrust::sort(thrust::device, rank_ptr, rank_ptr + n,
                 [tdeg_ptr] __device__(vertex_t a, vertex_t b) {
                   if (tdeg_ptr[a] != tdeg_ptr[b])
                     return tdeg_ptr[a] < tdeg_ptr[b];
                   return a < b;
                 });

    rank_.resize(n);
    auto* rinv_ptr = rank_.data().get();
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator<vertex_t>(0),
        thrust::make_counting_iterator<vertex_t>(n),
        [rank_ptr, rinv_ptr] __device__(vertex_t i) {
          rinv_ptr[rank_ptr[i]] = i;
        });

    auto* rank_stable = rank_.data().get(); // FIX [E]

    // ----------------------------------------------------------------
    // Step C: build standard Unsplit Forward DAG CSR
    // ----------------------------------------------------------------
    f_offsets.assign(n + 1, int64_t{0}); // FIX [B]
    auto* f_off_ptr = f_offsets.data().get();

    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator<edge_t>(0),
        thrust::make_counting_iterator<edge_t>(m),
        [G, rank_stable, f_off_ptr] __device__(edge_t eid) {
          vertex_t u = G.get_source_vertex(eid);
          vertex_t v = G.get_destination_vertex(eid);
          if (u == v) return;
          if (rank_stable[u] >= rank_stable[v]) return; // DAG gate

          math::atomic::add(reinterpret_cast<unsigned long long*>(f_off_ptr + u + 1), 1ULL);
        });

    thrust::inclusive_scan(thrust::device, f_offsets.begin(), f_offsets.end(), f_offsets.begin());
    cudaDeviceSynchronize(); // FIX [C]
    int64_t f_m = f_offsets.back();

    f_edges.resize(f_m);
    f_signs.resize(f_m);

    thrust::device_vector<int64_t> f_cur = f_offsets;
    auto* f_cur_ptr   = f_cur.data().get();
    auto* f_edge_ptr  = f_edges.data().get();
    auto* f_sign_ptr  = f_signs.data().get();

    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator<edge_t>(0),
        thrust::make_counting_iterator<edge_t>(m),
        [G, rank_stable, f_cur_ptr, f_edge_ptr, f_sign_ptr] __device__(edge_t eid) {
          vertex_t u = G.get_source_vertex(eid);
          vertex_t v = G.get_destination_vertex(eid);
          if (u == v) return;
          if (rank_stable[u] >= rank_stable[v]) return;

          int64_t idx = math::atomic::add(reinterpret_cast<unsigned long long*>(&f_cur_ptr[u]), 1ULL);
          f_edge_ptr[idx] = v;
          f_sign_ptr[idx] = G.get_edge_weight(eid);
        });

    // ----------------------------------------------------------------
    // Step D: Precompute Hash Capacities and allocate flattened Arena
    // ----------------------------------------------------------------
    ht_offsets.assign(n + 1, int64_t{0});
    auto* ht_off_ptr = ht_offsets.data().get();

    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator<vertex_t>(0),
        thrust::make_counting_iterator<vertex_t>(n),
        [f_off_ptr, ht_off_ptr] __device__(vertex_t v) {
          uint32_t deg = static_cast<uint32_t>(f_off_ptr[v + 1] - f_off_ptr[v]);
          uint32_t cap = next_pow2(deg * 2 + 1); // Maintain load factor < 0.5 for speed
          ht_off_ptr[v + 1] = static_cast<int64_t>(cap);
        });

    thrust::inclusive_scan(thrust::device, ht_offsets.begin(), ht_offsets.end(), ht_offsets.begin());
    cudaDeviceSynchronize();
    int64_t arena_size = ht_offsets.back();
    ht_arena.resize(arena_size);

    auto* arena_ptr = ht_arena.data().get();
    
    // Initialize keys to EMPTY
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator<int64_t>(0),
        thrust::make_counting_iterator<int64_t>(arena_size),
        [arena_ptr] __device__(int64_t idx) {
          arena_ptr[idx].key = EMPTY;
        });

    // Populate Hash Tables across global arena structure
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator<vertex_t>(0),
        thrust::make_counting_iterator<vertex_t>(n),
        [f_off_ptr, f_edge_ptr, f_sign_ptr, ht_off_ptr, arena_ptr] __device__(vertex_t v) {
          int64_t f_start = f_off_ptr[v];
          uint32_t deg = static_cast<uint32_t>(f_off_ptr[v + 1] - f_start);
          
          int64_t ht_start = ht_off_ptr[v];
          uint32_t mask = static_cast<uint32_t>(ht_off_ptr[v + 1] - ht_start) - 1;

          for (uint32_t i = 0; i < deg; ++i) {
            vertex_t target = f_edge_ptr[f_start + i];
            weight_t sign   = f_sign_ptr[f_start + i];

            uint32_t pos = hash32(static_cast<uint32_t>(target)) & mask;
            while (arena_ptr[ht_start + pos].key != EMPTY) {
              pos = (pos + 1) & mask;
            }
            arena_ptr[ht_start + pos].key  = target;
            arena_ptr[ht_start + pos].sign = sign;
          }
        });

    cudaMallocManaged(&d_balanced,   sizeof(unsigned long long));
    cudaMallocManaged(&d_unbalanced, sizeof(unsigned long long));
    *d_balanced   = 0ULL;
    *d_unbalanced = 0ULL;
  }

  void reset() override {
    if (d_balanced)   *d_balanced   = 0ULL;
    if (d_unbalanced) *d_unbalanced = 0ULL;
  }

  ~problem_t() {
    if (d_balanced)   { cudaFree(d_balanced);   d_balanced   = nullptr; }
    if (d_unbalanced) { cudaFree(d_unbalanced); d_unbalanced = nullptr; }
  }
};

// ============================================================
// enactor_t
// ============================================================

template <typename problem_t>
struct enactor_t : gunrock::enactor_t<problem_t> {
  using vertex_t = typename problem_t::vertex_t;
  using weight_t = typename problem_t::weight_t;

  enactor_t(problem_t* _problem,
            std::shared_ptr<gcuda::multi_context_t> _context,
            enactor_properties_t _properties = enactor_properties_t{})
      : gunrock::enactor_t<problem_t>(_problem, _context, _properties) {}

  void loop(gcuda::multi_context_t& context) override {
    auto P = this->get_problem();
    vertex_t n = P->get_graph().get_number_of_vertices();

    auto* f_off   = P->f_offsets.data().get();
    auto* f_e     = P->f_edges.data().get();
    auto* f_s     = P->f_signs.data().get();
    auto* rank    = P->rank_.data().get();
    auto* ht_off  = P->ht_offsets.data().get();
    auto* arena   = P->ht_arena.data().get();
    auto* d_bal   = P->d_balanced;
    auto* d_unbal = P->d_unbalanced;

    // FIX [D]: Map directly over vertex ID range exactly matching tc_hash_sign_enu.cpp
    thrust::for_each(
        thrust::device,
        thrust::make_counting_iterator<vertex_t>(0),
        thrust::make_counting_iterator<vertex_t>(n),
        [f_off, f_e, f_s, rank, ht_off, arena, d_bal, d_unbal] __device__(vertex_t v) {
          int64_t begin_v = f_off[v];
          int64_t end_v   = f_off[v + 1];
          uint32_t fdeg_v = static_cast<uint32_t>(end_v - begin_v);

          unsigned long long local_b  = 0;
          unsigned long long local_ub = 0;

          for (int64_t wi = begin_v; wi < end_v; ++wi) {
            vertex_t w = f_e[wi];
            if (rank[v] >= rank[w]) continue;

            weight_t sign_vw = f_s[wi];
            int64_t begin_w  = f_off[w];
            uint32_t fdeg_w  = static_cast<uint32_t>(f_off[w + 1] - begin_w);

            // Optimization: Iterate through the smaller forward degree neighbor array
            bool v_smaller = (fdeg_v <= fdeg_w);
            int64_t iter_start   = v_smaller ? begin_v : begin_w;
            uint32_t iter_deg    = v_smaller ? fdeg_v  : fdeg_w;
            
            // Query target table configuration
            vertex_t probe_node = v_smaller ? w : v;
            int64_t ht_start    = ht_off[probe_node];
            uint32_t mask       = static_cast<uint32_t>(ht_off[probe_node + 1] - ht_start) - 1;

            for (uint32_t i = 0; i < iter_deg; ++i) {
              vertex_t u = f_e[iter_start + i];
              weight_t sign_u = f_s[iter_start + i];

              if (rank[u] <= rank[v] || rank[u] <= rank[w]) continue;

              // Inline hash probing query
              weight_t sign_wu = 0.0f;
              uint32_t pos = problem_t::hash32(static_cast<uint32_t>(u)) & mask;
              while (arena[ht_start + pos].key != problem_t::EMPTY) {
                if (arena[ht_start + pos].key == u) {
                  sign_wu = arena[ht_start + pos].sign;
                  break;
                }
                pos = (pos + 1) & mask;
              }

              if (sign_wu == 0.0f) continue; // Not found

              if ((sign_vw * sign_u * sign_wu) > 0.0f) {
                local_b++;
              } else {
                local_ub++;
              }
            }
          }

          if (local_b)  math::atomic::add(d_bal,   local_b);
          if (local_ub) math::atomic::add(d_unbal, local_ub);
        });

    context.get_context(0)->synchronize();

    *P->result.total_balanced   = (std::size_t)*d_bal;
    *P->result.total_unbalanced = (std::size_t)*d_unbal;
  }

  bool is_converged(gcuda::multi_context_t& context) override {
    return this->iteration == 1;
  }
};

// ============================================================
// run()
// ============================================================

template <typename graph_t>
float run(graph_t& G,
          param_t& param,
          result_t& result,
          std::shared_ptr<gcuda::multi_context_t> context) {
  using problem_type = problem_t<graph_t, param_t, result_t>;
  using enactor_type = enactor_t<problem_type>;

  enactor_properties_t props;
  props.self_manage_frontiers = true;

  problem_type problem(G, param, result, context);
  problem.init();
  problem.reset();

  enactor_type enactor(&problem, context, props);
  return enactor.enact();
}

}  // namespace signed_tc_hash
}  // namespace gunrock
