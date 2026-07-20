#pragma once

#include <gunrock/algorithms/algorithms.hxx>
#include <limits>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

namespace gunrock {
namespace signed_tc_hash {

struct param_t {};

struct result_t {
  std::size_t *total_balanced;
  std::size_t *total_unbalanced;

  result_t(std::size_t *b, std::size_t *u)
      : total_balanced(b), total_unbalanced(u) {}
};

template <typename vertex_t, typename weight_t> struct HTEntry {
  vertex_t key;
  weight_t sign;
};

template <typename graph_t, typename param_type, typename result_type>
struct problem_t : gunrock::problem_t<graph_t> {
  param_type param;
  result_type result;

  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

  static constexpr vertex_t EMPTY = std::numeric_limits<vertex_t>::max();

  thrust::device_vector<vertex_t> rank_;

  thrust::device_vector<int64_t> f_offsets;
  thrust::device_vector<vertex_t> f_edges;
  thrust::device_vector<weight_t> f_signs;

  thrust::device_vector<int64_t> ht_offsets;
  thrust::device_vector<HTEntry<vertex_t, weight_t>> ht_arena;

  unsigned long long *d_balanced = nullptr;
  unsigned long long *d_unbalanced = nullptr;

  problem_t(graph_t &G, param_type &_param, result_type &_result,
            std::shared_ptr<gcuda::multi_context_t> _context)
      : gunrock::problem_t<graph_t>(G, _context), param(_param),
        result(_result) {}

  __device__ static inline uint32_t hash32(uint32_t x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    return (x >> 16) ^ x;
  }

  __device__ static inline uint32_t next_pow2(uint32_t x) {
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

  void init() override {
    auto G = this->get_graph();
    vertex_t n = G.get_number_of_vertices();
    edge_t m = G.get_number_of_edges();

    thrust::device_vector<vertex_t> total_deg(n, vertex_t{0});
    auto *tdeg_ptr = total_deg.data().get();
    thrust::for_each(thrust::device, thrust::make_counting_iterator<edge_t>(0),
                     thrust::make_counting_iterator<edge_t>(m),
                     [G, tdeg_ptr] __device__(edge_t eid) {
                       vertex_t u = G.get_source_vertex(eid);
                       vertex_t v = G.get_destination_vertex(eid);
                       if (u == v)
                         return;
                       math::atomic::add(&tdeg_ptr[u], vertex_t{1});
                     });

    thrust::device_vector<vertex_t> rank_vec(n);
    thrust::sequence(thrust::device, rank_vec.begin(), rank_vec.end());
    auto *rank_ptr = rank_vec.data().get();
    thrust::sort(thrust::device, rank_ptr, rank_ptr + n,
                 [tdeg_ptr] __device__(vertex_t a, vertex_t b) {
                   if (tdeg_ptr[a] != tdeg_ptr[b])
                     return tdeg_ptr[a] < tdeg_ptr[b];
                   return a < b;
                 });

    rank_.resize(n);
    auto *rinv_ptr = rank_.data().get();
    thrust::for_each(thrust::device,
                     thrust::make_counting_iterator<vertex_t>(0),
                     thrust::make_counting_iterator<vertex_t>(n),
                     [rank_ptr, rinv_ptr] __device__(vertex_t i) {
                       rinv_ptr[rank_ptr[i]] = i;
                     });

    auto *rank_stable = rank_.data().get();

    f_offsets.assign(n + 1, int64_t{0});
    auto *f_off_ptr = f_offsets.data().get();

    thrust::for_each(
        thrust::device, thrust::make_counting_iterator<edge_t>(0),
        thrust::make_counting_iterator<edge_t>(m),
        [G, rank_stable, f_off_ptr] __device__(edge_t eid) {
          vertex_t u = G.get_source_vertex(eid);
          vertex_t v = G.get_destination_vertex(eid);
          if (u == v)
            return;
          if (rank_stable[u] >= rank_stable[v])
            return;

          math::atomic::add(
              reinterpret_cast<unsigned long long *>(f_off_ptr + u + 1), 1ULL);
        });

    thrust::inclusive_scan(thrust::device, f_offsets.begin(), f_offsets.end(),
                           f_offsets.begin());
    cudaDeviceSynchronize();
    int64_t f_m = f_offsets.back();

    f_edges.resize(f_m);
    f_signs.resize(f_m);

    thrust::device_vector<int64_t> f_cur = f_offsets;
    auto *f_cur_ptr = f_cur.data().get();
    auto *f_edge_ptr = f_edges.data().get();
    auto *f_sign_ptr = f_signs.data().get();

    thrust::for_each(
        thrust::device, thrust::make_counting_iterator<edge_t>(0),
        thrust::make_counting_iterator<edge_t>(m),
        [G, rank_stable, f_cur_ptr, f_edge_ptr,
         f_sign_ptr] __device__(edge_t eid) {
          vertex_t u = G.get_source_vertex(eid);
          vertex_t v = G.get_destination_vertex(eid);
          if (u == v)
            return;
          if (rank_stable[u] >= rank_stable[v])
            return;

          int64_t idx = math::atomic::add(
              reinterpret_cast<unsigned long long *>(&f_cur_ptr[u]), 1ULL);
          f_edge_ptr[idx] = v;
          f_sign_ptr[idx] = G.get_edge_weight(eid);
        });

    ht_offsets.assign(n + 1, int64_t{0});
    auto *ht_off_ptr = ht_offsets.data().get();

    thrust::for_each(
        thrust::device, thrust::make_counting_iterator<vertex_t>(0),
        thrust::make_counting_iterator<vertex_t>(n),
        [f_off_ptr, ht_off_ptr] __device__(vertex_t v) {
          uint32_t deg = static_cast<uint32_t>(f_off_ptr[v + 1] - f_off_ptr[v]);
          uint32_t cap = next_pow2(deg * 2 + 1);
          ht_off_ptr[v + 1] = static_cast<int64_t>(cap);
        });

    thrust::inclusive_scan(thrust::device, ht_offsets.begin(), ht_offsets.end(),
                           ht_offsets.begin());
    cudaDeviceSynchronize();
    int64_t arena_size = ht_offsets.back();
    ht_arena.resize(arena_size);

    auto *arena_ptr = ht_arena.data().get();

    thrust::for_each(
        thrust::device, thrust::make_counting_iterator<int64_t>(0),
        thrust::make_counting_iterator<int64_t>(arena_size),
        [arena_ptr] __device__(int64_t idx) { arena_ptr[idx].key = EMPTY; });

    thrust::for_each(
        thrust::device, thrust::make_counting_iterator<vertex_t>(0),
        thrust::make_counting_iterator<vertex_t>(n),
        [f_off_ptr, f_edge_ptr, f_sign_ptr, ht_off_ptr,
         arena_ptr] __device__(vertex_t v) {
          int64_t f_start = f_off_ptr[v];
          uint32_t deg = static_cast<uint32_t>(f_off_ptr[v + 1] - f_start);

          int64_t ht_start = ht_off_ptr[v];
          uint32_t mask =
              static_cast<uint32_t>(ht_off_ptr[v + 1] - ht_start) - 1;

          for (uint32_t i = 0; i < deg; ++i) {
            vertex_t target = f_edge_ptr[f_start + i];
            weight_t sign = f_sign_ptr[f_start + i];

            uint32_t pos = hash32(static_cast<uint32_t>(target)) & mask;
            while (arena_ptr[ht_start + pos].key != EMPTY) {
              pos = (pos + 1) & mask;
            }
            arena_ptr[ht_start + pos].key = target;
            arena_ptr[ht_start + pos].sign = sign;
          }
        });

    cudaMallocManaged(&d_balanced, sizeof(unsigned long long));
    cudaMallocManaged(&d_unbalanced, sizeof(unsigned long long));
    *d_balanced = 0ULL;
    *d_unbalanced = 0ULL;
  }

  void reset() override {
    if (d_balanced)
      *d_balanced = 0ULL;
    if (d_unbalanced)
      *d_unbalanced = 0ULL;
  }

  ~problem_t() {
    if (d_balanced) {
      cudaFree(d_balanced);
      d_balanced = nullptr;
    }
    if (d_unbalanced) {
      cudaFree(d_unbalanced);
      d_unbalanced = nullptr;
    }
  }
};

template <typename problem_t> struct enactor_t : gunrock::enactor_t<problem_t> {
  using vertex_t = typename problem_t::vertex_t;
  using weight_t = typename problem_t::weight_t;

  enactor_t(problem_t *_problem,
            std::shared_ptr<gcuda::multi_context_t> _context,
            enactor_properties_t _properties = enactor_properties_t{})
      : gunrock::enactor_t<problem_t>(_problem, _context, _properties) {}

  void loop(gcuda::multi_context_t &context) override {
    auto P = this->get_problem();
    vertex_t n = P->get_graph().get_number_of_vertices();

    auto *f_off = P->f_offsets.data().get();
    auto *f_e = P->f_edges.data().get();
    auto *f_s = P->f_signs.data().get();
    auto *rank = P->rank_.data().get();
    auto *ht_off = P->ht_offsets.data().get();
    auto *arena = P->ht_arena.data().get();
    auto *d_bal = P->d_balanced;
    auto *d_unbal = P->d_unbalanced;

    thrust::for_each(
        thrust::device, thrust::make_counting_iterator<vertex_t>(0),
        thrust::make_counting_iterator<vertex_t>(n),
        [f_off, f_e, f_s, rank, ht_off, arena, d_bal,
         d_unbal] __device__(vertex_t v) {
          int64_t begin_v = f_off[v];
          int64_t end_v = f_off[v + 1];
          uint32_t fdeg_v = static_cast<uint32_t>(end_v - begin_v);

          unsigned long long local_b = 0;
          unsigned long long local_ub = 0;

          for (int64_t wi = begin_v; wi < end_v; ++wi) {
            vertex_t w = f_e[wi];
            if (rank[v] >= rank[w])
              continue;

            weight_t sign_vw = f_s[wi];
            int64_t begin_w = f_off[w];
            uint32_t fdeg_w = static_cast<uint32_t>(f_off[w + 1] - begin_w);

            bool v_smaller = (fdeg_v <= fdeg_w);
            int64_t iter_start = v_smaller ? begin_v : begin_w;
            uint32_t iter_deg = v_smaller ? fdeg_v : fdeg_w;

            vertex_t probe_node = v_smaller ? w : v;
            int64_t ht_start = ht_off[probe_node];
            uint32_t mask =
                static_cast<uint32_t>(ht_off[probe_node + 1] - ht_start) - 1;

            for (uint32_t i = 0; i < iter_deg; ++i) {
              vertex_t u = f_e[iter_start + i];
              weight_t sign_u = f_s[iter_start + i];

              if (rank[u] <= rank[v] || rank[u] <= rank[w])
                continue;

              weight_t sign_wu = 0.0f;
              uint32_t pos = problem_t::hash32(static_cast<uint32_t>(u)) & mask;
              while (arena[ht_start + pos].key != problem_t::EMPTY) {
                if (arena[ht_start + pos].key == u) {
                  sign_wu = arena[ht_start + pos].sign;
                  break;
                }
                pos = (pos + 1) & mask;
              }

              if (sign_wu == 0.0f)
                continue;

              if ((sign_vw * sign_u * sign_wu) > 0.0f) {
                local_b++;
              } else {
                local_ub++;
              }
            }
          }

          if (local_b)
            math::atomic::add(d_bal, local_b);
          if (local_ub)
            math::atomic::add(d_unbal, local_ub);
        });

    context.get_context(0)->synchronize();

    *P->result.total_balanced = (std::size_t)*d_bal;
    *P->result.total_unbalanced = (std::size_t)*d_unbal;
  }

  bool is_converged(gcuda::multi_context_t &context) override {
    return this->iteration == 1;
  }
};

template <typename graph_t>
float run(graph_t &G, param_t &param, result_t &result,
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

} // namespace signed_tc_hash
} // namespace gunrock
