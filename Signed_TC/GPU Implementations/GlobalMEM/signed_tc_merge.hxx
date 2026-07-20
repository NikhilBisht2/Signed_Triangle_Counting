#pragma once

#include <gunrock/algorithms/algorithms.hxx>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

namespace gunrock {
namespace signed_tc_merge {

struct param_t {};

struct result_t {
  std::size_t *total_balanced;
  std::size_t *total_unbalanced;

  result_t(std::size_t *b, std::size_t *u)
      : total_balanced(b), total_unbalanced(u) {}
};

template <typename graph_t, typename param_type, typename result_type>
struct problem_t : gunrock::problem_t<graph_t> {
  param_type param;
  result_type result;

  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

  thrust::device_vector<vertex_t> rank_;

  thrust::device_vector<int64_t> f_offsets;
  thrust::device_vector<vertex_t> f_edges;
  thrust::device_vector<weight_t> f_signs;

  unsigned long long *d_balanced = nullptr;
  unsigned long long *d_unbalanced = nullptr;

  problem_t(graph_t &G, param_type &_param, result_type &_result,
            std::shared_ptr<gcuda::multi_context_t> _context)
      : gunrock::problem_t<graph_t>(G, _context), param(_param),
        result(_result) {}

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

    thrust::for_each(
        thrust::device, thrust::make_counting_iterator<vertex_t>(0),
        thrust::make_counting_iterator<vertex_t>(n),
        [f_off_ptr, f_edge_ptr, f_sign_ptr] __device__(vertex_t u) {
          int64_t start = f_off_ptr[u];
          int64_t end = f_off_ptr[u + 1];
          int64_t len = end - start;
          if (len < 2)
            return;

          for (int64_t i = start + 1; i < end; ++i) {
            vertex_t key_e = f_edge_ptr[i];
            weight_t key_s = f_sign_ptr[i];
            int64_t j = i - 1;
            while (j >= start && f_edge_ptr[j] > key_e) {
              f_edge_ptr[j + 1] = f_edge_ptr[j];
              f_sign_ptr[j + 1] = f_sign_ptr[j];
              j--;
            }
            f_edge_ptr[j + 1] = key_e;
            f_sign_ptr[j + 1] = key_s;
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
    auto *d_bal = P->d_balanced;
    auto *d_unbal = P->d_unbalanced;

    thrust::for_each(
        thrust::device, thrust::make_counting_iterator<vertex_t>(0),
        thrust::make_counting_iterator<vertex_t>(n),
        [f_off, f_e, f_s, rank, d_bal, d_unbal] __device__(vertex_t v) {
          int64_t begin_v = f_off[v];
          int64_t end_v = f_off[v + 1];

          unsigned long long local_b = 0;
          unsigned long long local_ub = 0;

          uint32_t rank_v = rank[v];

          for (int64_t e = begin_v; e < end_v; ++e) {
            vertex_t w = f_e[e];
            uint32_t rank_w = rank[w];
            weight_t sign_vw = f_s[e];

            int64_t a_begin = begin_v;
            int64_t a_end = end_v;
            int64_t b_begin = f_off[w];
            int64_t b_end = f_off[w + 1];

            while (a_begin < a_end && b_begin < b_end) {
              vertex_t x = f_e[a_begin];
              vertex_t y = f_e[b_begin];

              if (x == y) {
                if (rank[x] > rank_v && rank[x] > rank_w) {
                  if ((sign_vw * f_s[a_begin] * f_s[b_begin]) > 0.0f) {
                    local_b++;
                  } else {
                    local_ub++;
                  }
                }
                a_begin++;
                b_begin++;
              } else if (x < y) {
                a_begin++;
              } else {
                b_begin++;
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

} // namespace signed_tc_merge
} // namespace gunrock
