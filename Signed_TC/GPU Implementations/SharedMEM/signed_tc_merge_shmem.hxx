#pragma once

#include <gunrock/algorithms/algorithms.hxx>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

namespace gunrock {
namespace signed_tc_merge_shmem {

static constexpr int WARP_SIZE = 32;
static constexpr int WARPS_PER_BLOCK = 8;
static constexpr int BLOCK_DIM = WARP_SIZE * WARPS_PER_BLOCK;

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

    {
      thrust::device_vector<uint64_t> sort_keys(f_m);
      auto *sort_key_ptr = sort_keys.data().get();

      thrust::for_each(
          thrust::device, thrust::make_counting_iterator<vertex_t>(0),
          thrust::make_counting_iterator<vertex_t>(n),
          [f_off_ptr, f_edge_ptr, sort_key_ptr] __device__(vertex_t u) {
            for (int64_t i = f_off_ptr[u]; i < f_off_ptr[u + 1]; ++i) {
              sort_key_ptr[i] =
                  (static_cast<uint64_t>(static_cast<uint32_t>(u)) << 32) |
                  static_cast<uint64_t>(static_cast<uint32_t>(f_edge_ptr[i]));
            }
          });

      auto val_begin = thrust::make_zip_iterator(
          thrust::make_tuple(f_edges.begin(), f_signs.begin()));
      thrust::stable_sort_by_key(thrust::device, sort_keys.begin(),
                                 sort_keys.end(), val_begin);
    }

    cudaMalloc(&d_balanced, sizeof(unsigned long long));
    cudaMalloc(&d_unbalanced, sizeof(unsigned long long));
    cudaMemset(d_balanced, 0, sizeof(unsigned long long));
    cudaMemset(d_unbalanced, 0, sizeof(unsigned long long));
  }

  void reset() override {
    if (d_balanced)
      cudaMemset(d_balanced, 0, sizeof(unsigned long long));
    if (d_unbalanced)
      cudaMemset(d_unbalanced, 0, sizeof(unsigned long long));
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

__global__ void warp_merge_kernel(int n, const int64_t *__restrict__ f_off,
                                  const int *__restrict__ f_e,
                                  const float *__restrict__ f_s,
                                  const int *__restrict__ rank,
                                  unsigned long long *d_bal,
                                  unsigned long long *d_unbal) {

  int lane = threadIdx.x & 31;
  int warp_id = threadIdx.x >> 5;

  int global_warp = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  int warp_stride = gridDim.x * WARPS_PER_BLOCK;

  for (int v = global_warp; v < n; v += warp_stride) {

    int64_t bv = f_off[v];
    int64_t ev = f_off[v + 1];
    int dv = (int)(ev - bv);
    int rank_v = rank[v];

    unsigned long long local_bal = 0;
    unsigned long long local_unbal = 0;

    for (int wi_off = lane; wi_off < dv; wi_off += WARP_SIZE) {
      int64_t wi = bv + wi_off;
      int w = f_e[wi];
      float sign_vw = f_s[wi];
      int rank_w = rank[w];

      int64_t bw = f_off[w];
      int64_t ew = f_off[w + 1];

      int64_t a = bv;
      int64_t b = bw;

      while (a < ev && b < ew) {
        int u_a = f_e[a];
        int u_b = f_e[b];

        if (u_a == u_b) {
          int rank_u = rank[u_a];
          if (rank_u > rank_v && rank_u > rank_w) {
            if (sign_vw * f_s[a] * f_s[b] > 0.0f)
              local_bal++;
            else
              local_unbal++;
          }
          a++;
          b++;
        } else if (u_a < u_b) {
          a++;
        } else {
          b++;
        }
      }
    }

    unsigned long long sum_bal = local_bal;
    unsigned long long sum_unbal = local_unbal;

#pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
      sum_bal += __shfl_down_sync(0xFFFFFFFF, sum_bal, offset);
      sum_unbal += __shfl_down_sync(0xFFFFFFFF, sum_unbal, offset);
    }

    if (lane == 0) {
      if (sum_bal)
        atomicAdd(d_bal, sum_bal);
      if (sum_unbal)
        atomicAdd(d_unbal, sum_unbal);
    }
  }
}

template <typename problem_t> struct enactor_t : gunrock::enactor_t<problem_t> {
  using vertex_t = typename problem_t::vertex_t;
  using weight_t = typename problem_t::weight_t;

  enactor_t(problem_t *_problem,
            std::shared_ptr<gcuda::multi_context_t> _context,
            enactor_properties_t _properties = enactor_properties_t{})
      : gunrock::enactor_t<problem_t>(_problem, _context, _properties) {}

  void loop(gcuda::multi_context_t &context) override {
    auto P = this->get_problem();
    int n = (int)P->get_graph().get_number_of_vertices();

    auto *f_off = P->f_offsets.data().get();
    auto *f_e = P->f_edges.data().get();
    auto *f_s = P->f_signs.data().get();
    auto *rank = P->rank_.data().get();
    auto *d_bal = P->d_balanced;
    auto *d_unbal = P->d_unbalanced;

    int device;
    cudaGetDevice(&device);
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);

    int max_blocks = (n + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    int grid = min(max_blocks, num_sms * 32);

    warp_merge_kernel<<<grid, BLOCK_DIM>>>(n, f_off, f_e, f_s, rank, d_bal,
                                           d_unbal);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
      std::printf("Launch error = %s\n", cudaGetErrorString(err));

    context.get_context(0)->synchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess)
      std::printf("Runtime error = %s\n", cudaGetErrorString(err));

    unsigned long long h_bal = 0, h_unbal = 0;
    cudaMemcpy(&h_bal, d_bal, sizeof(unsigned long long),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_unbal, d_unbal, sizeof(unsigned long long),
               cudaMemcpyDeviceToHost);

    *P->result.total_balanced = static_cast<std::size_t>(h_bal);
    *P->result.total_unbalanced = static_cast<std::size_t>(h_unbal);
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

} // namespace signed_tc_merge_shmem
} // namespace gunrock
