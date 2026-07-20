#pragma once

#include <gunrock/algorithms/algorithms.hxx>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

namespace gunrock {
namespace signed_tc {

struct param_t {};

struct result_t {
  std::size_t *total_balanced;
  std::size_t *total_unbalanced;
  result_t(std::size_t *b, std::size_t *u)
      : total_balanced(b), total_unbalanced(u) {}
};

inline void mem_instrument(const char *tag) {
  size_t free_b, total_b;
  cudaMemGetInfo(&free_b, &total_b);
  std::cout << "[MEM_INSTRUMENT] " << tag
            << " free=" << free_b / (1024.0 * 1024.0) << " MB\n";
}

template <typename vertex_t>
__device__ __forceinline__ unsigned long long
intersect_count_warp(const vertex_t *__restrict__ A, int64_t a0, int64_t a1,
                     const vertex_t *__restrict__ B, int64_t b0, int64_t b1,
                     int lane_id) {
  int64_t lenA = a1 - a0;
  int64_t lenB = b0 - b1;

  if (a1 <= a0 || b1 <= b0)
    return 0;
  lenA = a1 - a0;
  lenB = b1 - b0;

  unsigned long long local_cnt = 0;

  if (lenA > lenB * 4) {

    for (int64_t b_idx = lane_id; b_idx < lenB; b_idx += 32) {
      vertex_t target = B[b0 + b_idx];
      int64_t low = a0, high = a1 - 1;
      while (low <= high) {
        int64_t mid = low + (high - low) / 2;
        vertex_t val = A[mid];
        if (val == target) {
          local_cnt++;
          break;
        } else if (val < target)
          low = mid + 1;
        else
          high = mid - 1;
      }
    }
  } else if (lenB > lenA * 4) {

    for (int64_t a_idx = lane_id; a_idx < lenA; a_idx += 32) {
      vertex_t target = A[a0 + a_idx];
      int64_t low = b0, high = b1 - 1;
      while (low <= high) {
        int64_t mid = low + (high - low) / 2;
        vertex_t val = B[mid];
        if (val == target) {
          local_cnt++;
          break;
        } else if (val < target)
          low = mid + 1;
        else
          high = mid - 1;
      }
    }
  } else {

    int64_t a_idx = 0;
    int64_t b_idx = 0;
    while (a_idx < lenA && b_idx < lenB) {
      int64_t remA = lenA - a_idx;
      int64_t step = (remA < 32) ? remA : 32;
      vertex_t y = B[b0 + b_idx];
      bool match = false;
      int64_t matched_a = -1;

      if (lane_id < step) {
        vertex_t x = A[a0 + a_idx + lane_id];
        if (x == y) {
          match = true;
          matched_a = a_idx + lane_id;
        }
      }

      int mask = __ballot_sync(0xFFFFFFFF, match);
      if (mask != 0) {
        if (lane_id == 0)
          local_cnt++;
        int leader = __ffs(mask) - 1;
        int64_t next_a = __shfl_sync(0xFFFFFFFF, matched_a, leader);
        a_idx = next_a + 1;
        b_idx++;
      } else {
        vertex_t max_x =
            __shfl_sync(0xFFFFFFFF, A[a0 + a_idx + step - 1], step - 1);
        if (max_x < y) {
          a_idx += step;
        } else {
          b_idx++;
        }
      }
    }
  }

  for (int offset = 16; offset > 0; offset /= 2) {
    local_cnt += __shfl_down_sync(0xFFFFFFFF, local_cnt, offset);
  }
  return local_cnt;
}

template <typename vertex_t>
__global__ void signed_tc_kernel(
    const int64_t *__restrict__ pos_off, const int64_t *__restrict__ neg_off,
    const vertex_t *__restrict__ pos_e, const vertex_t *__restrict__ neg_e,
    const vertex_t *__restrict__ pos_src, const vertex_t *__restrict__ neg_src,
    const int64_t *__restrict__ work_list, int64_t pos_m, int64_t neg_m,
    unsigned long long *__restrict__ d_bal,
    unsigned long long *__restrict__ d_unbal) {
  constexpr int WARPS_PER_BLOCK = 256 / 32;

  __shared__ unsigned long long s_bal[WARPS_PER_BLOCK];
  __shared__ unsigned long long s_unbal[WARPS_PER_BLOCK];

  int lane_id = threadIdx.x & 31;
  int warp_idx_in_block = threadIdx.x / 32;

  if (threadIdx.x < WARPS_PER_BLOCK) {
    s_bal[threadIdx.x] = 0;
    s_unbal[threadIdx.x] = 0;
  }
  __syncthreads();

  unsigned long long local_bal = 0;
  unsigned long long local_unbal = 0;

  int64_t global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int64_t total_warps = (gridDim.x * blockDim.x) / 32;
  int64_t max_edges = (pos_m > neg_m) ? pos_m : neg_m;

  for (int64_t i = global_warp_id; i < max_edges; i += total_warps) {

    int64_t edge_idx = work_list[i];

    if (edge_idx < pos_m) {
      vertex_t u = pos_src[edge_idx];
      vertex_t v = pos_e[edge_idx];

      const int64_t pu0 = pos_off[u];
      const int64_t pu1 = pos_off[u + 1];
      const int64_t nu0 = neg_off[u];
      const int64_t nu1 = neg_off[u + 1];
      const int64_t pv0 = pos_off[v];
      const int64_t pv1 = pos_off[v + 1];
      const int64_t nv0 = neg_off[v];
      const int64_t nv1 = neg_off[v + 1];

      local_bal +=
          intersect_count_warp(pos_e, pu0, pu1, pos_e, pv0, pv1, lane_id);
      local_bal +=
          intersect_count_warp(neg_e, nu0, nu1, neg_e, nv0, nv1, lane_id);
      local_unbal +=
          intersect_count_warp(pos_e, pu0, pu1, neg_e, nv0, nv1, lane_id);
      local_unbal +=
          intersect_count_warp(neg_e, nu0, nu1, pos_e, pv0, pv1, lane_id);
    }

    if (edge_idx < neg_m) {
      vertex_t u = neg_src[edge_idx];
      vertex_t v = neg_e[edge_idx];

      const int64_t pu0 = pos_off[u];
      const int64_t pu1 = pos_off[u + 1];
      const int64_t nu0 = neg_off[u];
      const int64_t nu1 = neg_off[u + 1];
      const int64_t pv0 = pos_off[v];
      const int64_t pv1 = pos_off[v + 1];
      const int64_t nv0 = neg_off[v];
      const int64_t nv1 = neg_off[v + 1];

      local_bal +=
          intersect_count_warp(pos_e, pu0, pu1, neg_e, nv0, nv1, lane_id);
      local_bal +=
          intersect_count_warp(neg_e, nu0, nu1, pos_e, pv0, pv1, lane_id);
      local_unbal +=
          intersect_count_warp(pos_e, pu0, pu1, pos_e, pv0, pv1, lane_id);
      local_unbal +=
          intersect_count_warp(neg_e, nu0, nu1, neg_e, nv0, nv1, lane_id);
    }
  }

  if (lane_id == 0) {
    s_bal[warp_idx_in_block] = local_bal;
    s_unbal[warp_idx_in_block] = local_unbal;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    unsigned long long block_bal = 0;
    unsigned long long block_unbal = 0;

#pragma unroll
    for (int i = 0; i < WARPS_PER_BLOCK; i++) {
      block_bal += s_bal[i];
      block_unbal += s_unbal[i];
    }

    if (block_bal > 0)
      atomicAdd(d_bal, block_bal);
    if (block_unbal > 0)
      atomicAdd(d_unbal, block_unbal);
  }
}

template <typename graph_t, typename param_type, typename result_type>
struct problem_t : gunrock::problem_t<graph_t> {
  param_type param;
  result_type result;

  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;

  thrust::device_vector<vertex_t> rank_;

  thrust::device_vector<int64_t> pos_offsets;
  thrust::device_vector<vertex_t> pos_edges;
  thrust::device_vector<int64_t> neg_offsets;
  thrust::device_vector<vertex_t> neg_edges;

  thrust::device_vector<vertex_t> pos_src;
  thrust::device_vector<vertex_t> neg_src;

  thrust::device_vector<int64_t> work_list;

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

    mem_instrument("Start Init");

    {
      thrust::device_vector<vertex_t> total_deg(n, vertex_t{0});
      auto *tdeg_ptr = total_deg.data().get();
      thrust::for_each(thrust::device,
                       thrust::make_counting_iterator<edge_t>(0),
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
    }
    mem_instrument("Ranks Allocated & Scoped Out");

    auto *rank_stable = rank_.data().get();

    pos_offsets.assign(n + 1, int64_t{0});
    neg_offsets.assign(n + 1, int64_t{0});
    auto *pos_off_ptr = pos_offsets.data().get();
    auto *neg_off_ptr = neg_offsets.data().get();

    thrust::for_each(
        thrust::device, thrust::make_counting_iterator<edge_t>(0),
        thrust::make_counting_iterator<edge_t>(m),
        [G, rank_stable, pos_off_ptr, neg_off_ptr] __device__(edge_t eid) {
          vertex_t u = G.get_source_vertex(eid);
          vertex_t v = G.get_destination_vertex(eid);
          if (u == v)
            return;
          if (rank_stable[u] >= rank_stable[v])
            return;
          if (G.get_edge_weight(eid) > 0.0f)
            math::atomic::add(
                reinterpret_cast<unsigned long long *>(pos_off_ptr + u), 1ULL);
          else
            math::atomic::add(
                reinterpret_cast<unsigned long long *>(neg_off_ptr + u), 1ULL);
        });

    thrust::exclusive_scan(thrust::device, pos_offsets.begin(),
                           pos_offsets.end(), pos_offsets.begin());
    thrust::exclusive_scan(thrust::device, neg_offsets.begin(),
                           neg_offsets.end(), neg_offsets.begin());

    cudaDeviceSynchronize();

    int64_t pos_m = pos_offsets[n];
    int64_t neg_m = neg_offsets[n];

    pos_edges.resize(pos_m);
    neg_edges.resize(neg_m);
    pos_src.resize(pos_m);
    neg_src.resize(neg_m);
    mem_instrument("Split CSR Structures Allocated");

    {
      thrust::device_vector<int64_t> pos_cur = pos_offsets;
      thrust::device_vector<int64_t> neg_cur = neg_offsets;
      auto *pos_cur_ptr = pos_cur.data().get();
      auto *neg_cur_ptr = neg_cur.data().get();
      auto *pos_edge_ptr = pos_edges.data().get();
      auto *neg_edge_ptr = neg_edges.data().get();
      auto *pos_src_ptr = pos_src.data().get();
      auto *neg_src_ptr = neg_src.data().get();

      thrust::for_each(
          thrust::device, thrust::make_counting_iterator<edge_t>(0),
          thrust::make_counting_iterator<edge_t>(m),
          [G, rank_stable, pos_cur_ptr, neg_cur_ptr, pos_edge_ptr, neg_edge_ptr,
           pos_src_ptr, neg_src_ptr] __device__(edge_t eid) {
            vertex_t u = G.get_source_vertex(eid);
            vertex_t v = G.get_destination_vertex(eid);
            if (u == v)
              return;
            if (rank_stable[u] >= rank_stable[v])
              return;
            if (G.get_edge_weight(eid) > 0.0f) {
              int64_t idx = math::atomic::add(
                  reinterpret_cast<unsigned long long *>(&pos_cur_ptr[u]),
                  1ULL);
              pos_edge_ptr[idx] = v;
              pos_src_ptr[idx] = u;
            } else {
              int64_t idx = math::atomic::add(
                  reinterpret_cast<unsigned long long *>(&neg_cur_ptr[u]),
                  1ULL);
              neg_edge_ptr[idx] = v;
              neg_src_ptr[idx] = u;
            }
          });

      pos_cur.clear();
      pos_cur.shrink_to_fit();
      neg_cur.clear();
      neg_cur.shrink_to_fit();
    }
    mem_instrument("Scatter Complete and Cursors Freed");

    cudaDeviceSynchronize();

    if (pos_m > 0) {
      auto pos_keys = thrust::make_zip_iterator(
          thrust::make_tuple(pos_src.begin(), pos_edges.begin()));
      thrust::sort(thrust::device, pos_keys, pos_keys + pos_m);
    }

    if (neg_m > 0) {
      auto neg_keys = thrust::make_zip_iterator(
          thrust::make_tuple(neg_src.begin(), neg_edges.begin()));
      thrust::sort(thrust::device, neg_keys, neg_keys + neg_m);
    }

    cudaDeviceSynchronize();

    int64_t max_edges = (pos_m > neg_m) ? pos_m : neg_m;
    work_list.resize(max_edges);
    thrust::sequence(thrust::device, work_list.begin(), work_list.end());

    thrust::device_vector<int64_t> edge_costs(max_edges, 0);
    auto *cost_ptr = edge_costs.data().get();
    auto *p_off = pos_offsets.data().get();
    auto *n_off = neg_offsets.data().get();
    auto *p_src = pos_src.data().get();
    auto *n_src = neg_src.data().get();
    auto *p_dst = pos_edges.data().get();
    auto *n_dst = neg_edges.data().get();

    thrust::for_each(thrust::device, thrust::make_counting_iterator<int64_t>(0),
                     thrust::make_counting_iterator<int64_t>(max_edges),
                     [=] __device__(int64_t idx) {
                       int64_t cost = 0;
                       if (idx < pos_m) {
                         vertex_t u = p_src[idx];
                         vertex_t v = p_dst[idx];
                         int64_t deg_u = (p_off[u + 1] - p_off[u]) +
                                         (n_off[u + 1] - n_off[u]);
                         int64_t deg_v = (p_off[v + 1] - p_off[v]) +
                                         (n_off[v + 1] - n_off[v]);
                         cost += (deg_u < deg_v) ? deg_u : deg_v;
                       }
                       if (idx < neg_m) {
                         vertex_t u = n_src[idx];
                         vertex_t v = n_dst[idx];
                         int64_t deg_u = (p_off[u + 1] - p_off[u]) +
                                         (n_off[u + 1] - n_off[u]);
                         int64_t deg_v = (p_off[v + 1] - p_off[v]) +
                                         (n_off[v + 1] - n_off[v]);
                         cost += (deg_u < deg_v) ? deg_u : deg_v;
                       }
                       cost_ptr[idx] = cost;
                     });

    thrust::sort_by_key(thrust::device, edge_costs.begin(), edge_costs.end(),
                        work_list.begin(), thrust::greater<int64_t>());
    edge_costs.clear();
    edge_costs.shrink_to_fit();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::cerr << "CUDA ERROR AFTER INITIALIZATION: "
                << cudaGetErrorString(err) << std::endl;
    }

    mem_instrument("Dynamic Scheduling Worklist Created");

    cudaMalloc(&d_balanced, sizeof(unsigned long long));
    cudaMalloc(&d_unbalanced, sizeof(unsigned long long));
    cudaMemset(d_balanced, 0, sizeof(unsigned long long));
    cudaMemset(d_unbalanced, 0, sizeof(unsigned long long));
    mem_instrument("Counters Set Up");
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

template <typename problem_t> struct enactor_t : gunrock::enactor_t<problem_t> {
  using vertex_t = typename problem_t::vertex_t;
  using edge_t = typename problem_t::edge_t;
  using weight_t = typename problem_t::weight_t;

  enactor_t(problem_t *_problem,
            std::shared_ptr<gcuda::multi_context_t> _context,
            enactor_properties_t _properties = enactor_properties_t{})
      : gunrock::enactor_t<problem_t>(_problem, _context, _properties) {}

  void loop(gcuda::multi_context_t &context) override {
    auto P = this->get_problem();

    auto *pos_off = P->pos_offsets.data().get();
    auto *neg_off = P->neg_offsets.data().get();
    auto *pos_e = P->pos_edges.data().get();
    auto *neg_e = P->neg_edges.data().get();
    auto *pos_src = P->pos_src.data().get();
    auto *neg_src = P->neg_src.data().get();
    auto *w_list = P->work_list.data().get();
    auto *d_bal = P->d_balanced;
    auto *d_unbal = P->d_unbalanced;

    int64_t pos_m = static_cast<int64_t>(P->pos_edges.size());
    int64_t neg_m = static_cast<int64_t>(P->neg_edges.size());

    int threadsPerBlock = 256;
    int blocksPerGrid = 4096;

    signed_tc_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        pos_off, neg_off, pos_e, neg_e, pos_src, neg_src, w_list, pos_m, neg_m,
        d_bal, d_unbal);

    context.get_context(0)->synchronize();

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

} // namespace signed_tc
} // namespace gunrock
