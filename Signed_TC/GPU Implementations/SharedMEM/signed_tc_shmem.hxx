#pragma once

#include <algorithm>
#include <gunrock/algorithms/algorithms.hxx>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <vector>

namespace gunrock {
namespace signed_tc_shmem {

static constexpr int TILE_SZ = 128;

static constexpr int BLOCK_DIM = 128;

static constexpr int MAX_BLOCKS = 65535;

template <typename vertex_t>
__device__ __forceinline__ bool shmem_contains(const vertex_t *__restrict__ sh,
                                               int tile_len, vertex_t key) {
  int lo = 0, hi = tile_len - 1;
  while (lo <= hi) {
    int mid = (lo + hi) >> 1;
    vertex_t v = sh[mid];
    if (v == key)
      return true;
    if (v < key)
      lo = mid + 1;
    else
      hi = mid - 1;
  }
  return false;
}

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

  thrust::device_vector<vertex_t> rank_;

  thrust::device_vector<int64_t> pos_offsets;
  thrust::device_vector<vertex_t> pos_edges;
  thrust::device_vector<int64_t> neg_offsets;
  thrust::device_vector<vertex_t> neg_edges;

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

    cudaDeviceSynchronize();

    thrust::device_vector<vertex_t> rank_vec(n);
    thrust::sequence(thrust::device, rank_vec.begin(), rank_vec.end());
    auto *rank_ptr = rank_vec.data().get();
    thrust::sort(thrust::device, rank_ptr, rank_ptr + n,
                 [tdeg_ptr] __device__(vertex_t a, vertex_t b) {
                   if (tdeg_ptr[a] != tdeg_ptr[b])
                     return tdeg_ptr[a] < tdeg_ptr[b];
                   return a < b;
                 });
    cudaDeviceSynchronize();

    rank_.resize(n);
    auto *rinv_ptr = rank_.data().get();
    thrust::for_each(thrust::device,
                     thrust::make_counting_iterator<vertex_t>(0),
                     thrust::make_counting_iterator<vertex_t>(n),
                     [rank_ptr, rinv_ptr] __device__(vertex_t i) {
                       rinv_ptr[rank_ptr[i]] = i;
                     });
    cudaDeviceSynchronize();

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
          auto w = G.get_edge_weight(eid);
          if (w > 0.0f)
            math::atomic::add(
                reinterpret_cast<unsigned long long *>(pos_off_ptr + u + 1),
                1ULL);
          else
            math::atomic::add(
                reinterpret_cast<unsigned long long *>(neg_off_ptr + u + 1),
                1ULL);
        });
    cudaDeviceSynchronize();

    thrust::inclusive_scan(thrust::device, pos_offsets.begin(),
                           pos_offsets.end(), pos_offsets.begin());
    thrust::inclusive_scan(thrust::device, neg_offsets.begin(),
                           neg_offsets.end(), neg_offsets.begin());

    cudaDeviceSynchronize();
    int64_t pos_m = pos_offsets.back();
    int64_t neg_m = neg_offsets.back();
    pos_edges.resize(pos_m);
    neg_edges.resize(neg_m);

    thrust::device_vector<int64_t> pos_cur = pos_offsets;
    thrust::device_vector<int64_t> neg_cur = neg_offsets;
    auto *pos_cur_ptr = pos_cur.data().get();
    auto *neg_cur_ptr = neg_cur.data().get();
    auto *pos_edge_ptr = pos_edges.data().get();
    auto *neg_edge_ptr = neg_edges.data().get();

    thrust::for_each(
        thrust::device, thrust::make_counting_iterator<edge_t>(0),
        thrust::make_counting_iterator<edge_t>(m),
        [G, rank_stable, pos_cur_ptr, neg_cur_ptr, pos_edge_ptr,
         neg_edge_ptr] __device__(edge_t eid) {
          vertex_t u = G.get_source_vertex(eid);
          vertex_t v = G.get_destination_vertex(eid);
          if (u == v)
            return;
          if (rank_stable[u] >= rank_stable[v])
            return;
          auto w = G.get_edge_weight(eid);
          if (w > 0.0f) {
            int64_t idx = math::atomic::add(
                reinterpret_cast<unsigned long long *>(&pos_cur_ptr[u]), 1ULL);
            pos_edge_ptr[idx] = v;
          } else {
            int64_t idx = math::atomic::add(
                reinterpret_cast<unsigned long long *>(&neg_cur_ptr[u]), 1ULL);
            neg_edge_ptr[idx] = v;
          }
        });
    cudaDeviceSynchronize();

    {
      std::vector<vertex_t> h_pos(pos_m), h_neg(neg_m);
      thrust::copy(pos_edges.begin(), pos_edges.end(), h_pos.begin());
      thrust::copy(neg_edges.begin(), neg_edges.end(), h_neg.begin());

      std::vector<int64_t> h_pos_off(n + 1), h_neg_off(n + 1);
      thrust::copy(pos_offsets.begin(), pos_offsets.end(), h_pos_off.begin());
      thrust::copy(neg_offsets.begin(), neg_offsets.end(), h_neg_off.begin());

      for (vertex_t u = 0; u < n; ++u) {
        std::sort(h_pos.data() + h_pos_off[u], h_pos.data() + h_pos_off[u + 1]);
        std::sort(h_neg.data() + h_neg_off[u], h_neg.data() + h_neg_off[u + 1]);
      }

      thrust::copy(h_pos.begin(), h_pos.end(), pos_edges.begin());
      thrust::copy(h_neg.begin(), h_neg.end(), neg_edges.begin());
      cudaDeviceSynchronize();
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

/**
 * Grid-stride: each block processes vertices u = blockIdx.x, blockIdx.x +
 * gridDim.x, ... so grid can be << n while all vertices are covered.
 *
 * [F2] Shared-memory layout (all in the single extern __shared__ region):
 *   sh_pos   [TILE_SZ]   tile of u's positive forward neighbors
 *   sh_neg   [TILE_SZ]   tile of u's negative forward neighbors
 *   blk_bal  [1]         block-level balanced accumulator   (ull)
 *   blk_unbal[1]         block-level unbalanced accumulator (ull)
 *
 * Accumulators are reset at the start of each vertex (not just once per block
 * lifetime) because one block now handles multiple vertices.
 */
__global__ void tc_shmem_intersect_kernel(
    int n, const int64_t *__restrict__ pos_off, const int *__restrict__ pos_e,
    const int64_t *__restrict__ neg_off, const int *__restrict__ neg_e,
    unsigned long long *d_bal, unsigned long long *d_unbal) {
  using vertex_t = int;

  extern __shared__ char smem_raw[];
  vertex_t *sh_pos = reinterpret_cast<vertex_t *>(smem_raw);
  vertex_t *sh_neg = sh_pos + TILE_SZ;

  constexpr size_t tile_bytes = 2 * TILE_SZ * sizeof(vertex_t);
  constexpr size_t acc_offset = (tile_bytes + 7) & ~size_t(7);
  unsigned long long *blk_bal =
      reinterpret_cast<unsigned long long *>(smem_raw + acc_offset);
  unsigned long long *blk_unbal = blk_bal + 1;

  for (vertex_t u = static_cast<vertex_t>(blockIdx.x); u < n;
       u += static_cast<vertex_t>(gridDim.x)) {

    if (threadIdx.x == 0) {
      *blk_bal = 0ULL;
      *blk_unbal = 0ULL;
    }
    __syncthreads();

    const int64_t pu0 = pos_off[u], pu1 = pos_off[u + 1];
    const int64_t nu0 = neg_off[u], nu1 = neg_off[u + 1];
    const int64_t pos_len_u = pu1 - pu0;
    const int64_t neg_len_u = nu1 - nu0;

    for (int64_t e = pu0; e < pu1; ++e) {
      vertex_t v = pos_e[e];

      const int64_t pv0 = pos_off[v], pv1 = pos_off[v + 1];
      const int64_t nv0 = neg_off[v], nv1 = neg_off[v + 1];
      const int64_t plen_v = pv1 - pv0;
      const int64_t nlen_v = nv1 - nv0;

      unsigned long long thr_b = 0, thr_ub = 0;

      for (int64_t tile_u = 0; tile_u < pos_len_u; tile_u += TILE_SZ) {
        int tile_len_u = (int)min((int64_t)TILE_SZ, pos_len_u - tile_u);
        for (int i = threadIdx.x; i < tile_len_u; i += blockDim.x)
          sh_pos[i] = pos_e[pu0 + tile_u + i];
        __syncthreads();
        for (int64_t j = threadIdx.x; j < plen_v; j += blockDim.x)
          if (shmem_contains(sh_pos, tile_len_u, pos_e[pv0 + j]))
            thr_b++;
        for (int64_t j = threadIdx.x; j < nlen_v; j += blockDim.x)
          if (shmem_contains(sh_pos, tile_len_u, neg_e[nv0 + j]))
            thr_ub++;
        __syncthreads();
      }

      for (int64_t tile_u = 0; tile_u < neg_len_u; tile_u += TILE_SZ) {
        int tile_len_u = (int)min((int64_t)TILE_SZ, neg_len_u - tile_u);
        for (int i = threadIdx.x; i < tile_len_u; i += blockDim.x)
          sh_neg[i] = neg_e[nu0 + tile_u + i];
        __syncthreads();
        for (int64_t j = threadIdx.x; j < nlen_v; j += blockDim.x)
          if (shmem_contains(sh_neg, tile_len_u, neg_e[nv0 + j]))
            thr_b++;
        for (int64_t j = threadIdx.x; j < plen_v; j += blockDim.x)
          if (shmem_contains(sh_neg, tile_len_u, pos_e[pv0 + j]))
            thr_ub++;
        __syncthreads();
      }

      if (thr_b)
        atomicAdd(blk_bal, thr_b);
      if (thr_ub)
        atomicAdd(blk_unbal, thr_ub);
      __syncthreads();
    }

    for (int64_t e = nu0; e < nu1; ++e) {
      vertex_t v = neg_e[e];

      const int64_t pv0 = pos_off[v], pv1 = pos_off[v + 1];
      const int64_t nv0 = neg_off[v], nv1 = neg_off[v + 1];
      const int64_t plen_v = pv1 - pv0;
      const int64_t nlen_v = nv1 - nv0;

      unsigned long long thr_b = 0, thr_ub = 0;

      for (int64_t tile_u = 0; tile_u < pos_len_u; tile_u += TILE_SZ) {
        int tile_len_u = (int)min((int64_t)TILE_SZ, pos_len_u - tile_u);
        for (int i = threadIdx.x; i < tile_len_u; i += blockDim.x)
          sh_pos[i] = pos_e[pu0 + tile_u + i];
        __syncthreads();
        for (int64_t j = threadIdx.x; j < nlen_v; j += blockDim.x)
          if (shmem_contains(sh_pos, tile_len_u, neg_e[nv0 + j]))
            thr_b++;
        for (int64_t j = threadIdx.x; j < plen_v; j += blockDim.x)
          if (shmem_contains(sh_pos, tile_len_u, pos_e[pv0 + j]))
            thr_ub++;
        __syncthreads();
      }

      for (int64_t tile_u = 0; tile_u < neg_len_u; tile_u += TILE_SZ) {
        int tile_len_u = (int)min((int64_t)TILE_SZ, neg_len_u - tile_u);
        for (int i = threadIdx.x; i < tile_len_u; i += blockDim.x)
          sh_neg[i] = neg_e[nu0 + tile_u + i];
        __syncthreads();
        for (int64_t j = threadIdx.x; j < plen_v; j += blockDim.x)
          if (shmem_contains(sh_neg, tile_len_u, pos_e[pv0 + j]))
            thr_b++;
        for (int64_t j = threadIdx.x; j < nlen_v; j += blockDim.x)
          if (shmem_contains(sh_neg, tile_len_u, neg_e[nv0 + j]))
            thr_ub++;
        __syncthreads();
      }

      if (thr_b)
        atomicAdd(blk_bal, thr_b);
      if (thr_ub)
        atomicAdd(blk_unbal, thr_ub);
      __syncthreads();
    }

    __syncthreads();
    if (threadIdx.x == 0) {
      if (*blk_bal)
        atomicAdd(d_bal, *blk_bal);
      if (*blk_unbal)
        atomicAdd(d_unbal, *blk_unbal);
    }

    __syncthreads();
  }
}

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
    auto G = P->get_graph();

    vertex_t n = G.get_number_of_vertices();

    auto *pos_off = P->pos_offsets.data().get();
    auto *neg_off = P->neg_offsets.data().get();
    auto *pos_e = P->pos_edges.data().get();
    auto *neg_e = P->neg_edges.data().get();
    auto *d_bal = P->d_balanced;
    auto *d_unbal = P->d_unbalanced;

    constexpr size_t tile_bytes = 2 * TILE_SZ * sizeof(vertex_t);
    constexpr size_t acc_offset = (tile_bytes + 7) & ~size_t(7);
    std::size_t smem_bytes = acc_offset + 2 * sizeof(unsigned long long);

    int grid = (int)min((vertex_t)MAX_BLOCKS, n);

    tc_shmem_intersect_kernel<<<grid, BLOCK_DIM, smem_bytes>>>(
        n, pos_off, pos_e, neg_off, neg_e, d_bal, d_unbal);

    context.get_context(0)->synchronize();

    unsigned long long h_bal = 0, h_unbal = 0;
    cudaMemcpy(&h_bal, d_bal, sizeof(unsigned long long),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_unbal, d_unbal, sizeof(unsigned long long),
               cudaMemcpyDeviceToHost);
    *P->result.total_balanced = (std::size_t)h_bal;
    *P->result.total_unbalanced = (std::size_t)h_unbal;
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

} // namespace signed_tc_shmem
} // namespace gunrock
