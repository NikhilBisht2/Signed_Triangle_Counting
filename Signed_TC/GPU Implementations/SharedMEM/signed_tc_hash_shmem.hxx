#pragma once

#include <gunrock/algorithms/algorithms.hxx>
#include <iostream>
#include <limits>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>

namespace gunrock {
namespace signed_tc_hash_shmem {

static constexpr int TILE_SZ = 512;

static constexpr int HT_CAP = 1024;
static constexpr int HT_MASK = HT_CAP - 1;

static constexpr int BLOCK_DIM = 256;

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

__device__ __forceinline__ uint32_t ht_hash32(uint32_t x) {
  x = ((x >> 16) ^ x) * 0x45d9f3b;
  x = ((x >> 16) ^ x) * 0x45d9f3b;
  return (x >> 16) ^ x;
}

/**
 * Each block processes one pivot vertex v.
 *
 * Shared-memory layout:
 * HTEntry<vertex_t,weight_t>  sh_ht[HT_CAP]
 * unsigned long long          blk_bal
 * unsigned long long          blk_unbal
 *
 * For each tile of TILE_SZ entries from v's forward list:
 * 1. Thread 0 clears sh_ht and inserts the tile (serial; deg d TILE_SZ d 512,
 * so d 512 inserts  fast).
 * 2. __syncthreads() makes the HT visible to all threads.
 * 3. For each forward neighbor w of v, all BLOCK_DIM threads stride over
 * w's forward list from global mem, probe sh_ht for each element u.
 * 4. On hit: classify triangle (v,w,u) as balanced/unbalanced from signs.
 *
 * Rank filter (inherited from global-mem version):
 * u must satisfy rank[u] > rank[v] AND rank[u] > rank[w].
 * Since we iterate v's forward list for w (rank[w] > rank[v] by DAG
 * construction), we only need rank[u] > rank[w].
 * The shmem tile holds entries from v's forward list  all have rank > rank[v]
 * already.  So we check rank[u] > rank[w] at probe time.
 *
 * Sign classification:
 * Triangle (v, w, u):
 * sign_vw = sign of edge v�w  (from f_signs when iterating w)
 * sign_vu = sign of edge v�u  (stored in sh_ht[slot].sign)
 * sign_wu = sign of edge w�u  (from f_signs when iterating w's list)
 * Product > 0 � balanced; product < 0 � unbalanced.
 */
__global__ void shmem_hash_kernel(int n, const int64_t *__restrict__ f_off,
                                  const int *__restrict__ f_e,
                                  const float *__restrict__ f_s,
                                  const int *__restrict__ rank,
                                  int empty_sentinel, unsigned long long *d_bal,
                                  unsigned long long *d_unbal) {
  using vertex_t = int;
  using weight_t = float;

  extern __shared__ char smem_raw[];

  struct HTEntry {
    vertex_t key;
    weight_t sign;
  };
  HTEntry *sh_ht = reinterpret_cast<HTEntry *>(smem_raw);
  unsigned long long *sh_acc =
      reinterpret_cast<unsigned long long *>(sh_ht + HT_CAP);

  int tid = threadIdx.x;
  int bdim = blockDim.x;

  for (int v = (int)blockIdx.x; v < n; v += (int)gridDim.x) {
    if (tid == 0) {
      sh_acc[0] = 0ULL;
      sh_acc[1] = 0ULL;
    }
    __syncthreads();

    int64_t bv = f_off[v];
    int64_t ev = f_off[v + 1];
    int dv = static_cast<int>(ev - bv);

    for (int tile_start = 0; tile_start < dv; tile_start += TILE_SZ) {
      int tile_len = min(TILE_SZ, dv - tile_start);

      if (tid == 0) {

        for (int i = 0; i < HT_CAP; ++i)
          sh_ht[i].key = empty_sentinel;

        for (int i = 0; i < tile_len; ++i) {
          vertex_t u_key = f_e[bv + tile_start + i];
          weight_t u_sign = f_s[bv + tile_start + i];
          uint32_t pos = ht_hash32(static_cast<uint32_t>(u_key)) & HT_MASK;
          while (sh_ht[pos].key != empty_sentinel)
            pos = (pos + 1) & HT_MASK;
          sh_ht[pos].key = u_key;
          sh_ht[pos].sign = u_sign;
        }
      }
      __syncthreads();

      for (int64_t wi = bv; wi < ev; ++wi) {
        vertex_t w = f_e[wi];
        weight_t sign_vw = f_s[wi];

        int64_t bw = f_off[w];
        int64_t ew = f_off[w + 1];
        int dw = static_cast<int>(ew - bw);

        for (int j = tid; j < dw; j += bdim) {
          vertex_t u = f_e[bw + j];
          weight_t sign_wu = f_s[bw + j];

          if (rank[u] <= rank[w])
            continue;

          uint32_t pos = ht_hash32(static_cast<uint32_t>(u)) & HT_MASK;
          bool found = false;
          weight_t sign_vu = 0.0f;

          while (sh_ht[pos].key != empty_sentinel) {
            if (sh_ht[pos].key == u) {
              sign_vu = sh_ht[pos].sign;
              found = true;
              break;
            }
            pos = (pos + 1) & HT_MASK;
          }

          if (!found)
            continue;

          float product = sign_vw * sign_wu * sign_vu;
          if (product > 0.0f)
            atomicAdd(&sh_acc[0], 1ULL);
          else
            atomicAdd(&sh_acc[1], 1ULL);
        }
      }
      __syncthreads();
    }

    __syncthreads();
    if (tid == 0) {
      if (sh_acc[0])
        atomicAdd(d_bal, sh_acc[0]);
      if (sh_acc[1])
        atomicAdd(d_unbal, sh_acc[1]);
    }
    __syncthreads();
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
    vertex_t n = P->get_graph().get_number_of_vertices();

    auto *f_off = P->f_offsets.data().get();
    auto *f_e = P->f_edges.data().get();
    auto *f_s = P->f_signs.data().get();
    auto *rank = P->rank_.data().get();
    auto *d_bal = P->d_balanced;
    auto *d_unbal = P->d_unbalanced;

    vertex_t empty_sentinel = std::numeric_limits<vertex_t>::max();

    std::size_t smem_bytes = HT_CAP * (sizeof(vertex_t) + sizeof(weight_t)) +
                             2 * sizeof(unsigned long long);

    int grid = min((int)n, 65535);
    shmem_hash_kernel<<<grid, BLOCK_DIM, smem_bytes>>>(
        n, f_off, f_e, f_s, rank, empty_sentinel, d_bal, d_unbal);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::printf("Launch error = %s\n", cudaGetErrorString(err));
    }

    context.get_context(0)->synchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::printf("Runtime error = %s\n", cudaGetErrorString(err));
    }

    unsigned long long h_bal = 0;
    unsigned long long h_unbal = 0;

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

} // namespace signed_tc_hash_shmem
} // namespace gunrock
