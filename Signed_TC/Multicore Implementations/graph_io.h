#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct EdgeList {
  uint32_t n;
  uint32_t m;
  uint32_t *src;
  uint32_t *dst;
  int8_t *sign;
};

struct ForwardCSR {
  uint32_t n;
  uint64_t m;
  uint64_t *offsets;
  uint32_t *edges;
  int8_t *signs;
  uint32_t *fdeg;
  uint32_t *rank_;
};

namespace _io {

struct MMapFile {
  const char *data;
  uint64_t size;
  int fd;

  explicit MMapFile(const char *path) {
    fd = open(path, O_RDONLY);
    if (fd < 0) {
      perror(path);
      exit(1);
    }
    struct stat st;
    if (fstat(fd, &st) < 0) {
      perror("fstat");
      exit(1);
    }
    size = (uint64_t)st.st_size;
    if (size == 0) {
      data = nullptr;
      close(fd);
      return;
    }
    data = (const char *)mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
      perror("mmap");
      exit(1);
    }
    madvise((void *)data, size, MADV_SEQUENTIAL);
    madvise((void *)data, size, MADV_WILLNEED);
  }
  ~MMapFile() {
    if (data && size > 0) {
      munmap((void *)data, size);
      close(fd);
    }
  }
};

static inline const char *skip_whitespace(const char *p, const char *end) {
  while (p < end && (*p == ' ' || *p == '\t'))
    p++;
  return p;
}
static inline const char *parse_uint(const char *p, const char *end,
                                     uint32_t &out) {
  out = 0;
  while (p < end && *p >= '0' && *p <= '9')
    out = out * 10 + (uint32_t)(*p++ - '0');
  return p;
}
static inline const char *parse_sign(const char *p, const char *end,
                                     int8_t &out) {
  int8_t s = 1;
  if (p < end && *p == '-') {
    s = -1;
    p++;
  } else if (p < end && *p == '+') {
    p++;
  }
  while (p < end && *p >= '0' && *p <= '9')
    p++;
  out = s;
  return p;
}
} // namespace _io

static inline EdgeList load_graph(const char *path) {
  _io::MMapFile mm(path);
  const char *data = mm.data;
  uint64_t size = mm.size;
  if (size == 0)
    return EdgeList{0, 0, nullptr, nullptr, nullptr};

  int num_threads = omp_get_max_threads();
  uint64_t *thread_edge_counts =
      (uint64_t *)std::malloc(num_threads * sizeof(uint64_t));
  uint64_t *thread_offsets =
      (uint64_t *)std::malloc((num_threads + 1) * sizeof(uint64_t));
  const char **thread_starts =
      (const char **)std::malloc(num_threads * sizeof(const char *));
  const char **thread_ends =
      (const char **)std::malloc(num_threads * sizeof(const char *));

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    uint64_t chunk = size / num_threads;
    const char *l_start = data + tid * chunk;
    const char *l_end =
        (tid == num_threads - 1) ? (data + size) : (data + (tid + 1) * chunk);
    if (tid > 0) {
      while (l_start < data + size && *(l_start - 1) != '\n')
        l_start++;
    }
    if (tid < num_threads - 1) {
      while (l_end < data + size && *(l_end - 1) != '\n')
        l_end++;
    }
    thread_starts[tid] = l_start;
    thread_ends[tid] = l_end;

    uint64_t local_edges = 0;
    const char *p = l_start;
    while (p < l_end) {
      if (*p == '\n' || *p == '\r') {
        p++;
        continue;
      }
      if (*p == '#' || *p == '%') {
        while (p < l_end && *p != '\n')
          p++;
        continue;
      }
      local_edges++;
      while (p < l_end && *p != '\n')
        p++;
    }
    thread_edge_counts[tid] = local_edges;
  }

  thread_offsets[0] = 0;
  for (int i = 0; i < num_threads; i++)
    thread_offsets[i + 1] = thread_offsets[i] + thread_edge_counts[i];
  uint64_t total_edges = thread_offsets[num_threads];

  uint32_t *src = new uint32_t[total_edges];
  uint32_t *dst = new uint32_t[total_edges];
  int8_t *sign = new int8_t[total_edges];
  uint32_t max_id = 0;

#pragma omp parallel reduction(max : max_id)
  {
    int tid = omp_get_thread_num();
    const char *p = thread_starts[tid];
    const char *l_end = thread_ends[tid];
    uint64_t write_idx = thread_offsets[tid];

    while (p < l_end) {
      if (*p == '\n' || *p == '\r') {
        p++;
        continue;
      }
      if (*p == '#' || *p == '%') {
        while (p < l_end && *p != '\n')
          p++;
        continue;
      }
      uint32_t u, v;
      int8_t s;
      p = _io::parse_uint(p, l_end, u);
      p = _io::skip_whitespace(p, l_end);
      p = _io::parse_uint(p, l_end, v);
      p = _io::skip_whitespace(p, l_end);
      p = _io::parse_sign(p, l_end, s);
      while (p < l_end && *p != '\n')
        p++;

      if (u == v)
        continue;
      src[write_idx] = std::min(u, v);
      dst[write_idx] = std::max(u, v);
      sign[write_idx] = s;
      write_idx++;

      if (u > max_id)
        max_id = u;
      if (v > max_id)
        max_id = v;
    }
  }

  std::free(thread_edge_counts);
  std::free(thread_offsets);
  std::free(thread_starts);
  std::free(thread_ends);

  EdgeList el{max_id + 1, (uint32_t)total_edges, src, dst, sign};
  return el;
}

static inline void free_edge_list(EdgeList &el) {
  delete[] el.src;
  delete[] el.dst;
  delete[] el.sign;
  el.n = el.m = 0;
}

struct DedupeEdge {
  uint32_t u, v;
  int8_t s;
  bool operator<(const DedupeEdge &o) const {
    if (u != o.u)
      return u < o.u;
    return v < o.v;
  }
};

static inline ForwardCSR build_forward_csr(const EdgeList &el) {
  const uint32_t n = el.n;
  const uint32_t m = el.m;

  DedupeEdge *temp_arr = new DedupeEdge[m];
#pragma omp parallel for
  for (uint32_t i = 0; i < m; i++) {
    temp_arr[i] = DedupeEdge{el.src[i], el.dst[i], el.sign[i]};
  }

#pragma omp parallel
  {
#pragma omp single nowait
    std::sort(temp_arr, temp_arr + m);
  }

  uint32_t unique_m = 0;
  if (m > 0) {
    uint32_t write_pos = 0;
    for (uint32_t i = 1; i < m; i++) {
      if (temp_arr[i].u == temp_arr[write_pos].u &&
          temp_arr[i].v == temp_arr[write_pos].v) {
        continue;
      }
      write_pos++;
      temp_arr[write_pos] = temp_arr[i];
    }
    unique_m = write_pos + 1;
  }

  uint32_t *degree = new uint32_t[n]();
#pragma omp parallel for
  for (uint32_t i = 0; i < unique_m; i++) {
#pragma omp atomic update
    degree[temp_arr[i].u]++;
#pragma omp atomic update
    degree[temp_arr[i].v]++;
  }

  uint32_t max_deg = 0;
#pragma omp parallel for reduction(max : max_deg)
  for (uint32_t i = 0; i < n; i++) {
    if (degree[i] > max_deg)
      max_deg = degree[i];
  }

  const uint32_t buck_sz = max_deg + 2;
  const int nb_threads = omp_get_max_threads();
  uint32_t *local_buckets = new uint32_t[(uint64_t)nb_threads * buck_sz]();

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    uint32_t *lbuck = local_buckets + (uint64_t)tid * buck_sz;
#pragma omp for schedule(static) nowait
    for (uint32_t i = 0; i < n; i++)
      lbuck[degree[i] + 1]++;
  }

  uint32_t *bucket = new uint32_t[buck_sz]();
#pragma omp parallel for schedule(static)
  for (uint32_t d = 0; d < buck_sz; d++) {
    uint32_t sum = 0;
    for (int t = 0; t < nb_threads; t++)
      sum += local_buckets[(uint64_t)t * buck_sz + d];
    bucket[d] = sum;
  }
  delete[] local_buckets;

  for (uint32_t d = 1; d < buck_sz; d++)
    bucket[d] += bucket[d - 1];

  uint32_t *verts = new uint32_t[n];

  uint32_t *cursor = new uint32_t[buck_sz];
  std::memcpy(cursor, bucket, buck_sz * sizeof(uint32_t));

  for (uint32_t i = 0; i < n; i++)
    verts[cursor[degree[i]]++] = i;
  delete[] bucket;
  delete[] cursor;

  uint32_t *rank_ = new uint32_t[n];
#pragma omp parallel for
  for (uint32_t i = 0; i < n; i++)
    rank_[verts[i]] = i;
  delete[] verts;

  uint32_t *fdeg = new uint32_t[n]();
#pragma omp parallel for
  for (uint32_t i = 0; i < unique_m; i++) {
    uint32_t u = temp_arr[i].u, v = temp_arr[i].v;
    if (rank_[u] < rank_[v]) {
#pragma omp atomic update
      fdeg[u]++;
    } else {
#pragma omp atomic update
      fdeg[v]++;
    }
  }

  uint64_t *offsets = new uint64_t[n + 1];
  offsets[0] = 0;

  {
    const int np = omp_get_max_threads();
    uint64_t *carries = new uint64_t[np + 1]();

#pragma omp parallel num_threads(np)
    {
      int tid = omp_get_thread_num();
      uint32_t lo = (uint32_t)((uint64_t)tid * n / np);
      uint32_t hi = (uint32_t)((uint64_t)(tid + 1) * n / np);
      uint64_t local_sum = 0;
      for (uint32_t v = lo; v < hi; v++) {
        local_sum += fdeg[v];
        offsets[v + 1] = local_sum;
      }
      carries[tid + 1] = local_sum;
    }

    for (int t = 1; t <= np; t++)
      carries[t] += carries[t - 1];

#pragma omp parallel num_threads(np)
    {
      int tid = omp_get_thread_num();
      if (tid > 0) {
        uint32_t lo = (uint32_t)((uint64_t)tid * n / np);
        uint32_t hi = (uint32_t)((uint64_t)(tid + 1) * n / np);
        uint64_t carry = carries[tid];
        for (uint32_t v = lo; v < hi; v++)
          offsets[v + 1] += carry;
      }
    }
    delete[] carries;
  }

  const uint64_t total = offsets[n];
  uint32_t *edges = new uint32_t[total];
  int8_t *signs = new int8_t[total];

  uint64_t *csr_cursor = new uint64_t[n];
#pragma omp parallel for schedule(static)
  for (uint32_t v = 0; v < n; v++)
    csr_cursor[v] = offsets[v];

#pragma omp parallel for
  for (uint32_t i = 0; i < unique_m; i++) {
    uint32_t u = temp_arr[i].u, v = temp_arr[i].v;
    if (rank_[u] < rank_[v]) {
      uint64_t pos;
#pragma omp atomic capture
      pos = csr_cursor[u]++;
      edges[pos] = v;
      signs[pos] = temp_arr[i].s;
    } else {
      uint64_t pos;
#pragma omp atomic capture
      pos = csr_cursor[v]++;
      edges[pos] = u;
      signs[pos] = temp_arr[i].s;
    }
  }

  delete[] temp_arr;
  delete[] csr_cursor;
  delete[] degree;

  ForwardCSR csr{n, total, offsets, edges, signs, fdeg, rank_};
  return csr;
}

static inline void free_forward_csr(ForwardCSR &csr) {
  delete[] csr.offsets;
  delete[] csr.edges;
  delete[] csr.signs;
  delete[] csr.fdeg;
  delete[] csr.rank_;
  csr.n = 0;
  csr.m = 0;
}
