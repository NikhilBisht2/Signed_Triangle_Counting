/**
 * @file signed_tc_merge_shmem.cu
 * @brief Driver for Shared-Memory Merge-Based Signed Triangle Counting.
 */

#include <chrono>
#include <iostream>
#include <string>

#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/algorithms/signed_tc_merge_shmem.hxx>

using namespace gunrock;
using namespace memory;

using vertex_t = int;
using edge_t   = int;
using weight_t = float;

void test_signed_tc_merge_shmem(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <graph.mtx>\n";
    std::exit(1);
  }

  std::string filename = argv[1];

  // Wall-clock timer matching the CPU baseline's `Timer total_time`:
  // starts before file load, ends after the full pipeline (load + init + kernel).
  auto wall_start = std::chrono::high_resolution_clock::now();

  io::matrix_market_t<vertex_t, edge_t, weight_t> mm;
  using csr_t =
      format::csr_t<memory_space_t::device, vertex_t, edge_t, weight_t>;
  csr_t csr;

  std::cout << "Loading graph and edge signs from: " << filename << "...\n";
  auto [props, coo] = mm.load(filename);
  csr.from_coo(coo);

  gunrock::graph::graph_properties_t graph_props;
  graph_props.directed = true;
  auto G = graph::build<memory_space_t::device>(graph_props, csr);

  std::cout << "Vertices : " << G.get_number_of_vertices() << "\n";
  std::cout << "Edges    : " << G.get_number_of_edges()
            << " (both directions)\n";

  std::size_t total_balanced   = 0;
  std::size_t total_unbalanced = 0;

  auto context = std::make_shared<gcuda::multi_context_t>(0);
  gunrock::signed_tc_merge_shmem::param_t  param;
  gunrock::signed_tc_merge_shmem::result_t result(&total_balanced,
                                                   &total_unbalanced);

  float elapsed =
      gunrock::signed_tc_merge_shmem::run(G, param, result, context);

  auto wall_end = std::chrono::high_resolution_clock::now();
  double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

  std::cout << "\n";
  std::cout << "Balanced   = " << total_balanced                      << "\n";
  std::cout << "Unbalanced = " << total_unbalanced                    << "\n";
  std::cout << "Total      = " << (total_balanced + total_unbalanced) << "\n";
  std::cout << "GPU kernel time (enact only) = " << elapsed  << " ms\n";
  std::cout << "GPU wall time (load+init+kernel, matches CPU total_time) = "
             << wall_ms << " ms\n";
}

int main(int argc, char **argv) {
  test_signed_tc_merge_shmem(argc, argv);
  return 0;
}
