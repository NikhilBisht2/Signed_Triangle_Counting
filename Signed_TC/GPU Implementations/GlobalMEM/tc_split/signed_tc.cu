#include <iostream>
#include <string>

#include <gunrock/algorithms/algorithms.hxx>
#include <gunrock/algorithms/signed_tc.hxx>

using namespace gunrock;
using namespace memory;

using vertex_t = int;
using edge_t = int;
using weight_t = float;

void test_signed_tc(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <graph.mtx>\n";
    std::exit(1);
  }

  std::string filename = argv[1];
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
            << "  (both directions; DAG uses half)\n";

  std::size_t total_balanced = 0;
  std::size_t total_unbalanced = 0;

  auto context = std::make_shared<gcuda::multi_context_t>(0);
  gunrock::signed_tc::param_t param;
  gunrock::signed_tc::result_t result(&total_balanced, &total_unbalanced);

  float elapsed = gunrock::signed_tc::run(G, param, result, context);

  std::cout << "\n";
  std::cout << "Balanced   = " << total_balanced << "\n";
  std::cout << "Unbalanced = " << total_unbalanced << "\n";
  std::cout << "Total      = " << (total_balanced + total_unbalanced) << "\n";
  std::cout << "GPU time   = " << elapsed << " ms\n";
}

int main(int argc, char **argv) {
  test_signed_tc(argc, argv);
  return 0;
}
