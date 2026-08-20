#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "../include/cpu_neighborhood.hpp"
#include "../include/features.hpp"
#include "../include/point_cloud.hpp"

namespace nb = nanobind;
using namespace nb::literals;

nb::ndarray<double, nb::ndim<2>, nb::numpy>
extract_features(nb::ndarray<double, nb::c_contig, nb::device::cpu>					host_pts,
				 nb::ndarray<double, nb::c_contig, nb::device::cpu>					query_pts,
				 std::optional<nb::ndarray<int64_t, nb::c_contig, nb::device::cpu>> edge_index,
				 const std::string&													neighborhood_name,
				 size_t																k,
				 double																radius,
				 bool																add_height,
				 bool																add_density,
				 bool																add_xyz,
				 unsigned int														num_threads)
{

	size_t num_host	   = host_pts.shape(0);
	size_t num_queries = query_pts.shape(0);

	threed::PointCloud host_pc;
	host_pc.data.reserve(num_host);
	host_pc.offsets = { 0.0, 0.0, 0.0 };

	const double* h_data = host_pts.data();
	for (size_t i = 0; i < num_host; ++i) {
		host_pc.data.push_back({ h_data[i * 3], h_data[i * 3 + 1], h_data[i * 3 + 2] });
	}

	threed::PointCloud query_pc;
	query_pc.data.reserve(num_queries);
	query_pc.offsets = { 0.0, 0.0, 0.0 };

	const double* q_data = query_pts.data();
	for (size_t i = 0; i < num_queries; ++i) {
		query_pc.data.push_back({ q_data[i * 3], q_data[i * 3 + 1], q_data[i * 3 + 2] });
	}

	std::vector<std::vector<size_t>> neighborhood_indices(num_queries);

	if (edge_index.has_value()) {
		size_t num_edges = edge_index->shape(1);
		neighborhood_indices.resize(num_queries);

		// C-Contiguous PyTorch edge_index tensor is flat: [Q_0, Q_1... H_0, H_1...]
		const int64_t* query_idx_ptr = edge_index->data();
		const int64_t* host_idx_ptr	 = edge_index->data() + num_edges;

		for (size_t i = 0; i < num_edges; ++i) {
			size_t q_idx = static_cast<size_t>(query_idx_ptr[i]);
			size_t h_idx = static_cast<size_t>(host_idx_ptr[i]);
			neighborhood_indices[q_idx].push_back(h_idx);
		}
	} else {
		bool   is_radius  = (neighborhood_name == "radius");
		size_t max_k	  = 512;
		size_t batch_size = 65536;

		threed::compute_indices_by_nanoflann(host_pc, query_pc, neighborhood_indices, k, radius, is_radius, false,
											 max_k, batch_size, num_threads);
	}

	std::vector<std::vector<double>> features;
	threed::compute_features(host_pc, query_pc, neighborhood_indices, features, add_height, add_density, add_xyz,
							 num_threads);

	size_t	num_features = features.empty() ? 0 : features[0].size();
	double* out_data	 = new double[num_queries * num_features];

	for (size_t i = 0; i < num_queries; ++i) {
		std::memcpy(out_data + (i * num_features), features[i].data(), num_features * sizeof(double));
	}

	nb::capsule cleanup(out_data, [](void* p) noexcept { delete[] (double*)p; });
	return nb::ndarray<double, nb::ndim<2>, nb::numpy>(out_data, { num_queries, num_features }, cleanup);
}

NB_MODULE(threed_py, m)
{
	m.doc() = "threed: Fast 3D point cloud geometric feature extraction";

	m.def("extract_features", &extract_features, "host_pts"_a, "query_pts"_a, "edge_index"_a = nb::none(),
		  "neighborhood_name"_a = "radius", "k"_a = 8, "radius"_a = 0.25, "add_height"_a = false,
		  "add_density"_a = false, "add_xyz"_a = false, "num_threads"_a = 4,
		  "Extract 3D features. Provide edge_index for pre-computed neighborhoods (e.g. from torch-cluster), or omit "
		  "for CPU nanoflann search.");
}