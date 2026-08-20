#include "cpu_neighborhood.hpp"

#include "nanoflann.hpp"
#include "nflann_adaptor.hpp"
#include "timer.hpp"

#include <algorithm>
#include <atomic>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
/**
 * @brief Compute the neighborhood indices by nanoflann library -supports both knn and radius.
 *
 * @param host_point_cloud host point cloud, where the spatial kdtree is build on
 * @param query_point_cloud query point cloud, where each point is queries against host point cloud
 * @param indices each row consists of several neighborhood indices
 * @param k number of neighbors in knn
 * @param radius radius distance
 * @param is_radius determine whether it is radius or knn
 * @param is_sorted determine the indices are sorted or not
 * @param max_k maximum number of neigbors for allocation
 * @param batch_size batch size, not used in this function
 * @param num_threads number of threads in parallel processing
 */
void threed::compute_indices_by_nanoflann(const threed::PointCloud&			host_point_cloud,
										  const threed::PointCloud&			query_point_cloud,
										  std::vector<std::vector<size_t>>& indices,
										  const size_t						k,
										  const double						radius,
										  const bool						is_radius,
										  const bool						is_sorted,
										  const size_t						max_k,
										  const size_t						batch_size,
										  unsigned int						num_threads)
{
	// TODO make this search function a better way with templates
	// See
	// https://github.com/jlblancoc/nanoflann/blob/master/examples/example_with_cmake/pointcloud_example.cpp

	if (query_point_cloud.data.empty() || host_point_cloud.data.empty()) {
		throw std::runtime_error("Point cloud data cannot be empty.");
	}

	unsigned int max_num_threads = std::thread::hardware_concurrency();
	num_threads					 = std::min(std::max(1U, num_threads), max_num_threads);

	std::cout << "-> Creating kdtree   ... ";
	auto start_time = std::chrono::steady_clock::now();

	threed::NFlannPointCloudAdaptor adaptor(host_point_cloud);

	threed::KDTreeAdaptor kd_tree(
		3, adaptor,
		nanoflann::KDTreeSingleIndexAdaptorParams(10, nanoflann::KDTreeSingleIndexAdaptorFlags::None, num_threads));

	kd_tree.buildIndex();

	std::cout << " done in " << static_cast<double>(since(start_time).count()) / 1000.0 << "secs.\n";

	std::cout << "-> Computing indices  ... ";
	start_time = std::chrono::steady_clock::now();

	double						radius2 = radius * radius;
	nanoflann::SearchParameters params(0.0, is_sorted);

	std::atomic<size_t> next_index(0);
	size_t				num_points = query_point_cloud.data.size();
	indices.resize(num_points);

	const size_t chunk_size	   = 512;
	auto		 process_query = [&]() {
		std::vector<size_t>									 ret_index(k);
		std::vector<double>									 out_dist_sqr(k);
		std::vector<nanoflann::ResultItem<uint32_t, double>> radius_matches(max_k);

		while (true) {
			size_t start_idx = next_index.fetch_add(chunk_size);

			if (start_idx >= num_points)
				break;
			size_t end_idx = std::min(start_idx + chunk_size, num_points);

			if ((start_idx % 10000) < chunk_size || end_idx == num_points) {
				double completed_percentage
					= (static_cast<double>(start_idx) / static_cast<double>(num_points)) * 100.0f;
				std::cout << "\r-> Computing indices  ... %" << std::setprecision(0) << std::fixed
						  << completed_percentage << std::flush;
			}
			for (size_t idx = start_idx; idx < end_idx; ++idx) {
				const auto&			query_point = query_point_cloud.data[idx];
				std::vector<size_t> pts_indices;

				if (is_radius) {
					const size_t num_matches
						= kd_tree.radiusSearch(query_point.data(), radius2, radius_matches, params);
					pts_indices.reserve(num_matches);

					for (size_t m = 0; m < num_matches; ++m) {
						pts_indices.push_back(static_cast<size_t>(radius_matches[m].first));
					}
				} else {
					nanoflann::KNNResultSet<double, size_t> resultSet(k);
					resultSet.init(&ret_index[0], &out_dist_sqr[0]);
					kd_tree.findNeighbors(resultSet, query_point.data(), params);

					pts_indices.reserve(k);
					for (size_t ik = 0; ik < k; ik++) {
						pts_indices.push_back(static_cast<size_t>(ret_index[ik]));
					}
				}
				indices[idx] = pts_indices;
			}
		}
	};
	std::vector<std::thread> threads;
	threads.reserve(num_threads);

	for (unsigned int t = 0; t < num_threads; ++t) {
		threads.emplace_back(process_query);
	}

	for (auto& thread : threads) {
		thread.join();
	}

	std::cout << " done in " << static_cast<double>(since(start_time).count()) / 1000.0 << "secs.\n";
}
