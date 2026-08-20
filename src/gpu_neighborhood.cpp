#include "gpu_neighborhood.hpp"

#include "timer.hpp"

#include <pcl/gpu/octree/octree.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <algorithm>
#include <atomic>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>

/**
 * @brief Compute the neighborhood indices by pcl library -supports only radius on gpu.
 * The parameters are identical to nanoflann version.
 *
 * @param host_point_cloud host point cloud, where the spatial kdtree is build on
 * @param query_point_cloud query point cloud, where each point is queries against host point cloud
 * @param indices each row consists of several neighborhood indices
 * @param k number of neighbors in knn
 * @param radius radius distance
 * @param is_radius determine whether it is radius or knn
 * @param is_sorted determine the indices are sorted or not
 * @param max_k maximum number of neigbors for allocation
 * @param batch_size batch size
 * @param num_threads number of threads in parallel processing
 */
void threed::compute_indices_by_pcl(threed::PointCloud&				  host_point_cloud,
									threed::PointCloud&				  query_point_cloud,
									std::vector<std::vector<size_t>>& indices,
									const size_t					  k,
									const double					  radius,
									const bool						  is_radius,
									const bool						  is_sorted,
									const size_t					  max_k,
									const size_t					  batch_size,
									unsigned int					  num_threads)
{
	pcl::PointCloud<pcl::PointXYZ> pcl_host_point_cloud;
	pcl_host_point_cloud.width	  = host_point_cloud.data.size();
	pcl_host_point_cloud.height	  = 1;
	pcl_host_point_cloud.is_dense = false;
	pcl_host_point_cloud.resize(host_point_cloud.data.size());

	indices.resize(query_point_cloud.data.size());

	for (size_t i = 0; i < host_point_cloud.data.size(); i++) {
		// TODO make an alternative way to initialize
		pcl::PointXYZ p;
		p.x							   = static_cast<float>(host_point_cloud.data[i][0]);
		p.y							   = static_cast<float>(host_point_cloud.data[i][1]);
		p.z							   = static_cast<float>(host_point_cloud.data[i][2]);
		pcl_host_point_cloud.points[i] = p;
	}

	std::cout << "-> Creating octree ... ";
	auto start_time = std::chrono::steady_clock::now();

	pcl::gpu::Octree::PointCloud pcl_host_point_cloud_device;
	pcl_host_point_cloud_device.upload(pcl_host_point_cloud.points);

	pcl::gpu::Octree octree_gpu;
	octree_gpu.setCloud(pcl_host_point_cloud_device);
	octree_gpu.build();

	std::cout << " done in " << static_cast<double>(since(start_time).count()) / 1000.0 << "secs.\n";

	std::cout << "-> Computing indices  ... ";
	start_time = std::chrono::steady_clock::now();

	if (is_radius) {
		pcl::gpu::Octree::Queries  octree_gpu_quaries;
		std::vector<pcl::PointXYZ> pcl_query_point_cloud;
		pcl_query_point_cloud.reserve(batch_size);

		size_t points_read		   = 0;
		size_t num_points_in_query = query_point_cloud.data.size();
		double completed_percentage;
		while (points_read < num_points_in_query) {

			size_t points_to_read = std::min(batch_size, num_points_in_query - points_read);

			completed_percentage
				= (static_cast<double>(points_read) / static_cast<double>(num_points_in_query)) * 100.0f;

			std::cout << "\r-> Computing indices  ... %" << std::setprecision(0) << std::fixed << completed_percentage;

			for (size_t i = points_read; i < points_read + points_to_read; ++i) {
				pcl_query_point_cloud.emplace_back(pcl::PointXYZ({ static_cast<float>(query_point_cloud.data[i][0]),
																   static_cast<float>(query_point_cloud.data[i][1]),
																   static_cast<float>(query_point_cloud.data[i][2]) })

				);
			}

			octree_gpu_quaries.upload(pcl_query_point_cloud);

			pcl::gpu::NeighborIndices result_gpu(octree_gpu_quaries.size(), max_k);

			octree_gpu.radiusSearch(octree_gpu_quaries, radius, max_k, result_gpu);

			std::vector<int> sizes, data;
			result_gpu.sizes.download(sizes);
			result_gpu.data.download(data);

			threed::parallel_add_indices(sizes, data, indices, points_read, max_k, num_threads);
			points_read += points_to_read;
			octree_gpu_quaries.release();
			pcl_query_point_cloud.clear();
		}
		completed_percentage = (static_cast<double>(points_read) / static_cast<double>(num_points_in_query)) * 100.0f;
		std::cout << "\r-> Computing indices  ... %" << std::setprecision(0) << std::fixed << completed_percentage;
	} else {
		throw std::runtime_error(std::string("Knn on gpu not supported, use knn on cpu instead"));
	}

	std::cout << " done in " << static_cast<double>(since(start_time).count()) / 1000.0 << "secs.\n";
}

/**
 * @brief Copy gpu indices to cpu
 *
 * @param sizes number of neighbors
 * @param data the flattened indices on the gpu
 * @param indices cpu indices
 * @param points_read points read up to now
 * @param max_k maximum number of points in radius search
 * @param num_threads number of threads in parallel processing
 */
void threed::parallel_add_indices(const std::vector<int>&			sizes,
								  const std::vector<int>&			data,
								  std::vector<std::vector<size_t>>& indices,
								  size_t							points_read,
								  size_t							max_k,
								  unsigned int						num_threads)
{
	std::atomic<size_t> next_index(0); // Atomic counter for task distribution
	size_t				num_points = sizes.size();
	const size_t		chunk_size = 512;

	auto process_task = [&]() {
		while (true) {
			size_t start_idx = next_index.fetch_add(chunk_size);
			if (start_idx >= num_points)
				break;

			size_t end_idx = std::min(start_idx + chunk_size, num_points);

			for (size_t idx = start_idx; idx < end_idx; ++idx) {
				size_t current_pts_idx = idx + points_read;
				int	   neighbor_size   = sizes[idx];

				indices[current_pts_idx].reserve(neighbor_size);

				for (size_t m = 0; m < neighbor_size; ++m) {
					indices[current_pts_idx].push_back(data[m + idx * max_k]);
				}
			}
		}
	};

	std::vector<std::thread> threads;
	for (int t = 0; t < num_threads; ++t) {
		threads.emplace_back(process_task);
	}

	for (auto& thread : threads) {
		thread.join();
	}
}
