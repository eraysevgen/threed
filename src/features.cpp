#include "features.hpp"
#include "timer.hpp"
#include <atomic>
#include <mutex>
#include <thread>

/**
 * @brief
 *
 * @param host_point_cloud host point cloud, where the local features are computed on
 * @param query_point_cloud query point cloud, in each point there is a row of feature set
 * @param neighborhood_indices neighborhood indices used in having local neighborhoods
 * @param features out feature vector
 * @param add_height add the height features
 * @param add_density  add the density to the end of the feature set
 * @param add_xyz add xyz to the end of the feature set
 * @param num_threads number of threads in parallel processing
 */
void threed::compute_features(const threed::PointCloud&				  host_point_cloud,
							  const threed::PointCloud&				  query_point_cloud,
							  const std::vector<std::vector<size_t>>& neighborhood_indices,
							  std::vector<std::vector<float>>&		  features,
							  bool									  add_height,
							  bool									  add_density,
							  bool									  add_xyz,
							  unsigned int							  num_threads)
{
	std::cout << "-> Computing features ... ";
	auto start_time = std::chrono::steady_clock::now();

	features.resize(neighborhood_indices.size());

	std::atomic<size_t> next_index(0);
	size_t				num_points = neighborhood_indices.size();

	auto process_point = [&](size_t idx) {
		size_t num_neighbors = neighborhood_indices[idx].size();

		if (num_neighbors <= 4) {
			threed::make_the_vector_zeros(features[idx], 8); // Geometric features
			threed::make_the_vector_zeros(features[idx], 3); // Height features
			return;
		}

		pcl::PointCloud<pcl::PointXYZ>::Ptr neighbors(new pcl::PointCloud<pcl::PointXYZ>);
		neighbors->width  = num_neighbors;
		neighbors->height = 1;
		neighbors->resize(num_neighbors);

		std::vector<float> neighbor_heights(num_neighbors);

		for (size_t j = 0; j < num_neighbors; ++j) {
			size_t		neighbor_index = neighborhood_indices[idx][j];
			const auto& neighbor	   = host_point_cloud.data[neighbor_index];
			neighbors->points[j]	   = pcl::PointXYZ(static_cast<float>(neighbor[0]), static_cast<float>(neighbor[1]),
													   static_cast<float>(neighbor[2]));
			neighbor_heights[j]		   = static_cast<float>(neighbor[2]);
		}

		pcl::PCA<pcl::PointXYZ> pca;
		pca.setInputCloud(neighbors);
		Eigen::Vector3f eigenvalues	 = pca.getEigenValues();
		Eigen::Matrix3f eigenvectors = pca.getEigenVectors();

		threed::compute_geometric_features(eigenvalues, eigenvectors, features[idx], "sqrt");

		if (add_height) {
			float query_height = static_cast<float>(host_point_cloud.data[idx][2]);
			threed::compute_height_features(query_height, neighbor_heights, features[idx]);
		}

		if (add_density) {
			features[idx].push_back(static_cast<float>(num_neighbors));
		}

		if (add_xyz) {
			const auto& query_point = host_point_cloud.data[idx];
			features[idx].push_back(static_cast<float>(query_point[0]));
			features[idx].push_back(static_cast<float>(query_point[1]));
			features[idx].push_back(static_cast<float>(query_point[2]));
		}
	};
	double completed_percentage;
	auto   process_computation = [&]() {
		  while (true) {
			  size_t idx = next_index.fetch_add(1);
			  if (idx >= num_points)
				  break;

			  if (((idx % 10000) == 0) || (idx - 1 == num_points)) {
				  completed_percentage = (static_cast<double>(idx) / static_cast<double>(num_points)) * 100.0f;

				  std::cout << "\r-> Computing features ... %" << std::setprecision(0) << std::fixed
							<< completed_percentage;
			  }
			  process_point(idx);
		  }
	};

	std::vector<std::thread> threads;
	for (unsigned int t = 0; t < num_threads; ++t) {
		threads.emplace_back(process_computation);
	}

	for (auto& thread : threads) {
		thread.join();
	}

	auto end_time = std::chrono::steady_clock::now();
	std::cout << " done in "
			  << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0
			  << " secs.\n";
}
/**
 * @brief Compute geometric features
 *
 * @param val eigenvalues
 * @param vec eigenvectors
 * @param features out features
 * @param mode not implemented
 * @details
 * In the paper we are using
 * Sum of eigenvalues
 * Linearity
 * planarity
 * sphericity
 * omnivariance
 * eigenentropy
 * surface variation
 * anisotropy
 * absolute moment (6) -> skip those in current version
 * vertical moment (2) -> skip those in curent version
 * verticality
 * As a total of (9 + 6 + 2 = 17)

 */
void threed::compute_geometric_features(const Eigen::Vector3f& val,
										const Eigen::Matrix3f& vec,
										std::vector<float>&	   features,
										std::string			   mode)
{
	// TODO add mode !

	float			e1			  = val[0];
	float			e2			  = val[1];
	float			e3			  = val[2];
	float			sum_of_eigens = e1 + e2 + e3;
	Eigen::Vector3f ev1			  = vec.col(0);
	Eigen::Vector3f ev2			  = vec.col(1);
	Eigen::Vector3f ev3			  = vec.col(2);

	Eigen::Vector3f z(0.0, 0.0, 1.0);

	float verticality = 1.0 - std::abs(z.dot(ev3));

	float linearity	   = (e1 - e2) / e1;
	float planarity	   = (e2 - e3) / e1;
	float sphericity   = (e3 / e1);
	float omnivariance = std::cbrtf(e1 * e2 * e3);
	float anisotropy   = (e1 - e3) / e1;
	float eigenentropy = -(e1 * std::logf(e1)) + (e2 * std::logf(e2)) + (e3 * std::logf(e3));

	float surface_variation = e3 / (sum_of_eigens);

	features.push_back(sum_of_eigens);
	features.push_back(linearity);
	features.push_back(planarity);
	features.push_back(sphericity);
	features.push_back(omnivariance);
	features.push_back(anisotropy);
	features.push_back(eigenentropy);
	features.push_back(surface_variation);
	features.push_back(verticality);
}

/**
 * @brief Make the all vector zero
 * @param vec vector
 * @param n size of vector
 */
void threed::make_the_vector_zeros(std::vector<float>& vec, const size_t n)
{
	// TODO make this function more efficient
	for (size_t i = 0; i < n; ++i)
		vec.push_back(0.0);
}

/**
 * @brief Compute the height features
 *
 * @param h interested height value
 * @param heights neighborhodo heigts
 * @param features out features
 * @details
 * In the paper
 * Height range        zmax - zmin
 * height above min    z - zmin
 * height below max    zmax - z
 * average height      ave(zn)
 * variance            var(zn)
 * as a total of 5 features
 */

void threed::compute_height_features(const float h, std::vector<float> const& heights, std::vector<float>& features)

{
	size_t size	   = heights.size();
	auto   min_max = std::minmax_element(heights.begin(), heights.end());
	float  sum	   = std::accumulate(heights.begin(), heights.end(), static_cast<float>(0.0));

	float height_range	   = static_cast<float>(*min_max.second - *min_max.first);
	float height_above_min = static_cast<float>(h - *min_max.first);
	float height_below_max = static_cast<float>(*min_max.second - h);
	float mean			   = sum / static_cast<float>(heights.size());

	// the var function is from
	// https://stackoverflow.com/questions/33268513/calculating-standard-deviation-variance-in-c

	auto variance_func = [&mean, &size](float accumulator, const float& val) {
		return accumulator + ((val - mean) * (val - mean) / (size - 1));
	};
	float var = std::accumulate(heights.begin(), heights.end(), 0.0, variance_func);

	features.push_back(height_range);
	features.push_back(height_above_min);
	features.push_back(height_below_max);
	features.push_back(mean);
	features.push_back(var);
}