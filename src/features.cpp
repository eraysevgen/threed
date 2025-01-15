#include "features.hpp"
#include "timer.hpp"
#include <atomic>
#include <cmath>
#include <mutex>
#include <thread>
/**
 * @brief compute covariance matrix from a set of 3D point clouds
 * @param points
 * @return cov matrix in Eigen::Matrix3d
 */
Eigen::Matrix3d compute_covariance_matrix(const std::vector<std::array<double, 3>>& points)
{
	size_t			num_points = points.size();
	Eigen::MatrixXd data(num_points, 3);
	for (size_t i = 0; i < num_points; ++i) {
		for (size_t j = 0; j < 3; ++j) {
			data(i, j) = points[i][j];
		}
	}

	Eigen::RowVector3d mean	  = data.colwise().mean();
	Eigen::MatrixXd	   center = data.rowwise() - mean;

	Eigen::Matrix3d cov = (center.transpose() * center) / static_cast<double>(num_points - 1);

	return cov;
}
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
							  std::vector<std::vector<double>>&		  features,
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

		if (num_neighbors < 3) {

			threed::make_the_vector_nan(features[idx], NUM_GEOMETRIC_FEATURES);

			if (add_height)

				threed::make_the_vector_nan(features[idx], NUM_HEIGHT_FEATURES);
			if (add_density) {
				features[idx].push_back(static_cast<float>(num_neighbors));
			}

			if (add_xyz) {
				const auto& query_point = query_point_cloud.data[idx];
				features[idx].push_back(query_point[0]);
				features[idx].push_back(query_point[1]);
				features[idx].push_back(query_point[2]);
			}
			return;
		}
		std::vector<std::array<double, 3>> neighbors;
		neighbors.reserve(num_neighbors);

		for (size_t j = 0; j < num_neighbors; ++j) {
			size_t		neighbor_index = neighborhood_indices[idx][j];
			const auto& neighbor	   = host_point_cloud.data[neighbor_index];

			neighbors.emplace_back(std::array<double, 3>({ neighbor[0], neighbor[1], neighbor[2] }));
		}
		Eigen::Matrix3d cov = compute_covariance_matrix(neighbors);

		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
		if (solver.info() != Eigen::Success) {
			throw std::runtime_error("Eigenvalue decomposition failed.");
		}
		Eigen::Vector3d eigenvalues	 = solver.eigenvalues();
		Eigen::Matrix3d eigenvectors = solver.eigenvectors();

		threed::compute_geometric_features(eigenvalues, eigenvectors, features[idx], "sqrt");

		if (add_height) {
			double query_height = host_point_cloud.data[idx][2];
			threed::compute_height_features(query_height, neighbors, features[idx]);
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
void threed::compute_geometric_features(const Eigen::Vector3d& val,
										const Eigen::Matrix3d& vec,
										std::vector<double>&   features,
										std::string			   mode)
{
	// TODO add mode !

	double			e1			  = val[2];
	double			e2			  = val[1];
	double			e3			  = val[0];
	double			sum_of_eigens = e1 + e2 + e3;
	Eigen::Vector3d ev1			  = vec.col(2);
	Eigen::Vector3d ev2			  = vec.col(1);
	Eigen::Vector3d ev3			  = vec.col(0);

	Eigen::Vector3d z(0.0, 0.0, 1.0);

	double verticality = 1.0 - std::abs(z.dot(ev3));

	double linearity	= (e1 - e2) / e1;
	double planarity	= (e2 - e3) / e1;
	double sphericity	= (e3 / e1);
	double omnivariance = std::cbrt(e1 * e2 * e3);
	double anisotropy	= (e1 - e3) / e1;
	double eigenentropy = -(e1 * std::log(e1)) + (e2 * std::log(e2)) + (e3 * std::log(e3));

	double surface_variation = e3 / (sum_of_eigens);

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
void threed::make_the_vector_zeros(std::vector<double>& vec, const size_t n)
{
	// TODO make this function more efficient
	for (size_t i = 0; i < n; ++i)
		vec.push_back(0.0);
}

void threed::make_the_vector_nan(std::vector<double>& vec, const size_t n)
{
	for (size_t i = 0; i < n; ++i)
		vec.push_back(std::numeric_limits<double>::quiet_NaN());
}
/**
 * @brief Compute the height features
 *
 * @param query_h interested height value
 * @param neighbors neighborhods
 * @param features out features
 * @details
 * In the paper
 * Height range        zmax - zmin
 * height above min    z - zmin
 * height below max    zmax - z
 * average height      ave(zn)
 * variance            var(zn)
 * as a total of 5 features
 *
 * @see https://stackoverflow.com/questions/33268513/calculating-standard-deviation-variance-in-c
 * for variance function
 */

void threed::compute_height_features(const double							   query_h,
									 std::vector<std::array<double, 3>> const& neighbors,
									 std::vector<double>&					   features)

{
	size_t				num_neighbors = neighbors.size();
	std::vector<double> heights;
	for (const auto& point : neighbors)
		heights.emplace_back(point[2]);
	// size_t size	   = heights.size();
	auto   min_max = std::minmax_element(heights.begin(), heights.end());
	double sum	   = std::accumulate(heights.begin(), heights.end(), static_cast<double>(0.0));

	double height_range		= *min_max.second - *min_max.first;
	double height_above_min = query_h - *min_max.first;
	double height_below_max = *min_max.second - query_h;
	double mean				= sum / static_cast<double>(num_neighbors);

	auto variance_func = [&mean, &num_neighbors](double accumulator, const double& val) {
		return accumulator + ((val - mean) * (val - mean) / static_cast<double>(num_neighbors - 1));
	};
	double var = std::accumulate(heights.begin(), heights.end(), 0.0, variance_func);

	features.push_back(height_range);
	features.push_back(height_above_min);
	features.push_back(height_below_max);
	features.push_back(mean);
	features.push_back(var);
}