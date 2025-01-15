#ifndef FEATURES_HPP_
#define FEATURES_HPP_

#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <vector>

#include <Eigen/Dense>

#include "timer.hpp"

#include "file_io.hpp"

#define NUM_GEOMETRIC_FEATURES 9
#define NUM_HEIGHT_FEATURES 5
#define NUM_XYZ_FEATURES 3
#define NUM_DENSITY_FEATURES 1

namespace threed {

void compute_features(const threed::PointCloud&,
					  const threed::PointCloud&,			   // query point cloud
					  const std::vector<std::vector<size_t>>&, // indices
					  std::vector<std::vector<double>>&,	   // out features
					  const bool,
					  const bool,
					  const bool,
					  unsigned int);

void compute_geometric_features(const Eigen::Vector3d&, // eigenvalue
								const Eigen::Matrix3d&, // eigenvectors
								std::vector<double>&,	// out features
								std::string mode		// mode sqrt,raw
);
void compute_height_features(const double,								// height
							 const std::vector<std::array<double, 3>>&, // neighbors
							 std::vector<double>&						// out features
);

void make_the_vector_zeros(std::vector<double>&, const size_t);
void make_the_vector_nan(std::vector<double>&, const size_t);
}
#endif