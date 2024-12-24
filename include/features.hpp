#ifndef FEATURES_HPP_
#define FEATURES_HPP_

#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <vector>

#include <Eigen/Dense>

#include "timer.hpp"

#include <pcl/common/pca.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "file_io.hpp"

namespace threed {

void compute_features(const threed::PointCloud&,
					  const threed::PointCloud&,			   // query point cloud
					  const std::vector<std::vector<size_t>>&, // indices
					  std::vector<std::vector<float>>&,		   // out features
					  const bool,
					  const bool,
					  const bool,
					  unsigned int);

void compute_geometric_features(const Eigen::Vector3f&, // eigenvalue
								const Eigen::Matrix3f&, // eigenvectors
								std::vector<float>&,	// out features
								std::string mode		// mode sqrt,raw
);
void compute_height_features(const float,				// height
							 const std::vector<float>&, // neighborhing heights
							 std::vector<float>&		// out features
);

void make_the_vector_zeros(std::vector<float>&, const size_t);
}
#endif