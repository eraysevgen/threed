#ifndef GPU_NEIGHBORHOOD_HPP_
#define GPU_NEIGHBORHOOD_HPP_

#include "point_cloud.hpp"
#include <cstddef>
#include <vector>

namespace threed {

void compute_indices_by_pcl(PointCloud&,					   // host point cloud
							PointCloud&,					   // query point cloud
							std::vector<std::vector<size_t>>&, // indices for queries
							const size_t,					   // k
							const double,					   // radius
							const bool,						   // is radius
							const bool	 = false,			   // is sorted
							const size_t = 32,				   // max num points in radius
							const size_t = 65536,			   // batch size in query,
							unsigned int = 1				   // num of threads
);

void parallel_add_indices(const std::vector<int>&,			 // size info from gpu
						  const std::vector<int>&,			 // indices from gpu
						  std::vector<std::vector<size_t>>&, // main indices vector in cpu
						  size_t,							 // how many points read
						  size_t,							 // max k is a need for indexing
						  unsigned int						 // num threads in parallel processing
);

}
#endif
