#ifndef CPU_NEIGHBORHOOD_HPP_
#define CPU_NEIGHBORHOOD_HPP_

#include "point_cloud.hpp"
#include <cstddef>
#include <vector>

namespace threed {
void compute_indices_by_nanoflann(const PointCloud&,				 // host point cloud
								  const PointCloud&,				 // query point cloud
								  std::vector<std::vector<size_t>>&, // indices
								  const size_t,						 // k
								  const double,						 // radius
								  const bool,						 // is radius
								  const bool   = false,				 // is sorted
								  const size_t = 256,				 // max num points in radius
								  const size_t = 65536,				 // batch size in query
																	 // (be consistent with pcl method)
								  unsigned int = 1					 // num of threads
);

}
#endif
