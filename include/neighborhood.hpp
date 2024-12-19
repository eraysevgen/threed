#ifndef NEIGHBORHOOD_HPP_
#define NEIGHBORHOOD_HPP_

#include <algorithm>
#include <array>
#include <iostream>
#include <thread>
#include <vector>

// nanoflann search
#include "nanoflann.hpp"

// pcl search
#include <pcl/common/pca.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/containers/device_memory.h>
#include <pcl/gpu/octree/octree.hpp>
#include <pcl/memory.h>
#include <pcl/octree/octree_search.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// time keeping
#include "file_io.hpp"
#include "timer.hpp"

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
