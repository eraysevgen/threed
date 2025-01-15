#ifndef NEIGHBORHOOD_HPP_
#define NEIGHBORHOOD_HPP_

#include <algorithm>
#include <array>
#include <iostream>
#include <thread>
#include <vector>

#include "nanoflann.hpp"

#include <pcl/gpu/octree/octree.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

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

/**
 * @brief nanofloann adaptor for thred::PointCloud
 *
 */
struct NFlannPointCloudAdaptor {
	const threed::PointCloud& point_cloud;

	NFlannPointCloudAdaptor(const threed::PointCloud& pts)
		: point_cloud(pts)
	{
	}

	inline size_t kdtree_get_point_count() const { return point_cloud.data.size(); }

	inline double kdtree_get_pt(const size_t idx, const size_t dim) const { return point_cloud.data[idx][dim]; }

	template <class BBOX> bool kdtree_get_bbox(BBOX& bb) const
	{
		if (point_cloud.data.empty())
			bb = {};
		else {
			for (size_t j = 0; j < 3; ++j) {

				bb[j].low  = std::numeric_limits<double>::max();  // val;
				bb[j].high = -std::numeric_limits<double>::max(); // val;
			}

			for (size_t i = 1; i < point_cloud.data.size(); ++i) {
				for (size_t j = 0; j < 3; ++j) {

					bb[j].low  = std::min(bb[j].low, point_cloud.data[i][j]);
					bb[j].high = std::max(bb[j].high, point_cloud.data[i][j]);
				}
			}
		}
		return true;
	}
};
}
#endif
