#ifndef NFLANN_ADAPTOR_HPP_
#define NFLANN_ADAPTOR_HPP_

#include "nanoflann.hpp"
#include "point_cloud.hpp"

namespace threed {
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
			bb = { };
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

using KDTreeAdaptor
	= nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, threed::NFlannPointCloudAdaptor>,
										  threed::NFlannPointCloudAdaptor,
										  3>;
}
#endif
