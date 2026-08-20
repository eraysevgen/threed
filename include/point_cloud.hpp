#ifndef POINT_CLOUD_HPP_
#define POINT_CLOUD_HPP_

#include <array>
#include <vector>

namespace threed {

struct PointCloud {
	std::vector<std::array<double, 3>> data;
	std::array<double, 3>			   offsets;

	void release()
	{
		data.clear();
		data.shrink_to_fit();
		offsets = { 0.0, 0.0, 0.0 };
	}
};

}
#endif
