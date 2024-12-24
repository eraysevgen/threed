#ifndef FILE_IO_HPP_
#define FILE_IO_HPP_
#define NOMINMAX

#include "header.hpp"
#include "las.hpp"
#include "readers.hpp"

#include "cnpy.h"

#include "timer.hpp"

#include <filesystem>
#include <iostream>
#include <string.h>
#include <string>
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
void fill_point_cloud_data(const std::string, PointCloud&);
void read_las_file(const std::string, PointCloud&);
void write_features(std::string,						   // out file name
					const std::vector<std::vector<float>>& // features
);
bool is_file_exist(const std::string);
}
#endif