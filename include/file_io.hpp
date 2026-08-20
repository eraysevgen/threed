#ifndef FILE_IO_HPP_
#define FILE_IO_HPP_
#define NOMINMAX

#include "point_cloud.hpp"
#include <string.h>
#include <string>
#include <vector>

namespace threed {

void fill_point_cloud_data(const std::string, PointCloud&);
void read_las_file(const std::string, PointCloud&);
void write_features(std::string,							// out file name
					const std::vector<std::vector<double>>& // features
);
bool is_file_exist(const std::string);
}
#endif