#ifndef FILE_IO_HPP_
#define FILE_IO_HPP_
#define NOMINMAX

// lazperf
#include "header.hpp"
#include "las.hpp"
#include "readers.hpp"

// npy file io
#include "cnpy.h"

// timer for time keeping
#include "timer.hpp"

#include <filesystem>
#include <iostream>
#include <string.h>
#include <string>
#include <vector>

// #include <boost/filesystem.hpp>

// TODO make everything as struct
// it is easy to manage
namespace threed {

struct PointCloud {
	std::vector<std::array<double, 3>> data;
	std::array<double, 3>			   offsets;

	void release()
	{
		// Clear the vector and release its memory
		data.clear();
		data.shrink_to_fit();

		// Reset offsets to zeros (or another default value)
		offsets = { 0.0, 0.0, 0.0 };
	}
};

// a generic function to read point cloud data
// void fill_point_cloud_data(
//   const std::string,                      // file name
//   std::vector<std::array<double,3>>&,      // data
//   std::vector<double>&                    // offset
//   );

void fill_point_cloud_data(const std::string, PointCloud&);

// read specifically las or laz files
// void read_las_file(
//    const std::string,                      // file name
//    std::vector<std::array<double,3>>&,      // data
//    std::vector<double>&                    // offset
//   );

void read_las_file(const std::string, PointCloud&);

// write features to the file - npy
void write_features(std::string,						   // out file name
					const std::vector<std::vector<float>>& // features
);

bool is_file_exist(const std::string);
}

#endif