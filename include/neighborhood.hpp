#ifndef NEIGHBORHOOD_HPP_
#define NEIGHBORHOOD_HPP_

#include<vector>
#include<array> 
#include<iostream>
#include<algorithm>

// nanoflann search
#include"nanoflann.hpp"

// pcl search
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/gpu/octree/octree.hpp>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/memory.h>
#include <pcl/gpu/containers/device_memory.h>
#include <pcl/octree/octree_search.h>
#include <pcl/common/pca.h>

// time keeping
#include "timer.hpp"
#include "file_io.hpp"

namespace threed
{
void compute_indices_by_nanoflann(
    PointCloud&,                        // host point cloud
    PointCloud&,                        // query point cloud
    std::vector<std::vector<size_t>>&,  
    const size_t,                       // k
    const double,                       // radius
    const bool,                         // is radius
    const bool = false                  // is sorted
);


void compute_indices_by_pcl(
    PointCloud&,                        // host point cloud
    PointCloud&,                        // query point cloud
    std::vector<std::vector<size_t>>&,  // indices for queries
    const size_t,                       // k
    const double,                       // radius
    const bool,                         // is radius
    const bool   = false,               // is sorted
    const size_t = 32,                  // max num points in radius
    const size_t = 65536                // batch size in query
);

}
#endif
