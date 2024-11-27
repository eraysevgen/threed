#include "neighborhood.hpp"


// nanofloann adaptor for our data structure
//using PointCloud = std::vector<std::array<double, 3>>;
// TODO Make this file seperate for adaptors
// TODO Make an adaptor for PCL POintXYZ
// TODO make this adaptor for PointCloud structure


struct NFlannPointCloudAdaptor 
{
    const threed::PointCloud& point_cloud;

    // The constructor takes a reference to the data structure to adapt.
    NFlannPointCloudAdaptor(const threed::PointCloud& pts) : point_cloud(pts) {}

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const {
        return point_cloud.data.size();
    }

    // Returns the `dim`-th component of the `idx`-th point in the dataset.
    inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
        return point_cloud.data[idx][dim];
    }

    //template <class BBOX>
    //bool kdtree_get_bbox(BBOX&) const
    //{
    //    return false;
    //}
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& bb) const
    {
        if (point_cloud.data.empty())
            bb = {};
        else
        {
            for (size_t j = 0; j <3; ++j)
            {
                double val = point_cloud.data[0][j];
                bb[j].low = val;
                bb[j].high = val;
            }

            for (size_t i = 1; i < point_cloud.data.size(); ++i)
            {
                for (size_t j = 0; j <3; ++j)
                {
                    double val = point_cloud.data[i][j];
                    if (val < bb[j].low)
                        bb[j].low = val;
                    if (val > bb[j].high)
                        bb[j].high = val;
                }
            }
        }
        return true;
    }
};

using KDTreeAdaptor = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<double, NFlannPointCloudAdaptor>,
        NFlannPointCloudAdaptor,3>;

void threed::compute_indices_by_nanoflann(
    threed::PointCloud & host_point_cloud,
    threed::PointCloud & query_point_cloud,
    std::vector<std::vector<size_t>>& indices,
    const size_t k,
    const double radius,
    const bool is_radius,
    const bool is_sorted)
{
    // TODO make this search function a better way with templates
    // See https://github.com/jlblancoc/nanoflann/blob/master/examples/example_with_cmake/pointcloud_example.cpp

    std::cout << "-> Creating kdtree ... ";
	auto start_time = std::chrono::steady_clock::now();

    NFlannPointCloudAdaptor adaptor(host_point_cloud);

    KDTreeAdaptor kd_tree(3, adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));

    // Build the index
    kd_tree.buildIndex();
    
    std::cout << "\n \t done in " << static_cast<double>(since(start_time).count()) / 1000.0 << "secs.\n";
   
    
    std::cout << "-> Computing indices ... ";
    start_time = std::chrono::steady_clock::now();
    
    for (size_t i =0; i<query_point_cloud.data.size();i++)
    {
        std::array<double,3> query_point = query_point_cloud.data[i];
        std::vector<size_t> pts_indices;

        if (is_radius)
        {
            std::vector<nanoflann::ResultItem<uint32_t, double>> matches;
            nanoflann::SearchParameters params(0.0,is_sorted);
            
            const size_t num_matches = kd_tree.radiusSearch(query_point.data(), radius * radius , matches,params);
            
            for (size_t ir = 0; ir < num_matches; ir++)
                pts_indices.push_back(static_cast<size_t>(matches[ir].first));

        }
        else
        {
            nanoflann::KNNResultSet<double, size_t> resultSet(k);
            
            std::vector<size_t> ret_index(k);
            std::vector<double>   out_dist_sqr(k);
            resultSet.init(&ret_index[0], &out_dist_sqr[0]);
            kd_tree.findNeighbors(resultSet, query_point.data(), nanoflann::SearchParameters());

            //size_t k_results = kd_tree.knnSearch(query_point.data(), k, &ret_index[0], &out_dist_sqr[0]);

             // In case of less points in the tree than requested:
            //ret_index.resize(k_results);
            //out_dist_sqr.resize(k_results);

            for (size_t ik = 0; ik < k; ik++)
                pts_indices.push_back(static_cast<size_t>(ret_index[ik]));
    
        } // end if else

        indices.push_back(pts_indices);

    } // end for
    std::cout << "\n \t done in " << static_cast<double>(since(start_time).count()) / 1000.0 << "secs.\n";
   
} // end function


void threed::compute_indices_by_pcl(
    threed::PointCloud & host_point_cloud,
    threed::PointCloud & query_point_cloud,
    std::vector<std::vector<size_t>>& indices,
    const size_t k,
    const double radius,
    const bool is_radius,
    const bool is_sorted,
    const size_t batch_size,
    const size_t max_k
)
{

// create points clouds
	// host -> pcl point cloud
	pcl::PointCloud<pcl::PointXYZ>  pcl_host_point_cloud;

	// query -> vector of pcl point
	//std::vector<pcl::PointXYZ> pcl_query_point_cloud;

    // fill host data
	pcl_host_point_cloud.width = host_point_cloud.data.size();
	pcl_host_point_cloud.height = 1;
	pcl_host_point_cloud.is_dense = false;
	pcl_host_point_cloud.resize(host_point_cloud.data.size());

    indices.resize(host_point_cloud.data.size());
   
	for (size_t i = 0; i < host_point_cloud.data.size(); i ++)
	{
        // TODO make an alternative way to initialize
		pcl::PointXYZ p;
		p.x = static_cast<float>(host_point_cloud.data[i][0]);
		p.y = static_cast<float>(host_point_cloud.data[i][1]);
		p.z = static_cast<float>(host_point_cloud.data[i][2]);
		pcl_host_point_cloud.points[i] = p;
	}
    
    std::cout << "-> Creating octree ... ";
	auto start_time = std::chrono::steady_clock::now();
	
	pcl::gpu::Octree::PointCloud pcl_host_point_cloud_device;
	pcl_host_point_cloud_device.upload(pcl_host_point_cloud.points);

	pcl::gpu::Octree octree_gpu;
	octree_gpu.setCloud(pcl_host_point_cloud_device);
	octree_gpu.build();

	std::cout << "\n \t done in " << static_cast<double>(since(start_time).count()) / 1000.0 << "secs.\n";
    
    
    std::cout << "-> Computing indices ... ";
    start_time = std::chrono::steady_clock::now();
    
    if (is_radius)
    {
        pcl::gpu::Octree::Queries octree_gpu_quaries;
	    std::vector<pcl::PointXYZ> pcl_query_point_cloud;

        size_t buffer_points = 100000; 
        size_t points_read = 0;
        size_t num_points_in_query = query_point_cloud.data.size();

        while (points_read <num_points_in_query) 
        {
            // Determine how many points to read in this iteration
            size_t points_to_read = std::min(buffer_points, num_points_in_query - points_read) ;
          
            for (size_t i = points_read; i < points_read+points_to_read; ++i) 
            {
                pcl::PointXYZ p;
		        p.x =  static_cast<float>(query_point_cloud.data[i][0]);
		        p.y =  static_cast<float>(query_point_cloud.data[i][1]);
		        p.z =  static_cast<float>(query_point_cloud.data[i][2]);
		        pcl_query_point_cloud.push_back(p);
            }

            octree_gpu_quaries.upload(pcl_query_point_cloud);

            //const int max_answers = 32;

			pcl::gpu::NeighborIndices result_gpu(octree_gpu_quaries.size(), max_k);

			octree_gpu.radiusSearch(octree_gpu_quaries, radius, max_k, result_gpu);
			
            std::vector<int> sizes, data;
			result_gpu.sizes.download(sizes);
			result_gpu.data.download(data);
            
            for (size_t j = 0; j < sizes.size(); ++j)
			{
                size_t current_pts_idx = j + points_read;
                int neighbor_size = sizes[j];
                for (size_t m = 0;m<neighbor_size;++m)
                {
                    indices[current_pts_idx].push_back(data[m + j*max_k]);
                }
                // TODO disable this, not sorting based on distances, only for the indices
                if (is_sorted)
                    std::sort(indices[current_pts_idx].begin(),indices[current_pts_idx].end());

				//parallel_index_operation(sizes[j], j, start_idx, max_answers, neighbor_sizes, neighbor_indices,
				//	data);
			}

            points_read += points_to_read;
            octree_gpu_quaries.release();
			pcl_query_point_cloud.clear();
        }
    }
    else
    {
        throw std::runtime_error(std::string("Knn on gpu not supported, use knn on cpu instead"));
    }

    std::cout << "\n \t done in " << static_cast<double>(since(start_time).count()) / 1000.0 << "secs.\n";
    
}