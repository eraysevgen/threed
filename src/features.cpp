#include "features.hpp"
#include "timer.hpp"

void threed::compute_features(
    threed::PointCloud                      & host_point_cloud,
    threed::PointCloud                      & query_point_cloud,
    const std::vector<std::vector<size_t>>  & neighborhood_indices,
    std::vector<std::vector<float>>         & features)
{
    std::cout << "-> Computing features ... ";
    auto start_time = std::chrono::steady_clock::now();
    
    features.resize(neighborhood_indices.size());

    for (size_t i =0; i<neighborhood_indices.size();i++)
    {
        // TODO make an alternative for radius search less than 3 points in the neigborhood
        // you may use knn search, such an alternative
        
        if (neighborhood_indices[i].size() <3)
        {
            threed::make_the_vector_zeros(features[i],8);
            threed::make_the_vector_zeros(features[i],3);
            continue;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr neighbors(new pcl::PointCloud<pcl::PointXYZ>); 
        neighbors->width = neighborhood_indices[i].size();
	    neighbors->height = 1;
	    neighbors->resize(neighborhood_indices[i].size());

	    pcl::PCA<pcl::PointXYZ> pca(new pcl::PCA<pcl::PointXYZ>);
        std::vector<float> neighbor_heights;
        Eigen::Vector3f eigenvalues;
	    Eigen::Matrix3f eigenvectors;

        for (size_t j = 0; j<neighborhood_indices[i].size();j++)
        {   
            // TODO index should be size_t not use int

            int index = neighborhood_indices[i][j];
            pcl::PointXYZ point(static_cast<float>(host_point_cloud.data[index][0]),
                                static_cast<float>(host_point_cloud.data[index][1]),
                                static_cast<float>(host_point_cloud.data[index][2]));

            neighbors->points[j] = point;
		    neighbor_heights.push_back(point.z);
        } // end for - inner point cloud cosntruction

        pca.setInputCloud(neighbors);
	
	    eigenvalues = pca.getEigenValues();
	    eigenvectors = pca.getEigenVectors();

        threed::compute_geometric_features(eigenvalues,eigenvectors,features[i]);
        threed::compute_height_features(static_cast<float>(host_point_cloud.data[i][2]),neighbor_heights,features[i]);

    } // end for - outer query indices vector

    // TODO you can use PointXYZL or XYZLRGB for PCL structures 
    // think about it later

    std::cout << "\n \t done in " << static_cast<double>(since(start_time).count()) / 1000.0 << "secs.\n";

}
void threed::compute_geometric_features(
    const Eigen::Vector3f  & val,
	const Eigen::Matrix3f  & vec,
    std::vector<float>& features
)
{
    float e1 =val[0]; float e2 = val[1]; float e3 = val[2];
	float sum_of_eigens = e1 + e2 +e3;
    Eigen::Vector3f ev1 = vec.col(0);
	Eigen::Vector3f ev2 = vec.col(1);
	Eigen::Vector3f ev3 = vec.col(2);

    Eigen::Vector3f z(0.0, 0.0, 1.0);
	
	float verticality = 1.0 - std::abs(z.dot(ev3));	
   
	float linearity = (e1 - e2) / e1;
	float planarity = (e2 - e3) / e1;
	float sphericity = (e3 / e1);
	float omnivariance = std::cbrtf(e1 * e2 * e3);
	float anisotropy = (e1 - e3) / e1;
	float eigenentropy = -(e1 * std::logf(e1)) +
					      (e2 * std::logf(e2)) +
					      (e3 * std::logf(e3));

	float surface_variation = e3 / (sum_of_eigens);
    /*
    In the paper we are using
    
    Sum of eigenvalues
    Linearity
    planarity
    sphericity
    omnivariance
    eigenentropy
    surface variation
    anisotropy
    absolute moment (6) -> skip those in v2.1.1-beta
    vertical moment (2) -> skip those in v2.1.1 beta
    verticality

    As a total of (9 + 6 + 2 = 17) 

    */
    features.push_back(sum_of_eigens);
    features.push_back(linearity);
    features.push_back(planarity);
    features.push_back(sphericity);
    features.push_back(omnivariance);
    features.push_back(anisotropy);
    features.push_back(eigenentropy);
    features.push_back(surface_variation);
    features.push_back(verticality);

}

void threed::make_the_vector_zeros(
    std::vector<float>& vec,
    const size_t n
)
{
    for (size_t i =0; i<n; ++i)
        vec.push_back(0.0);
}

void threed::compute_height_features(
    const float h,
    std::vector<float> const & heights,
    std::vector<float>& features
)

/*
In the paper

Height range        zmax - zmin
height above min    z - zmin
height below max    zmax - z
average height      ave(zn)
variance            var(zn) 

as a total of 5 features


*/
{
    size_t size = heights.size();
    auto min_max = std::minmax_element(heights.begin(),heights.end());
    float sum = std::accumulate(heights.begin(),heights.end(),static_cast<float>(0.0));
    
    float height_range      =   static_cast<float>(*min_max.second - *min_max.first);
    float height_above_min  =   static_cast<float>( h - *min_max.first);
    float height_below_max  =   static_cast<float>(*min_max.second - h);
    float mean              =   sum / static_cast<float>(heights.size());
    
    // the var function is from
    //https://stackoverflow.com/questions/33268513/calculating-standard-deviation-variance-in-c
    auto variance_func = [&mean, &size](float accumulator, const float& val) 
    {
        return accumulator + ((val - mean)*(val - mean) / (size - 1));
    };
    float var = std::accumulate(heights.begin(),heights.end(),0.0,variance_func);

    //features.push_back(static_cast<float>( h - *min_max.first));
    //features.push_back(static_cast<float>(*min_max.second - h));
    //features.push_back(static_cast<float>(sum / static_cast<float>(heights.size())));
    features.push_back(height_range);
    features.push_back(height_above_min);
    features.push_back(height_below_max);
    features.push_back(mean);
    features.push_back(var);
    
}