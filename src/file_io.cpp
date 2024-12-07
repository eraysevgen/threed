#include "file_io.hpp"

void threed::fill_point_cloud_data(
    const std::string file_path,
    threed::PointCloud& point_cloud
    )
{
    std::string file_ext = file_path.substr(file_path.size() - 4,file_path.size());
 
    if ((file_ext ==".las") || (file_ext == ".laz")) 
    {
        read_las_file(file_path, point_cloud);
    }
    else if (file_ext == ".npy")
    {
         throw std::runtime_error(std::string("Not Implemented"));
    }
    else
    {  
        std::cout<< "No extention found" << "\n";
        throw std::runtime_error(std::string("Unknown file extension"));
        
    }
}

void threed::read_las_file(
    const std::string las_file_path,
    threed::PointCloud& point_cloud
    )
{
    if (!threed::is_file_exist(las_file_path))
        throw std::runtime_error("File not found");
    
    std::cout << "-> Reading the file: ... " << las_file_path.substr(las_file_path.size() - 15,las_file_path.size()) <<  " ..." ;
    auto start_reading = std::chrono::steady_clock::now();

    std::ifstream file_stream(las_file_path, std::ios::binary);
    if (!file_stream.is_open()) 
    {
        std::cerr << "File could not be opened" << las_file_path <<"\n";
        return;
    }

    lazperf::reader::generic_file f(file_stream);

    size_t num_points = f.pointCount();
    const lazperf::base_header& header = f.header();
    
    const uint8_t major_version = header.version.major;
    const uint8_t minor_version = header.version.minor;
    const uint8_t point_format = header.point_format_id;

    uint16_t pointSize = header.point_record_length;
    lazperf::vector3 scales = header.scale;

    lazperf::vector3 offsets = header.offset;
    point_cloud.offsets[0] = static_cast<double>(offsets.x);
    point_cloud.offsets[1] = static_cast<double>(offsets.y);
    point_cloud.offsets[2] = static_cast<double>(offsets.z);
    
    point_cloud.data.reserve(num_points);

    size_t buffer_points = 10000; 
    std::vector<char> buffer(buffer_points * pointSize);
    size_t points_read = 0;
    while (points_read < num_points) {
        
        // Determine how many points to read in this iteration
        size_t points_to_read = std::min(buffer_points, num_points - points_read);
        file_stream.read(buffer.data(), points_to_read * pointSize);

        // Process points in buffer
        for (size_t i = 0; i < points_to_read; ++i) 
        {
            int32_t x_raw = *reinterpret_cast<int32_t*>(&buffer[i * pointSize]);
            int32_t y_raw = *reinterpret_cast<int32_t*>(&buffer[i * pointSize + 4]);
            int32_t z_raw = *reinterpret_cast<int32_t*>(&buffer[i * pointSize + 8]);

            // Store raw values without scaling, for faster loading
            point_cloud.data.push_back({static_cast<double>(static_cast<double>(x_raw) * scales.x), 
                                        static_cast<double>(static_cast<double>(y_raw) * scales.y), 
                                        static_cast<double>(static_cast<double>(z_raw) * scales.z)});
        }

        points_read += points_to_read;
    }
    /* This is the first version, it is slower than the buffer version above
    
    for (size_t i = 0; i < num_points; i++) 
    {
        try
        {
             // Read point into buffer
            f.readPoint(buffer.data());

            Access XYZ coordinates 
            
            XYZ coordinates are in the first 12 bytes in both 10 and 14 versions
            
            
            int32_t x_raw = *reinterpret_cast<int32_t*>(&buffer[0]);
            int32_t y_raw = *reinterpret_cast<int32_t*>(&buffer[4]);
            int32_t z_raw = *reinterpret_cast<int32_t*>(&buffer[8]);
            
            // Apply scaling and offset if needed (use f->header() for scaling factors)
            double x = x_raw * scales.x; // + offsets.x;
            double y = y_raw * scales.y; //+ offsets.y;
            double z = z_raw * scales.z; //+ offsets.z;

             Then the intensity stored in the next bytes 
            //uint16_t intensity = *reinterpret_cast<uint16_t*>(&buffer[12]);

            //std::cout << "Point " << i << ": (" << x << ", " << y << ", " << z << ")\n";
            point_cloud.push_back({static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)});
        }
        catch(const std::exception& e)
        {
            std::cerr << "Exception reading point " << i << ": " << e.what() << std::endl;
            break;
        }
    }*/
    //std::cout << " \n \t num_points=" << num_points <<" succesfully read in " << since(start_reading).count()/1000.0 << "seconds\n";
    std::cout << "\n \t done in " << since(start_reading).count()/1000.0 << "secs. (num_points=" << num_points <<")\n";
}

void threed::write_features(
    std::string out_file_name,
    const std::vector<std::vector<float>>& out_features)
{
    std::cout << "-> Writing the file: ... " << out_file_name.substr(out_file_name.size() - 15,out_file_name.size()) <<  " ..." ;
    auto start_writing = std::chrono::steady_clock::now();

    std::string file_ext = out_file_name.substr(out_file_name.size() - 4,out_file_name.size());
 
    if (file_ext != ".npy")
    {
         throw std::runtime_error(std::string("Only npy output supported"));
    }

    size_t row = out_features.size();
    size_t col = out_features[0].size();

    std::vector<float> data(row * col);
    size_t counter = 0;
    for (size_t i = 0; i < row ; i++)
    {
        for (size_t j = 0; j <  col; j++)
        {
            data[counter] = out_features[i][j];
            counter++;
        }
    }
    
    cnpy::npy_save(out_file_name, &data[0], { row,col}, "w");
    std::cout << "\n \t done in " << since(start_writing).count()/1000.0 << "secs. (num_points=" << out_features.size() <<") (num_features=" << col << ")\n";
}

bool threed::is_file_exist(
    const std::string file_path
)
{
    std::filesystem::path path(file_path);
    return std::filesystem::exists(path);
    //if (boost::filesystem::exists(file_path)) return true;
    //else return false;

}