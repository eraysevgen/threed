#include "file_io.hpp"

/**
 * @brief Read the point cloud file
 * @param file_path las/laz file path
 * @param point_cloud threed::PointCloud reference
 */
void threed::fill_point_cloud_data(const std::string file_path, threed::PointCloud& point_cloud)
{
	std::string file_ext = file_path.substr(file_path.size() - 4, file_path.size());

	if ((file_ext == ".las") || (file_ext == ".laz")) {
		read_las_file(file_path, point_cloud);
	} else if (file_ext == ".npy") {
		throw std::runtime_error(std::string("Not Implemented"));
	} else {
		std::cout << "No extention found" << "\n";
		throw std::runtime_error(std::string("Unknown file extension"));
	}
}
/**
 * @brief Read las file
 *
 * @param las_file_path las/laz file path
 * @param point_cloud threed::PointCloud reference
 */
void threed::read_las_file(const std::string las_file_path, threed::PointCloud& point_cloud)
{
	// TODO merge this function with the previous one ?

	if (!threed::is_file_exist(las_file_path))
		throw std::runtime_error("File not found");

	std::cout << "-> Reading the file ..." << las_file_path.substr(las_file_path.size() - 15, las_file_path.size())
			  << " ...";
	auto start_reading = std::chrono::steady_clock::now();

	std::ifstream file_stream(las_file_path, std::ios::binary);
	if (!file_stream.is_open()) {
		std::cerr << "File could not be opened" << las_file_path << "\n";
		return;
	}

	lazperf::reader::generic_file f(file_stream);

	size_t						num_points = f.pointCount();
	const lazperf::base_header& header	   = f.header();

	const uint8_t major_version = header.version.major;
	const uint8_t minor_version = header.version.minor;
	const uint8_t point_format	= header.point_format_id;

	uint16_t		 pointSize = header.point_record_length;
	lazperf::vector3 scales	   = header.scale;

	lazperf::vector3 offsets = header.offset;
	point_cloud.offsets[0]	 = static_cast<double>(offsets.x);
	point_cloud.offsets[1]	 = static_cast<double>(offsets.y);
	point_cloud.offsets[2]	 = static_cast<double>(offsets.z);

	point_cloud.data.reserve(num_points);

	size_t			  buffer_points = 10000;
	std::vector<char> buffer(buffer_points * pointSize);
	size_t			  points_read = 0;
	while (points_read < num_points) {

		// Determine how many points to read in this iteration
		size_t points_to_read = std::min(buffer_points, num_points - points_read);
		file_stream.read(buffer.data(), points_to_read * pointSize);

		// Process points in buffer
		for (size_t i = 0; i < points_to_read; ++i) {
			int32_t x_raw = *reinterpret_cast<int32_t*>(&buffer[i * pointSize]);
			int32_t y_raw = *reinterpret_cast<int32_t*>(&buffer[i * pointSize + 4]);
			int32_t z_raw = *reinterpret_cast<int32_t*>(&buffer[i * pointSize + 8]);

			// Store raw values without scaling, for faster loading
			point_cloud.data.push_back(
				{ static_cast<double>(static_cast<double>(x_raw) * scales.x) - point_cloud.offsets[0],
				  static_cast<double>(static_cast<double>(y_raw) * scales.y) - point_cloud.offsets[1],
				  static_cast<double>(static_cast<double>(z_raw) * scales.z) - point_cloud.offsets[2] });
		}

		points_read += points_to_read;
	}

	std::cout << " done in " << since(start_reading).count() / 1000.0 << "secs. (num_points=" << num_points << ")\n";
}

/**
 * @brief Write the features to numpy file format
 *
 * @param out_file_name out npy file path
 * @param out_features features in std::vector<std::vector<float>>
 */
void threed::write_features(std::string out_file_name, const std::vector<std::vector<double>>& out_features)
{
	std::cout << "-> Writing the file ..." << out_file_name.substr(out_file_name.size() - 15, out_file_name.size())
			  << " ...";
	auto start_writing = std::chrono::steady_clock::now();

	std::string file_ext = out_file_name.substr(out_file_name.size() - 4, out_file_name.size());

	if (file_ext != ".npy") {
		throw std::runtime_error(std::string("Only npy output supported"));
	}

	size_t row = out_features.size();
	size_t col = out_features[0].size();

	std::vector<double> data(row * col);
	size_t				counter = 0;
	for (size_t i = 0; i < row; i++) {
		for (size_t j = 0; j < col; j++) {
			data[counter] = out_features[i][j];
			counter++;
		}
	}

	cnpy::npy_save(out_file_name, &data[0], { row, col }, "w");
	std::cout << " done in " << since(start_writing).count() / 1000.0 << "secs. (num_points=" << out_features.size()
			  << ") (num_features=" << col << ")\n";
}
/**
 * @brief Check the file exist
 *
 * @param file_path file path
 * @return true if exists
 * @return false not exist
 */
bool threed::is_file_exist(const std::string file_path)
{
	// TODO there is the same function in main, merge those

	std::filesystem::path path(file_path);
	return std::filesystem::exists(path);
}