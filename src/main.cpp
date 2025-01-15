/**
 * @file main.cpp
 * @brief The main file for threed.exe
 * @details
 * This file is a part of threed - a feature extraction program for 3d point clouds.
 * The las/laz files read, then local features are computed according to the selected neighborhood
 * method and parameters.
 *
 * Without other open source projects, this program would not be possible, see the dependencies.
 *
 * Even though this program is  heaviliy tested, it still may produce errorneous results,so please
 * use your own risks. See the license for further information.
 *
 * @author Eray Sevgen
 * @date 2024 October
 * @copyright Eray Sevgen (c) 2024
 *
 */
#include <algorithm>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

#include <yaml-cpp/yaml.h>

#include "features.hpp"
#include "file_io.hpp"
#include "neighborhood.hpp"
#include "timer.hpp"

/**
 * @brief Definitions
 *
 */
#define MAJOR_VERSION 2
#define MINOR_VERSION 1
#define PATCH_VERSION 3
#define BUILD_VERSION " "
#define AUTHOR "Eray Sevgen"
#define DESCRIPTION "A program for feature extraction from 3D point cloud data"
#define SHORT_NAME "threed.exe"

/**
 * @brief Get the version info object
 *
 * @return std::string
 */
std::string get_version_info()
{
	std::string main_version_info = "v" + std::to_string(MAJOR_VERSION) + "." + std::to_string(MINOR_VERSION) + "."
		+ std::to_string(PATCH_VERSION);
	if (BUILD_VERSION != "")
		return main_version_info + "-" + BUILD_VERSION;
	else
		return main_version_info;
}

/**
 * @brief Display version information
 *
 */
void show_version_info() { std::cout << SHORT_NAME << " " << get_version_info(); }

/**
 * @brief Display program information
 *
 */
void show_program_info()
{
	show_version_info();
	std::cout << "\n" << DESCRIPTION << "\n";
	std::cout << "build on " << __DATE__ << " at " << __TIME__;
	std::cout << " by " AUTHOR << "\n";
}

/**
 * @brief A helper function to check a file exists in the directory.
 *
 * @param file_path the file path
 * @return true if exists
 * @return false not exists
 */
bool is_file_exist(std::string file_path)
{
	const std::filesystem::path path(file_path);

	if (!std::filesystem::exists(path)) {
		std::cerr << "Error in file path: " << file_path << " does not exist\n";
		return false;
	} else
		return true;
}

/**
 * @brief Get the parameters, find the proper method, run the main process
 *
 * @param host_file_path host file path, where the spatial index created on
 * @param query_file_path query file path, a feature for each point in this cloud is computed based on the host file
 * @param out_file_path out npy file path
 * @param neighborhood_name neighborhood type name, radius or knn
 * @param neighborhood_engine neigborhood library, nanoflann or pcl
 * @param neighborhood_k number of neighbors in knn search
 * @param neigborhood_radius radius distance in radius search
 * @param neigborhood_max_k maximum number of neighbors in radius search; based on value both cpu or gpu is allocated
 * @param neighborhood_batch_size batch size
 * @param add_height add height features which is computed from the neighborhood points
 * @param add_density add number of neighbor information to the end of the feature set
 * @param add_xyz add xyz information to the end of the feature set
 * @param num_threads number of threads for parallel processing
 * @return true once the process in success
 * @return false once someshow in fail
 */
bool run(std::string  host_file_path,
		 std::string  query_file_path,
		 std::string  out_file_path,
		 std::string  neighborhood_name,
		 std::string  neighborhood_engine,
		 size_t		  neighborhood_k,
		 double		  neigborhood_radius,
		 size_t		  neigborhood_max_k,
		 size_t		  neighborhood_batch_size,
		 bool		  add_height,
		 bool		  add_density,
		 bool		  add_xyz,
		 unsigned int num_threads)
{
	if (!is_file_exist(host_file_path))
		return false;

	if (!is_file_exist(query_file_path))
		return false;

	assert(neighborhood_k > 0 && "k must be greater than zero.");
	assert(neigborhood_radius > 0 && "Radius must be greater than zero.");
	assert(neigborhood_max_k > 0 && "Maximum k must be greater than zero.");
	assert(neighborhood_batch_size > 0 && "Batch size must be greater than zero.");
	assert(num_threads > 0 && "Num threads must be greater than zero.");

	assert(neighborhood_name == "knn"
		   || neighborhood_name == "radius" && "Invalid neighborhood: must be 'knn' or 'radius'");

	assert(neighborhood_engine == "cpu"
		   || neighborhood_engine == "gpu" && "Invalid neighborhood engine: must be 'cpu' or 'gpu'");

	std::cout << "Processing ... \n";
	auto start_time = std::chrono::steady_clock::now();

	threed::PointCloud host_point_cloud, query_point_cloud;
	threed::fill_point_cloud_data(host_file_path, host_point_cloud);
	threed::fill_point_cloud_data(query_file_path, query_point_cloud);

	std::vector<std::vector<size_t>> query_neighbor_indices;

	// TODO make a switch here using the enums

	const bool is_radius = neighborhood_name == "radius" ? true : false;
	const bool is_sorted = false;

	if (neighborhood_engine == "cpu") {

		threed::compute_indices_by_nanoflann(host_point_cloud, query_point_cloud, query_neighbor_indices,
											 neighborhood_k, neigborhood_radius, is_radius, is_sorted,
											 neigborhood_max_k, neighborhood_batch_size, num_threads);

	} else if (neighborhood_engine == "gpu") {

		threed::compute_indices_by_pcl(host_point_cloud, query_point_cloud, query_neighbor_indices, neighborhood_k,
									   neigborhood_radius, is_radius, is_sorted, neigborhood_max_k,
									   neighborhood_batch_size, num_threads);
	}
	auto clear_vector = [](auto& vec) { vec.clear(); };

	std::vector<std::vector<double>> out_features;
	threed::compute_features(host_point_cloud, query_point_cloud, query_neighbor_indices, out_features, add_height,
							 add_density, add_xyz, num_threads);

	for (auto& vec : query_neighbor_indices) {
		clear_vector(vec);
	}

	threed::write_features(out_file_path, out_features);
	std::cout << " all done in " << static_cast<double>(since(start_time).count()) / 1000.0 << "secs.\n";

	return true;
}

/**
 * @brief Parse the command line arguments and obtain the yaml config file path
 * Display help, version and program description.
 *
 * @param argc argument counts
 * @param argv argument values
 * @param config_file_path configuration file path
 * @return true when the command line arguments are properly read and understood
 * @return false parsing failed
 *
 */
bool process_command_line(int argc, char** argv, std::string& config_file_path)
{
	std::string message = "Allowed option:\n \
    --help,   -h\t\t produce help message \n \
    --config, -c\t\t config file, default = ./config.yaml \n \
    --version,-v\t\t display version info \n \
    --desc,   -d\t\t display program description \n";

	try {
		if (argc == 1) {
			config_file_path = "./config.yaml";
		}
		if (argc == 2) {
			if ((strncmp(argv[1], "--help", 6) == 0) || ((strncmp(argv[1], "-h", 2) == 0) & (strlen(argv[1]) == 2))) {
				std::cout << "Usage: \nthreed.exe [option]\n\n";
				std::cout << message;

				return false;
			} else if ((strncmp(argv[1], "--version", 9) == 0)
					   || ((strncmp(argv[1], "-v", 2) == 0) & (strlen(argv[1]) == 2))) {
				show_version_info();
				return false;
			} else if ((strncmp(argv[1], "--desc", 6) == 0)
					   || ((strncmp(argv[1], "-d", 2) == 0) & (strlen(argv[1]) == 2))) {
				show_program_info();
				return false;

			} else if ((strncmp(argv[1], "--config", 8) == 0)
					   || ((strncmp(argv[1], "-c", 2) == 0) & (strlen(argv[1]) == 2))) {
				std::cerr << "Error in arguments: No config file provided\n ";
				return false;
			} else {
				std::cerr << "Error in arguments: '" << argv[1] << "' not understood\n ";
				return false;
			}
		}
		if (argc == 3) {
			if ((strncmp(argv[1], "--config", 8) == 0) || ((strncmp(argv[1], "-c", 2) == 0) & (strlen(argv[1]) == 2))) {
				config_file_path = std::string(argv[2]);
			} else {
				std::cerr << "Error in arguments: '" << argv[1] << "' not understood\n ";
				return false;
			}
		}
		if (argc > 3) {
			std::cerr << "Error in arguments: More than 3 arguments passed \n ";
			return false;
		}

	} catch (std::exception& e) {
		std::cerr << "Error: " << e.what() << "\n";
		return false;
	} catch (...) {
		std::cerr << "Unknown error!" << "\n";
		return false;
	}

	return is_file_exist(config_file_path);
}

/**
 * @brief The entry point, main function
 *
 * @param argc argument counts
 * @param argv argument values
 * @return int 1 in fail, otherwise 0 in success
 */
int main(int argc, char* argv[])
{
	std::string config_file_path;

	bool result = process_command_line(argc, argv, config_file_path);

	if (!result)
		return 1;

	std::string host_file_path, query_file_path, out_file_path, neighborhood_name, neighborhood_engine;

	double		 neigborhood_radius;
	size_t		 neighborhood_k, neighborhood_batch_size, neighborhood_max_k;
	bool		 add_height, add_xyz, add_density;
	unsigned int num_threads;
	YAML::Node	 config;
	try {
		config = YAML::LoadFile(config_file_path);

		YAML::Node files_node		 = config["files"].as<YAML::Node>();
		YAML::Node neighborhood_node = config["neighborhood"].as<YAML::Node>();
		YAML::Node features_node	 = config["features"].as<YAML::Node>();
		YAML::Node params_node		 = config["params"].as<YAML::Node>();

		host_file_path	= files_node["host_file"].as<std::string>();
		query_file_path = files_node["query_file"].as<std::string>();
		out_file_path	= files_node["out_file"].as<std::string>();

		neighborhood_name		= neighborhood_node["name"].as<std::string>();
		neighborhood_engine		= neighborhood_node["engine"].as<std::string>();
		neigborhood_radius		= neighborhood_node["radius"].as<double>();
		neighborhood_k			= neighborhood_node["k"].as<size_t>();
		neighborhood_batch_size = neighborhood_node["batch_size"].as<size_t>();
		neighborhood_max_k		= neighborhood_node["max_k"].as<size_t>();

		add_height	= features_node["add_height"].as<bool>();
		add_xyz		= features_node["add_xyz"].as<bool>();
		add_density = features_node["add_density"].as<bool>();

		num_threads = params_node["num_threads"].as<unsigned int>();

	} catch (std::exception& e) {
		std::cerr << "Error in YAML parsing: " << e.what() << "\n";
		return 1;
	} catch (...) {
		std::cerr << "Unknown error!" << "\n";
		return 1;
	}
	std::cout << std::boolalpha;
	std::cout << "------------------------------------------------------------------------\n";
	std::cout << "config file:\t" << config_file_path << "\n";
	std::cout << "------------------------------------------------------------------------\n";
	std::cout << "files:\n";
	std::cout << "------\n";
	std::cout << "  host file         : " << host_file_path << "\n";
	std::cout << "  query file        : " << query_file_path << "\n";
	std::cout << "  out file          : " << out_file_path << "\n";
	std::cout << "neighborhood:\n";
	std::cout << "-------------\n";
	std::cout << "  name              : " << neighborhood_name << "\n";
	std::cout << "  engine            : " << neighborhood_engine << "\n";
	std::cout << "  radius            : " << neigborhood_radius << "\n";
	std::cout << "  k                 : " << neighborhood_k << "\n";
	std::cout << "  batch_size        : " << neighborhood_batch_size << "\n";
	std::cout << "  max k             : " << neighborhood_max_k << "\n";
	std::cout << "features:\n";
	std::cout << "---------\n";
	std::cout << "  add height        : " << add_height << "\n";
	std::cout << "  add density       : " << add_density << "\n";
	std::cout << "  add xyz           : " << add_height << "\n";
	std::cout << "params:\n";
	std::cout << "-------\n";
	std::cout << "  num threads       : " << num_threads << "\n";
	std::cout << "------------------------------------------------------------------------\n";

	try {
		bool success = run(host_file_path, query_file_path, out_file_path, neighborhood_name, neighborhood_engine,
						   neighborhood_k, neigborhood_radius, neighborhood_max_k, neighborhood_batch_size, add_height,
						   add_density, add_xyz, num_threads);

		if (!success)
			return 1;
	}

	catch (std::exception& e) {
		std::cerr << "Error in Processing: " << e.what() << "\n";
		return 1;
	} catch (...) {
		std::cerr << "Unknown error!" << "\n";
		return 1;
	}

	std::string	  metadata_yaml_file_path = out_file_path.substr(0, out_file_path.size() - 4) + ".metadata";
	std::ofstream fout(metadata_yaml_file_path);

	fout << "created by " << SHORT_NAME << " " << get_version_info() << " build on " << __DATE__ << " at " << __TIME__
		 << "\n";
	fout << config;
}