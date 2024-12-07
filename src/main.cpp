/*/////////////////////////////////////////////////////////////////////////////////////////
* File        : main.cpp
* Descripion  : Main file, entry point
* Author      : Eray Sevgen
* Date        : 2024 October
*/
/////////////////////////////////////////////////////////////////////////////////////////
// std calls
#include <exception>
#include <iostream>
#include <string>
#include <algorithm>
#include <iterator>
#include<fstream>

// vcpkg calls
#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>

// local calls
#include "timer.hpp"
#include "file_io.hpp"
#include "neighborhood.hpp"
#include "features.hpp"


/////////////////////////////////////////////////////////////////////////////////////////
/*
* Program definitions
*/
/////////////////////////////////////////////////////////////////////////////////////////
#define MAJOR_VERSION 2
#define MINOR_VERSION 1
#define PATCH_VERSION 1
#define RELEASE_CANDIDATE "beta"
#define AUTHOR "Eray Sevgen"
#define DESCRIPTION "A program for feature extraction from 3D point cloud data"
#define SHORT_NAME "threed.exe"


/////////////////////////////////////////////////////////////////////////////////////////
/* get_version_info
* Returns string version information
*/
/////////////////////////////////////////////////////////////////////////////////////////
std::string get_version_info()
{
    std::string main_version_info = "v"+ std::to_string(MAJOR_VERSION) + "." + std::to_string(MINOR_VERSION) + "." + std::to_string(PATCH_VERSION);
    if (RELEASE_CANDIDATE != "")
    return main_version_info + "-"+ RELEASE_CANDIDATE;
    else return main_version_info;

}
/////////////////////////////////////////////////////////////////////////////////////////
/* show_version_info
* Show version information on the command line
*/
/////////////////////////////////////////////////////////////////////////////////////////
void show_version_info()
{
    std::cout << SHORT_NAME << " " << get_version_info() ;//<< " build on " << __DATE__ << " at " << __TIME__ << "\n";
}
/////////////////////////////////////////////////////////////////////////////////////////
/* show_program_info
* Show program information
*/
/////////////////////////////////////////////////////////////////////////////////////////
void show_program_info()
{
    show_version_info();
    std::cout  <<"\n" << DESCRIPTION << "\n";
    std::cout<< "build on " << __DATE__ << " at " << __TIME__;
    std::cout <<" by " AUTHOR << "\n";
}

/////////////////////////////////////////////////////////////////////////////////////////
/*
* run
*
* Parse the arguments and find the corresponding function
*
* @param std::string host_file_path      : host file path
* @param std::string query_file_path     : query file path
* @param std::string out_file_path       : out file path 
* @param std::string neighborhood_name   : neighborhood
* @param std::string neighborhood_engine : device
* @param double neigborhood_radius       : radius       
* @param size_t neighborhood_k           : k
* @param size_t neighborhood_batch_size  : batch_size
* @param size_t neighborhood_max_k       : max_k
* @param bool add_height                 : add height features
* @param bool add_density                : add density information
* @param bool add_xyz                    : add xyz
*/

bool run(std::string host_file_path,
         std::string query_file_path,
         std::string out_file_path,
         std::string neighborhood_name,
         std::string neighborhood_engine,
         double neigborhood_radius,
         size_t neighborhood_k,
         size_t neighborhood_batch_size,
         size_t neigborhood_max_k,
         bool add_height,
         bool add_density,
         bool add_xyz)
{
    std::cout << "Processing ... \n";    
    auto start_time = std::chrono::steady_clock::now();
   
    // fill the point cloud data
    threed::PointCloud host_point_cloud,query_point_cloud;
    threed::fill_point_cloud_data(host_file_path,host_point_cloud);
    threed::fill_point_cloud_data(query_file_path,query_point_cloud);

    std::vector<std::vector<size_t>> query_neighbor_indices;

    // find the neighbors
    
    // TODO make a switch here using the enums
    // TODO we should check the parametrs with asserts

    // select the proper method
    const bool is_radius = neighborhood_name == "radius" ? true:false; 
    const bool is_sorted = false;

    if (neighborhood_engine == "cpu" )
    {
        
        compute_indices_by_nanoflann(
            host_point_cloud,
            query_point_cloud,
            query_neighbor_indices,
            neighborhood_k,
            neigborhood_radius,
            is_radius,
            is_sorted);
    }
    else if (neighborhood_engine == "gpu")
    {
        compute_indices_by_pcl(
            host_point_cloud,
            query_point_cloud,
            query_neighbor_indices,
            neighborhood_k,
            neigborhood_radius,
            is_radius,
            is_sorted,
            neighborhood_batch_size,
            neigborhood_max_k);
    }
    else
    {
        throw std::runtime_error(std::string("Neighborhood device not understandable, cpu or gpu supported only"));
    }

    // compute features
    std::vector<std::vector<float>> out_features;
    compute_features(host_point_cloud,query_point_cloud,query_neighbor_indices,out_features);
    
    // output features
    threed::write_features(out_file_path,out_features);
    
    //std::cout << "\tTotal number of samples : " << out_features.size() << " - Total features per sample : " << out_features[0].size() << "\n";
    std::cout << "all done in " << static_cast<double>(since(start_time).count()) / 1000.0 << "secs.\n";
    return true;
    
}

/////////////////////////////////////////////////////////////////////////////////////////
/* process_command_line 
*
* Parse the yaml config file path
*
* @param argc     argument counts
* @param argv     argument values
* 
* See the webpage: https://stackoverflow.com/questions/5395503/required-and-optional-arguments-using-boost-library-program-options/5519200#5519200
*/
/////////////////////////////////////////////////////////////////////////////////////////
bool process_command_line(int argc, char** argv,
                          std::string& config_file_path)
{
    std::string message = "Allowed option:\n \
    --help,-h\t\t produce help message \n \
    --config,-c\t\t config file, default = ./config.yaml \n \
    --version,-v\t\t display version info \n \
    --desc,-d\t\t display program description \n";

    //using namespace boost;
    //namespace po = boost::program_options;
    //std::cout << "IN command line HERE" << "\n";
    try
    {
        //po::options_description desc("Allowed option", 1024, 512);
        //desc.add_options()
        //  ("help,h",     "produce help message")
        //  ("config,c", po::value<std::string>(&config_file_path)->default_value("config.yaml"), "set the config file path, default config.yaml")
        //  ("version,v", "show version info")
        //  ("desc,d",    "show program description")
        /;
        // TODO fix this one, if config used then argc is 3 ? 
        //std::cout << "IN command line HERE" << config_file_path << "No command line \n";

     

        // if argc is larger or equal 3
        /*
        if (argc >= 3 )
        {
            // check it includes the -c definition
            if (strncmp(argv[1],"-",1)!=0)
            {
                std::cout << "Usage: threed.exe [option]\n\n";
                std::cout << desc << "\n";
                return false;
            }

            
        }
        */
        if (argc == 1)
        {
            config_file_path = "config.yaml";
            std::cout << "argc1 called\n";
        }
        if (argc == 2)
        {
            if ((strncmp(argv[1],"--help",6)==0)  ||(strncmp(argv[1],"-h",2)==0))
            {
                //std::cout << "help called\n";
                std::cout << "Usage: \nthreed.exe [option]\n\n";
                std::cout << message;

                return false;
            }
            else if ((strncmp(argv[1],"--version",9)==0) || (strncmp(argv[1],"-v",2)==0))
            {
                //std::cout << "version called\n";
                show_version_info();
                return false;
            }
            else if ((strncmp(argv[1],"--desc",6)==0) || (strncmp(argv[1],"-d",2)==0))
            {
                //std::cout << "desc called\n";
                show_program_info();
                return false;
            }
            else if ((strncmp(argv[1],"--config",8)==0) || (strncmp(argv[1],"-c",2)==0))
            {
                std::cout << "no config file provided\n";
                return false;
            }
            else
            {
                std::cout << "none options\n";
                return false;
            }
        }
        if (argc ==3)
        {
            if ((strncmp(argv[1],"--config",8)==0) || (strncmp(argv[1],"-c",2)==0))
            {
                //config_file_path = argv[2];
                std::cout << "call config\n";
                //std::filesystem::exists
            }
            else
            {
                std::cout << "unable to understand config\n";
                return false;
            }

            // check the file system
        }
        if (argc>3)
        {
            std::cout << "larger than 33\n";
            return false;
        }

        //
        //if ((argc >2) || (strncmp(argv[1],"-",1)!=0))
        //{
        //    std::cout << "Usage: threed.exe [option]\n\n";
        //    std::cout << desc << "\n";
        //    return false;
        //}
        
        //po::variables_map vm;
        //po::store(po::parse_command_line(argc, argv, desc), vm);
        
        /*if (vm.count("help"))
        {
            std::cout << "Usage: threed.exe [option]\n\n";
            std::cout << desc << "\n";
            return false;
        }
        if (vm.count("version"))
        {
            show_version_info();
            return false;
        }

        if (vm.count("desc"))
        {
            show_program_info();
            return false;
        }
        */
        // There must be an easy way to handle the relationship between the
        // option "help" and "host"-"port"-"config"
        // Yes, the magic is putting the po::notify after "help" option check
        //po::notify(vm);
    }
    catch(std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return false;
    }
    catch(...)
    {
        std::cerr << "Unknown error!" << "\n";
        return false;
    }
    std::cout << "returning true HERE" << "\n";
    return true;
}
/////////////////////////////////////////////////////////////////////////////////////////
/* main
* Entry point
*
* @param argc     argument counts
* @param argv     argument values
*/
/////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    std::string config_file_path;

    bool result = process_command_line(argc, argv, config_file_path);
    result = false;
    if (!result)
        return 1;
    
    if (!threed::is_file_exist(config_file_path))
    {
        throw std::runtime_error("No config file");
        return 1;
    }
    std::string host_file_path,query_file_path,out_file_path,neighborhood_name,neighborhood_engine; 

    double neigborhood_radius;
    size_t neighborhood_k,neighborhood_batch_size,neighborhood_max_k; 
    bool add_height,add_xyz,add_density; 
    
    std::cout << "HERE" << "\n";

    try
    {
        YAML::Node config = YAML::LoadFile(config_file_path);

        YAML::Node files_node = config["files"].as<YAML::Node>();
        YAML::Node neighborhood_node=  config["neighborhood"].as<YAML::Node>();
        YAML::Node features_node=  config["features"].as<YAML::Node>();

        host_file_path = files_node["host_file"].as<std::string>();
        query_file_path = files_node["query_file"].as<std::string>();
        out_file_path = files_node["out_file"].as<std::string>();

        neighborhood_name = neighborhood_node["name"].as<std::string>();
        neighborhood_engine = neighborhood_node["engine"].as<std::string>();
        neigborhood_radius = neighborhood_node["radius"].as<double>();
        neighborhood_k = neighborhood_node["k"].as<size_t>();
        neighborhood_batch_size = neighborhood_node["batch_size"].as<size_t>();
        neighborhood_max_k = neighborhood_node["max_k"].as<size_t>();

        add_height = features_node["height"].as<bool>();
        add_xyz = features_node["add_xyz"].as<bool>();
        add_density = features_node["density"].as<bool>();
    }
    catch(std::exception& e)
    {
        std::cerr << "Error in YAML parsing: " << e.what() << "\n";
        return 1;
    }
    catch(...)
    {
        std::cerr << "Unknown error!" << "\n";
        return 1;
    }
    
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "config file:\t" << config_file_path << "\n";
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "\thost file     :\t" << host_file_path << "\n";
    std::cout << "\tquery file    :\t" << query_file_path << "\n";
    std::cout << "\tout file      :\t" << out_file_path << "\n";
    std::cout << "\tneighborhood  :\t" << neighborhood_name << "\n";
    std::cout << "\tengine         :\t" << neighborhood_engine << "\n";
    std::cout << "\tradius        :\t" << neigborhood_radius << "\n";
    std::cout << "\tk             :\t" << neighborhood_k << "\n";
    std::cout << "\tbatch_size    :\t" << neighborhood_batch_size << "\n";
    std::cout << "\tmax k         :\t" << neighborhood_max_k << "\n";
    std::cout << "\tadd height    :\t" << add_height << "\n";
    std::cout << "\tadd density   :\t" << add_density << "\n";
    std::cout << "\tadd xyz       :\t" << add_height << "\n";
    std::cout << "------------------------------------------------------------------------\n";
     
    bool success =  run(
                        host_file_path,
                        query_file_path,
                        out_file_path,
                        neighborhood_name,
                        neighborhood_engine,
                        neigborhood_radius,
                        neighborhood_k,
                        neighborhood_batch_size,
                        neighborhood_max_k,
                        add_height,
                        add_density,
                        add_xyz);
                        
    if (!success)
        return 1;
    
    return 0;
}
/////////////////////////////////////////////////////////////////////////////////////////
// end of file


