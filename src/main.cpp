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


/////////////////////////////////////////////////////////////////////////////////////////
/*
* Program definitions
*/
/////////////////////////////////////////////////////////////////////////////////////////
#define MAJOR_VERSION 2
#define MINOR_VERSION 1
#define PATCH_VERSION 1
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
    return "v"+ std::to_string(MAJOR_VERSION) + "." + std::to_string(MINOR_VERSION) + "." + std::to_string(PATCH_VERSION);
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
    using namespace boost;
    namespace po = boost::program_options;

    try
    {
        po::options_description desc("Allowed option", 1024, 512);
        desc.add_options()
          ("help,h",     "produce help message")
          ("config,c", po::value<std::string>(&config_file_path)->default_value("config.yaml"), "set the config file path, default config.yaml")
          ("version,v", "show version info")
          ("desc,d",    "show program description")
        ;
        // TODO fix this one, if config used then argc is 3 ? 
        if ((argc >2) || (strncmp(argv[1],"-",1)!=0))
        {
            std::cout << "Usage: threed.exe [option]\n\n";
            std::cout << desc << "\n";
            return false;
        }
        
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        
        if (vm.count("help"))
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

        // There must be an easy way to handle the relationship between the
        // option "help" and "host"-"port"-"config"
        // Yes, the magic is putting the po::notify after "help" option check
        po::notify(vm);
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
    if (!result)
        return 1;
    
    if (!threed::is_file_exist(config_file_path))
    {
        throw std::runtime_error("No config file");
        return 1;
    }
    std::string host_file_path,query_file_path,out_file_path,neighborhood_name,neighborhood_device; 
    float neigborhood_range;
    size_t neighborhood_k,neighborhood_batch,neighborhood_max_k; 
    bool add_height,add_intensity,add_density; 

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
        neighborhood_device = neighborhood_node["device"].as<std::string>();
        neigborhood_range = neighborhood_node["range"].as<double>();
        neighborhood_k = neighborhood_node["k"].as<size_t>();
        neighborhood_batch = neighborhood_node["batch"].as<size_t>();
        neighborhood_max_k = neighborhood_node["max_k"].as<size_t>();

        add_height = features_node["height"].as<bool>();
        add_intensity = features_node["intensity"].as<bool>();
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
    //show_info();
    //std::cout << "Parameters:\n";
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "config file:\t" << config_file_path << "\n";
    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "\thost file     :\t" << host_file_path << "\n";
    std::cout << "\tquery file    :\t" << query_file_path << "\n";
    std::cout << "\tout file      :\t" << out_file_path << "\n";
    std::cout << "\tneighborhood  :\t" << neighborhood_name << "\n";
    std::cout << "\tdevice        :\t" << neighborhood_device << "\n";
    std::cout << "\tradius        :\t" << neigborhood_range << "\n";
    std::cout << "\tk             :\t" << neighborhood_k << "\n";
    std::cout << "\tbatch         :\t" << neighborhood_batch << "\n";
    std::cout << "\tmax k         :\t" << neighborhood_max_k << "\n";
    std::cout << "\tadd intensity :\t" << add_intensity << "\n";
    std::cout << "\tadd height    :\t" << add_height << "\n";
    std::cout << "\tadd density   :\t" << add_density << "\n";
    std::cout << "------------------------------------------------------------------------\n";
    /*
    bool success =  run(
                        host_file_path,
                        query_file_path,
                        out_file_path,
                        neighborhood_name,
                        neighborhood_device,
                        neigborhood_range,
                        neighborhood_k,
                        neighborhood_batch,
                        neighborhood_max_k,
                        add_intensity,
                        add_height,
                        add_density);
                        
    if (!success)
        return 1;
    */
    return 0;
}
/////////////////////////////////////////////////////////////////////////////////////////
// end of file


