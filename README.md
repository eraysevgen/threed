# threed
A feature extraction program for 3D point clouds

## Usage
Prepare or modify config.yaml file.
```yaml
files:
	host_file: [host_file.las]
	query_file: [query_file.las]
	out_file: [output_features.npy]
neighborhood:
	...


```
Run threed.exe.

```bash
 threed.exe [option]
	Allowed option:
  	-h [ --help ]                      produce help message
  	-c [ --config ] arg (=config.yaml) set the config file path, default config.yaml
  	-v [ --version ]                   show version info
  	-d [ --desc ]                      show program description
```
# Dependencies
- lazperf for reading las files.
- cnpy for writing npy files.
- nanoflann for neighborhood.
- pcl for neighborhood.
- yaml-cpp for yaml parsing.
- boost [program_options, filesystem]
- Eigen 

vcpkg package manager is used for dependency management. 

## Compile

cmake  build

## Release
Download the release, unzip and run threed.exe.

Compiled and tested on Win10 machine.

## Properties
- Inputs are las file, and the outputs are npy files.
- Computes those features:
	- Linearity
	- Sphericity
	- .. 
- Optionally height features can be added with add_height flag in config.yaml
- Optionally ply output can be set with ply_file item in config.yaml
- The features are stored in [out_features].metadata text file.
- The [out_features].npy can be used directly in python machine learning and deep learning applications.

## Cite
If you use this repo, please cite our articles:

## License
MIT