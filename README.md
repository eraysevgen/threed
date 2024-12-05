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
- [lazperf](https://github.com/hobuinc/laz-perf)
- [cnpy](https://github.com/rogersce/cnpy)
- [nanoflann](https://github.com/jlblancoc/nanoflann)
- [pcl](https://pointclouds.org/)
- [yaml-cpp](https://github.com/jbeder/yaml-cpp)
- [boost](https://www.boost.org/) [program_options, filesystem]
- [Eigen](https://gitlab.com/libeigen/eigen) 

[vcpkg](https://vcpkg.io/en/) package manager is used for dependency management. 

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

{% raw %}
@article{Sevgen_Abdikan_2023_a,
  author = {Sevgen, Eray and Abdikan, Saygin},
  title = {Classification of Large-Scale Mobile Laser Scanning Data in Urban Area with LightGBM},
  journal = {Remote Sensing},
  volume = {15},
  year = {2023},
  number = {15},
  article-number = {3787},
  url = {https://www.mdpi.com/2072-4292/15/15/3787},
  issn = {2072-4292},
  doi = {10.3390/rs15153787},
}
{% endraw %}

{% raw %}
@article{Sevgen_Abdikan_2023_b,
  author = {Sevgen, E. and Abdikan, S.},
  title = {POINT-WISE CLASSIFICATION OF HIGH-DENSITY UAV-LIDAR DATA USING GRADIENT BOOSTING MACHINES},
  journal = {The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
  volume = {XLVIII-M-1-2023},
  year = {2023},
  pages = {587--593},
  url = {https://isprs-archives.copernicus.org/articles/XLVIII-M-1-2023/587/2023/},
  doi = {10.5194/isprs-archives-XLVIII-M-1-2023-587-2023},
}
{% endraw %}


## License
See LICENCE file. 
MIT