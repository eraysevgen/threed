cmake_minimum_required(VERSION 3.15)
project(threed_binding LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------
# dependencies
# -----------------------------------------------------------------------
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)

# -----------------------------------------------------------------------
# threed_core_py
# -----------------------------------------------------------------------
add_library(threed_core_py STATIC 
    ../src/features.cpp 
    ../src/cpu_neighborhood.cpp 
)

set_target_properties(threed_core_py PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(threed_core_py PUBLIC . ../include/)
target_link_libraries(threed_core_py PUBLIC Eigen3::Eigen)

if(MSVC)
    target_compile_options(threed_core_py PRIVATE /O2 /arch:AVX2)
else()
    target_compile_options(threed_core_py PRIVATE -O3 -march=native)
endif()

# -----------------------------------------------------------------------
# threed_py
# -----------------------------------------------------------------------
nanobind_add_module(threed_py bindings.cpp)
target_link_libraries(threed_py PRIVATE threed_core_py)
target_compile_definitions(threed_py PRIVATE NB_STABLE_ABI)

install(TARGETS threed_py DESTINATION .)