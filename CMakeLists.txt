cmake_minimum_required(VERSION 3.5.0)
project(threed VERSION 2.1.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(${CMAKE_SOURCE_DIR}/definitions.cmake)

# lazperf dirs
set(LAZPERF_MAIN ${PROJECT_SOURCE_DIR}/lazperf/cpp/lazperf)
set(LAZPERF_DETAILS ${PROJECT_SOURCE_DIR}/lazperf/cpp/lazperf/detail)

# cnpy dirs
set(CNPY_MAIN ${PROJECT_SOURCE_DIR}/cnpy)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${LAZPERF_MAIN}
    ${LAZPERF_DETAILS}
    ${CNPY_MAIN}
  )

#find_package(Boost REQUIRED COMPONENTS program_options filesystem REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(nanoflann)
find_package(PCL CONFIG REQUIRED)

# CNPY LIB
add_library(cnpy
          ${CNPY_MAIN}/cnpy.cpp
          )
target_link_libraries(cnpy ZLIB::ZLIB)

file(GLOB LAZPERF_SRCS
    ${LAZPERF_MAIN}/*.cpp
    ${LAZPERF_DETAILS}/*.cpp
)

file(GLOB LAZPERF_HEADERS
    ${LAZPERF_MAIN}/*.hpp
    ${LAZPERF_DETAILS}/*.hpp
)
add_library(lazperf
    ${LAZPERF_SRCS}
)

set(MAIN_SRC 
  ${PROJECT_SOURCE_DIR}/src/file_io.cpp
  ${PROJECT_SOURCE_DIR}/src/neighborhood.cpp
  ${PROJECT_SOURCE_DIR}/src/features.cpp
)

set(MAIN_HEADER
  ${PROJECT_SOURCE_DIR}/include/timer.hpp
  ${PROJECT_SOURCE_DIR}/include/file_io.hpp
  ${PROJECT_SOURCE_DIR}/include/features.hpp
  ${PROJECT_SOURCE_DIR}/include/neighborhood.hpp)

add_executable(${PROJECT_NAME}
  src/main.cpp
  ${MAIN_SRC}
  ${MAIN_HEADER}
  )

#target_compile_settings(${PROJECT_NAME})
target_compile_settings(lazperf)

#target_link_libraries(${PROJECT_NAME} 
#                        PRIVATE Boost::boost Boost::program_options
#                        PRIVATE Boost::boost Boost::filesystem)

target_link_libraries(${PROJECT_NAME} PUBLIC yaml-cpp::yaml-cpp) 
target_link_libraries(${PROJECT_NAME} PUBLIC nanoflann::nanoflann)
target_link_libraries(${PROJECT_NAME} PRIVATE ${PCL_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PUBLIC cnpy)
target_link_libraries(${PROJECT_NAME} PUBLIC lazperf)

# do need to move config files
set (MOVE_CONFIG ON)
if (MOVE_CONFIG)
add_custom_command(
  TARGET ${PROJECT_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy 
    ${CMAKE_CURRENT_SOURCE_DIR}/config.yaml 
    $<TARGET_FILE_DIR:${PROJECT_NAME}>)

endif()
