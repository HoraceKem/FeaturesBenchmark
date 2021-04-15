cmake_minimum_required(VERSION 3.16)
project(FeaturesBenchmark)

find_package(PkgConfig)
pkg_search_module(CV REQUIRED opencv4)
pkg_search_module(JSON REQUIRED jsoncpp)

message(STATUS "=== OpenCV_LIBRARIES: ${CV_LIBRARIES}")
message(STATUS "=== OpenCV_INCLUDE_DIRS: ${CV_INCLUDE_DIRS}")
message(STATUS "=== JsonCPP_LIBRARIES: ${JSON_LIBRARIES}")
message(STATUS "=== JsonCPP_INCLUDE_DIRS: ${JSON_INCLUDE_DIRS}")

include_directories(${CV_INCLUDE_DIRS} ${JSON_INCLUDE_DIRS})
link_directories(${CV_LIBRARY_DIRS} ${JSON_LIBRARY_DIRS})

set(CMAKE_CXX_STANDARD 14)

add_executable(FeaturesBenchmark main.cpp featuresExtractor.cpp featuresExtractor.h featuresMatching.cpp featuresMatching.h usedStruct.h featuresParams.h runParams.h jsonIO.h jsonIO.cpp inlierEstimation.cpp inlierEstimation.h)
target_link_libraries(FeaturesBenchmark ${CV_LIBRARIES} ${JSON_LIBRARIES})