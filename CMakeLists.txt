cmake_minimum_required(VERSION 3.15)
project(line_tracker)


if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)

set(GLOG_PREFER_EXPORTED_GLOG_CMAKE_CONFIGURATION TRUE)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wall -O3")

find_package(Glog REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(${PROJECT_NAME} SHARED "")

target_include_directories(${PROJECT_NAME} PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    $<BUILD_INTERFACE: ${CMAKE_CURRENT_LIST_DIR}/include>)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${GLOG_LIBRARIES})
add_subdirectory(src)

add_executable(test_line_tracker test/test_line_tracker.cpp)
target_link_libraries(test_line_tracker ${PROJECT_NAME})