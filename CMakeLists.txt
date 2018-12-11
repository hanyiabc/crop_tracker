cmake_minimum_required (VERSION 2.6)
set (CMAKE_CXX_STANDARD 11)
project (Crop_Tracker)

find_package (OpenCV REQUIRED)

add_executable(Crop_Tracker crop_tracker.cpp )
target_link_libraries( Crop_Tracker ${OpenCV_LIBS} )
