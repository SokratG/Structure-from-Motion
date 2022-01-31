cmake_minimum_required(VERSION 3.16)

project(SfM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV 4.2 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)

file(GLOB SRCSFM "src/*.c*")

include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIRS})

add_definitions(-D_USE_MATH_DEFINES)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

#build exe
add_executable(${PROJECT_NAME} ${SRCSFM})

#link lib 
target_link_libraries(${PROJECT_NAME} rt)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} ${CERES_LIBRARIES})
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})