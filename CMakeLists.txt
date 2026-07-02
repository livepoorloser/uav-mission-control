# 功能：drone_code ROS 包的编译目标、依赖和安装配置。

cmake_minimum_required(VERSION 3.0.2)
project(drone_code)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  mavros_msgs
  nav_msgs
  roscpp
  roslib
  sensor_msgs
  std_msgs
)

find_package(OpenCV REQUIRED COMPONENTS core dnn highgui imgproc videoio)

catkin_package(
  CATKIN_DEPENDS geometry_msgs mavros_msgs nav_msgs roscpp roslib sensor_msgs std_msgs
)


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/animal_detector_rknn.py
  scripts/analyze_vision_servo_bag.py
  scripts/camera_calibration.py
  scripts/convert_onnx_to_rknn.py
  scripts/color_block_detector.py
  scripts/rknn_folder_infer.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

add_executable(mission_controller_node
  src/mission_controller.cpp
  src/detection_deduplicator.cpp
  src/detection_processor.cpp
  src/grid_detection_filter.cpp
  src/trajectory_optimizer/trajectory_optimizer.cpp
  src/vision_servo_controller.cpp
)
target_link_libraries(mission_controller_node ${catkin_LIBRARIES})

add_executable(path_planning_node src/path_planning.cpp)
target_link_libraries(path_planning_node ${catkin_LIBRARIES})

add_executable(animal_detector_node src/animal_detector.cpp)
target_link_libraries(animal_detector_node ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(grid_projection_filter_demo src/grid_projection_filter_demo.cpp)
target_link_libraries(grid_projection_filter_demo ${catkin_LIBRARIES})
