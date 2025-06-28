cmake_minimum_required(VERSION 3.0.2)
project(empty_space_estimation)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  message_generation
)

add_message_files(
    FILES
    EmptySpace.msg
)

add_service_files(  
  FILES
    EmptySpaceService.srv
)

generate_messages(
  DEPENDENCIES
    std_msgs
    geometry_msgs
    sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS
    roscpp
    rospy
    std_msgs
    geometry_msgs
    sensor_msgs
    message_runtime
)


include_directories(
  ${catkin_INCLUDE_DIRS}
)