cmake_minimum_required(VERSION 3.0.2)
project(gridbased_dmpc)

find_package(catkin REQUIRED COMPONENTS
    actionlib_msgs
    geometry_msgs
    rospy
    std_msgs
    nav_msgs
    message_generation
)

catkin_python_setup()

 add_action_files(
   FILES
   AngleTargetPoint.action
   OSCSTarget.action
 )

add_message_files(
   FILES
   Edge.msg
   OSCS.msg
   QEllipse.msg
 )

 add_service_files(
    FILES
    EdgesList.srv
    OSCSList.srv
)

 generate_messages(
   DEPENDENCIES
   std_msgs
   geometry_msgs
   nav_msgs
   actionlib_msgs
 )

catkin_package(
  INCLUDE_DIRS src scripts
  CATKIN_DEPENDS rospy
)

install(
  DIRECTORY src/
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
  USE_SOURCE_PERMISSIONS
)
install(DIRECTORY worlds/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/worlds
  USE_SOURCE_PERMISSIONS
)
install(DIRECTORY models/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/models
  USE_SOURCE_PERMISSIONS
)

include_directories(
 ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
   scripts/global_planner.py
   scripts/local_planner.py
   scripts/local_gb_mpc.py
   scripts/fake_scheduler.py
   scripts/offline_set_computation.py
   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
