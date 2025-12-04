roslaunch custom_bringup custom_bringup.launch

rosrun custom_bringup movement_test.py

roslaunch transbot_nav transbot_map.launch map_type:=gmapping 