roslaunch custom_bringup custom_bringup.launch

roslaunch custom_bringup map_and_nav.launch

rosrun custom_bringup movement_test.py

rosrun custom_bringup custom_mapping.py

roslaunch transbot_nav transbot_map.launch map_type:=gmapping 

rosrun teleop_twist_keyboard teleop_twist_keyboard.py 

python3 ~/fyp/ws/src/custom_bringup/scripts/flaskapp.py