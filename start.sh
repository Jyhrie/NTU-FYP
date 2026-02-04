#!/bin/bash

echo "Starting Bringup..."
roslaunch custom_bringup custom_bringup.launch &
sleep 3

echo "Starting Rosbridge..."
roslaunch rosbridge_server rosbridge_websocket.launch &
sleep 3

echo "Starting Gmapping..."
roslaunch custom_bringup gmapping.launch