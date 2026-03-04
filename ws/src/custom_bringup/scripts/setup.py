#!/usr/bin/env python2

# -*- coding: utf-8 -*-
"""
engine_converter_node.py
ROS1 Melodic node — listens for a std_msgs/String signal on /convert_to_engine,
then runs trtexec to convert the .onnx to a FP16 .engine file in-place.

Topic subscribed : /convert_to_engine   (std_msgs/String)
  message data   : absolute path to the .onnx file
                   e.g. "/home/jetson/fyp/ws/src/custom_bringup/scripts/models/best.onnx"

Topic published  : /engine_ready        (std_msgs/String)
  message data   : absolute path to the generated .engine file (on success)
                   OR "ERROR: <reason>"  (on failure)
"""

import rospy
import subprocess
import os
from std_msgs.msg import String

# Path to trtexec — adjust if yours is elsewhere
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"


class EngineConverterNode(object):

    def __init__(self):
        rospy.init_node("engine_converter_node", anonymous=False)

        self.pub = rospy.Publisher("/engine_ready", String, queue_size=1)
        self.sub = rospy.Subscriber("/convert_to_engine", String,
                                    self.convert_callback, queue_size=1)

        rospy.loginfo("[EngineConverter] Node ready. Waiting on /convert_to_engine ...")
        rospy.spin()

    # ──────────────────────────────────────────────────────────────────────────

    def convert_callback(self, msg):
        onnx_path = msg.data.strip()
        rospy.loginfo("[EngineConverter] Received conversion request: %s", onnx_path)

        # ── Validate ──────────────────────────────────────────────────────────
        if not os.path.isfile(onnx_path):
            err = "ERROR: .onnx file not found: {}".format(onnx_path)
            rospy.logerr("[EngineConverter] %s", err)
            self.pub.publish(err)
            return

        if not os.path.isfile(TRTEXEC):
            err = "ERROR: trtexec not found at {}".format(TRTEXEC)
            rospy.logerr("[EngineConverter] %s", err)
            self.pub.publish(err)
            return

        engine_path = os.path.splitext(onnx_path)[0] + ".engine"

        # ── Build trtexec command ─────────────────────────────────────────────
        cmd = [
            TRTEXEC,
            "--onnx={}".format(onnx_path),
            "--saveEngine={}".format(engine_path),
            "--fp16",
            "--workspace=2048",   # MB — safe default for Jetson Nano 4 GB
            "--verbose",
        ]

        rospy.loginfo("[EngineConverter] Running trtexec (FP16)...")
        rospy.loginfo("[EngineConverter] Command: %s", " ".join(cmd))

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )

            # Stream trtexec output to roslog
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    rospy.loginfo("[trtexec] %s", line)

            proc.wait()

            if proc.returncode == 0 and os.path.isfile(engine_path):
                rospy.loginfo("[EngineConverter] Conversion successful: %s", engine_path)
                self.pub.publish(engine_path)
            else:
                err = "ERROR: trtexec exited with code {}".format(proc.returncode)
                rospy.logerr("[EngineConverter] %s", err)
                self.pub.publish(err)

        except Exception as e:
            err = "ERROR: {}".format(str(e))
            rospy.logerr("[EngineConverter] %s", err)
            self.pub.publish(err)


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        EngineConverterNode()
    except rospy.ROSInterruptException:
        pass