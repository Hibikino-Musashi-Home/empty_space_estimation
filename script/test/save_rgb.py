#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from datetime import datetime

class SaveImage():
    def __init__(self):
        
        bridge = CvBridge()

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")  # 例: 20250421_174530
        filename = f"saved_image_{timestamp}.jpg"

        # 画像メッセージを1回だけ取得
        rospy.loginfo("Waiting for image message...")
        # msg = rospy.wait_for_message("/hsrb/head_rgbd_sensor/rgb/image_raw", Image)
        msg = rospy.wait_for_message("/camera/rgb/image_raw", Image)
        rospy.loginfo("Image received.")

        img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv2.imwrite("/home/hma/Pictures/sakakibara/" + filename, img)
        print(f"/home/hma/Pictures/sakakibara/{filename}")
        rospy.loginfo(f"Image saved as {filename}")


if __name__ == "__main__":
    rospy.init_node('sub_image')
    si = SaveImage()
    rospy.spin()