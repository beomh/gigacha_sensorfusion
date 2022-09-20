#!/usr/bin/env python
#-*- coding: utf-8 -*-

# Python 2/3 compatibility
from __future__ import print_function
import rospy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

class Imageeditor:
    def __init__(self):
        # make node
        rospy.init_node("Image_Editor", anonymous=False)
        # Subscriber
        rospy.Subscriber("/usb_cam/image_raw", Image, self.img_callback)
        rospy.Subscriber("/sign_bbox", Detection2DArray, self.yolo_callback)
        # Publisher
        self.img_pub = rospy.Publisher("/usb_cam/image_refine", Image, queue_size=1)
        # instance variable
        self.t = None
        rospy.spin()

    def yolo_callback(self, msg):
        self.t = msg.header.stamp

    def img_callback(self, msg):
        img = Image()
        img.header.seq = msg.header.seq
        img.header.stamp = self.t
        img.header.frame_id = 'camera'
        img.height = msg.height
        img.width = msg.width
        img.encoding = msg.encoding
        img.is_bigendian = msg.is_bigendian
        img.step = msg.step
        img.data = msg.data
        rospy.loginfo('Refining')
        self.img_pub.publish(img)


if __name__ == "__main__":
    try:
        ie = Imageeditor()
    except rospy.ROSInitException:
        pass