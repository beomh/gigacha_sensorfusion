#!/usr/bin/env python
#-*-coding:utf-8-*-

# Python 2/3 compatibility
from __future__ import print_function

# Built-in modules
import os
import sys
import math
import time
import multiprocessing

# External modules
import cv2
import numpy as np
from numpy.linalg import inv

# ROS module
import rospy
import message_filters
import tf2_ros
import ros_numpy
import image_geometry
from cv_bridge import CvBridge, CvBridgeError
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, ChannelFloat32
from vision_msgs.msg import Detection2DArray, Detection2D
from geometry_msgs.msg import Pose, PoseArray


def point_in_triangle(p, v1, v2, v3):
    """Checks whether a point is within the given triangle

    The function checks, whether the given point p is within the triangle defined by the the three corner point v1,
    v2 and v3.
    This is done by checking whether the point is on all three half-planes defined by the three edges of the triangle.
    :param p: The point to be checked (tuple with x any y coordinate)
    :param v1: First vertex of the triangle (tuple with x any y coordinate)
    :param v2: Second vertex of the triangle (tuple with x any y coordinate)
    :param v3: Third vertex of the triangle (tuple with x any y coordinate)
    :return: True if the point is within the triangle, False if not
    """
    def _test(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    b1 = _test(p, v1, v2) < 0.0
    b2 = _test(p, v2, v3) < 0.0
    b3 = _test(p, v3, v1) < 0.0

    return (b1 == b2) and (b2 == b3) 

params_lidar = {
    "X": 0.0, # meter
    "Y": 0.0,
    "Z": 0.0,
    "YAW": 0.0, # deg
    "PITCH": 0.0,
    "ROLL": 0.0
}

params_cam = {
    "WIDTH": 640, # image width
    "HEIGHT": 480, # image height
    "FOV": 60, # Field of view
    "X": 0.36, # meter
    "Y": -0.15,
    "Z": 0,
    "YAW": 0.0, # deg
    "PITCH": 0.0,
    "ROLL": 0.0
}

def getRotMat(RPY):
    cosR = math.cos(RPY[0])
    cosP = math.cos(RPY[1])
    cosY = math.cos(RPY[2])
    sinR = math.sin(RPY[0])
    sinP = math.sin(RPY[1])
    sinY = math.sin(RPY[2])
    
    rotRoll = np.array([1,0,0, 0,cosR,-sinR, 0,sinR,cosR]).reshape(3,3)
    rotPitch = np.array([cosP,0,sinP, 0,1,0, -sinP,0,cosP]).reshape(3,3)
    rotYaw = np.array([cosY,-sinY,0, sinY,cosY,0, 0,0,1]).reshape(3,3)
    
    rotMat = rotYaw.dot(rotPitch.dot(rotRoll))
    return rotMat

def getTransformMat(params_lidar, params_cam):
    #With Respect to Vehicle ISO Coordinate
    lidarPosition = np.array([params_lidar.get(i) for i in (["X","Y","Z"])])
    # print(lidarPosition)
    camPosition = np.array([params_cam.get(i) for i in (["X","Y","Z"])])
    # print(camPosition)

    lidarRPY = np.array([params_lidar.get(i) for i in (["ROLL","PITCH","YAW"])])
    # print(lidarRPY)
    camRPY = np.array([params_cam.get(i) for i in (["ROLL","PITCH","YAW"])])
    # print(camRPY)
    camRPY = camRPY + np.array([-90*math.pi/180,0,-90*math.pi/180])
    # print(camRPY)

    camRot = getRotMat(camRPY)
    print(camRot)
    camTransl = np.array([camPosition])
    Tr_cam_to_vehicle = np.concatenate((camRot,camTransl.T),axis = 1)
    print(Tr_cam_to_vehicle)
    Tr_cam_to_vehicle = np.insert(Tr_cam_to_vehicle, 3, values=[0,0,0,1],axis = 0)

    lidarRot = getRotMat(lidarRPY)
    print(lidarRot)
    lidarTransl = np.array([lidarPosition])
    Tr_lidar_to_vehicle = np.concatenate((lidarRot,lidarTransl.T),axis = 1)
    print(Tr_lidar_to_vehicle)
    Tr_lidar_to_vehicle = np.insert(Tr_lidar_to_vehicle, 3, values=[0,0,0,1],axis = 0)

    invTr = inv(Tr_cam_to_vehicle)
    # print(invTr)
    Tr_lidar_to_cam = invTr.dot(Tr_lidar_to_vehicle).round(6)
    print(Tr_lidar_to_cam)
    return Tr_lidar_to_cam

a = [1, 1]
b = [5, 1]
c = [3, 5]
t = [3, 3]

print(point_in_triangle(t, a, c, b))