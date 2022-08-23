#!/usr/bin/env python
#-*- coding: utf-8 -*-

# Python 2/3 compatibility
from __future__ import print_function

# Built-in modules
import os
import sys
import time
import threading
import multiprocessing

# External modules
import cv2
import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ROS modules
PKG = 'lidar_camera_calibration'
import roslib; roslib.load_manifest(PKG)
import rospy
import tf2_ros
import ros_numpy
import image_geometry
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import euler_from_matrix
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from sensor_msgs.msg import Image, CameraInfo, PointCloud2

# Global variables
# PAUSE = False
# FIRST_TIME = True
# KEY_LOCK = threading.Lock()
TF_BUFFER = None
TF_LISTENER = None
CV_BRIDGE = CvBridge()
CAMERA_MODEL = image_geometry.PinholeCameraModel()

# Global paths
PKG_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CALIB_PATH = 'calibration_data/lidar_camera_calibration'

'''
Runs the image point selection GUI process

Inputs:
    img_msg - [sensor_msgs/Image] - ROS sensor image message
    now - [int] - ROS bag time in seconds
    rectify - [bool] - to specify whether to rectify image or not

Outputs:
    Picked points saved in PKG_PATH/CALIB_PATH/img_corners.npy
'''
def extract_points_2D(img_msg, now, rectify=False):
    # Log PID
    rospy.loginfo('2D Picker PID: [%d]' % os.getpid())

    # Read image using CV bridge
    try:
        img = CV_BRIDGE.imgmsg_to_cv2(img_msg, 'bgr8')
    except CvBridgeError as e: 
        rospy.logerr(e)
        return

    # Rectify image
    if rectify: CAMERA_MODEL.rectifyImage(img, img)
    disp = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

    # Setup matplotlib GUI
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Select 2D Image Points - %d' % now.secs)
    ax.set_axis_off()
    ax.imshow(disp)

    # Pick points
    picked, corners = [], []
    def onclick(event):
        x = event.xdata
        y = event.ydata
        if (x is None) or (y is None): return

        # Display the picked point
        picked.append((x, y))
        corners.append((x, y))
        rospy.loginfo('IMG: %s', str(picked[-1]))

        if len(picked) > 1:
            # Draw the line
            temp = np.array(picked)
            ax.plot(temp[:, 0], temp[:, 1])
            ax.figure.canvas.draw_idle()

            # Reset list for future pick events
            del picked[0]

    # Display GUI
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    # Save corner points and image
    # rect = '_rect' if rectify else ''
    # if len(corners) > 1: del corners[-1] # Remove last duplicate
    # save_data(corners, 'img_corners%s.npy' % (rect), CALIB_PATH)
    # save_data(img, 'image_color%s-%d.jpg' % (rect, now.secs), 
    #     os.path.join(CALIB_PATH, 'images'), True)

'''
Runs the LiDAR point selection GUI process

Inputs:
    velodyne - [sensor_msgs/PointCloud2] - ROS velodyne PCL2 message
    now - [int] - ROS bag time in seconds

Outputs:
    Picked points saved in PKG_PATH/CALIB_PATH/pcl_corners.npy
'''
def extract_points_3D(velodyne, now):
    # Log PID
    rospy.loginfo('3D Picker PID: [%d]' % os.getpid())

    # Extract points data
    points = ros_numpy.point_cloud2.pointcloud2_to_array(velodyne)
    points = np.asarray(points.tolist())
    
    # Group all beams together and pick the first 4 columns for X, Y, Z, intensity.
    # if OUSTER_LIDAR: points = points.reshape(-1, 9)[:, :4]

    # Select points within chessboard range
    inrange = np.where((points[:, 0] > 0) &
                       (points[:, 0] < 2.5) &
                       (np.abs(points[:, 1]) < 2.5) &
                       (points[:, 2] < 2))
    points = points[inrange[0]]
    print(points.shape)
    if points.shape[0] > 5:
        rospy.loginfo('PCL points available: %d', points.shape[0])
    else:
        rospy.logwarn('Very few PCL points available in range')
        return

    # Color map for the points
    cmap = matplotlib.cm.get_cmap('hsv')
    colors = cmap(points[:, -1] / np.max(points[:, -1]))

    # Setup matplotlib GUI
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Select 3D LiDAR Points - %d' % now.secs, color='white')
    ax.set_axis_off()
    ax.set_facecolor((0, 0, 0))
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=2, picker=5)

    # Equalize display aspect ratio for all axes
    max_range = (np.array([points[:, 0].max() - points[:, 0].min(), 
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()]).max() / 2.0)
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Pick points
    picked, corners = [], []
    def onpick(event):
        ind = event.ind[0]
        x, y, z = event.artist._offsets3d

        # Ignore if same point selected again
        if picked and (x[ind] == picked[-1][0] and y[ind] == picked[-1][1] and z[ind] == picked[-1][2]):
            return
        
        # Display picked point
        picked.append((x[ind], y[ind], z[ind]))
        corners.append((x[ind], y[ind], z[ind]))
        rospy.loginfo('PCL: %s', str(picked[-1]))

        if len(picked) > 1:
            # Draw the line
            temp = np.array(picked)
            ax.plot(temp[:, 0], temp[:, 1], temp[:, 2])
            ax.figure.canvas.draw_idle()

            # Reset list for future pick events
            del picked[0]

    # Display GUI
    fig.canvas.mpl_connect('pick_event', onpick)
    plt.show()

    # Save corner points
    # if len(corners) > 1: del corners[-1] # Remove last duplicate
    # save_data(corners, 'pcl_corners.npy', CALIB_PATH)


'''
Projects the point cloud on to the image plane using the extrinsics

Inputs:
    img_msg - [sensor_msgs/Image] - ROS sensor image message
    velodyne - [sensor_msgs/PointCloud2] - ROS velodyne PCL2 message
    image_pub - [sensor_msgs/Image] - ROS image publisher

Outputs:
    Projected points published on /sensors/camera/camera_lidar topic
'''
def project_point_cloud(velodyne, img_msg, image_pub):
    # Read image using CV bridge
    try:
        img = CV_BRIDGE.imgmsg_to_cv2(img_msg, 'bgr8')
    except CvBridgeError as e:
        rospy.logerr(e)
        return

    # Transform the point cloud
    try:
        transform = TF_BUFFER.lookup_transform('world', 'velodyne', rospy.Time())
        velodyne = do_transform_cloud(velodyne, transform)
    except tf2_ros.LookupException:
        pass

    # Extract points from message
    points3D = ros_numpy.point_cloud2.pointcloud2_to_array(velodyne)
    points3D = np.asarray(points3D.tolist())
    
    # Filter points in front of camera
    inrange = np.where((points3D[:, 2] > 0) &
                       (points3D[:, 2] < 6) &
                       (np.abs(points3D[:, 0]) < 6) &
                       (np.abs(points3D[:, 1]) < 6))
    max_intensity = np.max(points3D[:, -1])
    points3D = points3D[inrange[0]]

    # Color map for the points
    cmap = matplotlib.cm.get_cmap('jet')
    colors = cmap(points3D[:, -1] / max_intensity) * 255

    # Project to 2D and filter points within image boundaries
    points2D = [ CAMERA_MODEL.project3dToPixel(point) for point in points3D[:, :3] ]
    points2D = np.asarray(points2D)
    inrange = np.where((points2D[:, 0] >= 0) &
                       (points2D[:, 1] >= 0) &
                       (points2D[:, 0] < img.shape[1]) &
                       (points2D[:, 1] < img.shape[0]))
    points2D = points2D[inrange[0]].round().astype('int')

    # Draw the projected 2D points
    for i in range(len(points2D)):
        cv2.circle(img, tuple(points2D[i]), 2, tuple(colors[i]), -1)

    # Publish the projected points image
    try:
        image_pub.publish(CV_BRIDGE.cv2_to_imgmsg(img, "bgr8"))
    except CvBridgeError as e: 
        rospy.logerr(e)


'''
Callback function to publish project image and run calibration

Inputs:
    image - [sensor_msgs/Image] - ROS sensor image message
    camera_info - [sensor_msgs/CameraInfo] - ROS sensor camera info message
    velodyne - [sensor_msgs/PointCloud2] - ROS velodyne PCL2 message
    image_pub - [sensor_msgs/Image] - ROS image publisher

Outputs: None
'''
def callback(image, camera_info, velodyne, image_pub=None):
    global CAMERA_MODEL, FIRST_TIME, PAUSE, TF_BUFFER, TF_LISTENER

    rospy.loginfo('Setting up camera model')
    CAMERA_MODEL.fromCameraInfo(camera_info)

    # TF listener
    TF_BUFFER = tf2_ros.Buffer()
    TF_LISTENER = tf2_ros.TransformListener(TF_BUFFER)

    # Projection/display mode
    project_point_cloud(velodyne, image, image_pub)

    # Create GUI processes
    now = rospy.get_rostime()
    img_p = multiprocessing.Process(target=extract_points_2D, args=[image, now])
    pcl_p = multiprocessing.Process(target=extract_points_3D, args=[velodyne, now])
    img_p.start(); pcl_p.start()
    img_p.join(); pcl_p.join()


'''
The main ROS node which handles the topics

Inputs:
    camera_info - [str] - ROS sensor camera info topic
    image_color - [str] - ROS sensor image topic
    velodyne - [str] - ROS velodyne PCL2 topic
    camera_lidar - [str] - ROS projected points image topic

Outputs: None
'''
def listener(camera_info, image_color, velodyne_points):
    # Start node
    rospy.init_node('projection_camera_lidar', anonymous=True)
    rospy.loginfo('Current PID: [%d]' % os.getpid())
    rospy.loginfo('CameraInfo topic: %s' % camera_info)
    rospy.loginfo('Image topic: %s' % image_color)
    rospy.loginfo('PointCloud2 topic: %s' % velodyne_points)

    # Subscribe to topics
    info_sub = message_filters.Subscriber(camera_info, CameraInfo)
    image_sub = message_filters.Subscriber(image_color, Image)
    velodyne_sub = message_filters.Subscriber(velodyne_points, PointCloud2)

    # Publish output topic
    image_pub = rospy.Publisher('camera_lidar', Image, queue_size=5)

    # Synchronize the topics by time
    ats = message_filters.ApproximateTimeSynchronizer(
        [image_sub, info_sub, velodyne_sub], queue_size=5, slop=0.1)
    ats.registerCallback(callback, image_pub)

    # Keep python from exiting until this node is stopped
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo('Shutting down')

if __name__ == '__main__':

    # Camera, LiDAR Topic name
    camera_info = '/usb_cam/camera_info'
    image_color = '/usb_cam/image_raw'
    velodyne_points = '/velodyne_points'

    # Start subscriber
    listener(camera_info, image_color, velodyne_points)
