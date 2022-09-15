// ROS includes
#include "image_geometry/pinhole_camera_model.h"
#include <boost/foreach.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/console.h>
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/String.h>
#include <tf/transform_listener.h>

// OpenCV includes
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

// Include pcl
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>

// Include PointCloud2 ROS message
#include "pcl_ros/impl/transforms.hpp"
#include "pcl_ros/transforms.h"
#include <pcl/io/io.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud2.h>

#include <iostream>
#include <string>

#define IMAGE_WIDTH 640
#define IMAGE_HEIGHT 480

using namespace std;

// Topics
// static const std::string LIDAR_TOPIC = "pass_through_cloud";
static const std::string LIDAR_TOPIC = "/velodyne_points";
// static const std::string IMG_TOPIC = "camera/image_rect_color";
static const std::string IMG_TOPIC = "/usb_cam/image_raw";
// static const std::string CAMERA_INFO = "camera/camera_info";
static const std::string CAMERA_INFO = "/usb_cam/camera_info";
static const std::string COMPOSITE_IMG_OUT = "image_in_point_cloud"; // lidar points on camera image
static const std::string OPENCV_WINDOW = "LiDAR and Camera Calibration";

class ImageConverter 
{
  ros::NodeHandle nh;

  ros::Subscriber image_sub_;
  ros::Subscriber info_sub_;
  ros::Subscriber lidar_sub;
  ros::Time cloudHeader;

  image_transport::ImageTransport it_;
  image_transport::Publisher image_pub_;
  image_geometry::PinholeCameraModel cam_model_;

  const tf::TransformListener tf_listener_;
  tf::StampedTransform transform;

  std::vector<tf::Vector3> objectPoints;
  tf::Vector3 pt_cv;
  std::vector<cv::Point3d> pt_transformed;

public:
  ImageConverter() : it_(nh) 
  {
    //subscriber
    lidar_sub = nh.subscribe(LIDAR_TOPIC, 1, &ImageConverter::lidarCb, this);
    image_sub_ = nh.subscribe(IMG_TOPIC, 1, &ImageConverter::imageCb, this);
    info_sub_ = nh.subscribe(CAMERA_INFO, 1, &ImageConverter::cameraCallback, this);
   
    //publisher
    image_pub_ = it_.advertise(COMPOSITE_IMG_OUT, 1);

    //cv::namedWindow(OPENCV_WINDOW);
  }
  ~ImageConverter() 
  { 
    //cv::destroyWindow(OPENCV_WINDOW); 
  }

  void cameraCallback(const sensor_msgs::CameraInfoConstPtr &info_msg) 
  {
    cam_model_.fromCameraInfo(info_msg);
  }

  void imageCb(const sensor_msgs::ImageConstPtr &image_msg) 
  {
    cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);
    //std::cout<<image_msg.Header.stamp<<"\n";
    std_msgs::Header h = image_msg -> header;
    // cout << h.frame_id << endl;
    //std::cout<<h.stamp.sec<<"."<<h.stamp.nsec<<"\n";
    try 
    {
      cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception &e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    ros::Time acquisition_time = ros::Time(0);
    ros::Duration timeout(2.0);
    std::cerr<<timeout<<"\n";
    // try {
    //   //	ros::Time acquisition_time = cloudHeader;
    //   // ros::Duration timeout(1.0 / 30);
    //   tf_listener_.waitForTransform("usb_cam", "velodyne", ros::Time(0), timeout);
    //   tf_listener_.lookupTransform("usb_cam", "velodyne", acquisition_time, transform);
    //   // tf_listener_.waitForTransform("camera", "velodyne", ros::Time(0), timeout);
    //   // tf_listener_.lookupTransform("camera", "velodyne", acquisition_time, transform);
    // } catch (tf::TransformException &ex) 
    // {
    //   ROS_ERROR("%s", ex.what());
    //   //std::cerr<<"NO";
    //   // ros::Duration(1.0).sleep();
    // }

    // tranform the xyz point from pointcoud -> 이미지에 표시
    // int cnt = 0;
    for (size_t i = 0; i < objectPoints.size(); ++i) 
    {
      // cout << objectPoints.size() << "\n";
      // pt_cv = transform(objectPoints[i]);
      //input my transform function
      // cout << "objectPoints: [" << objectPoints[i].getX() << ", " << objectPoints[i].getY() << ", " << objectPoints[i].getZ() << "]\n";
      double A[3][4] = {{0.0, -1.0, 0.0, 0.0}, {0.0, 0.0, -1.0, 0.0}, {1.0, 0.0, 0.0, -0.2}};  // rot_mat & translation
      double B[4][1] = {{objectPoints[i].getX()}, {objectPoints[i].getY()}, {objectPoints[i].getZ()}, {1.0}};  // point mat
      double C[3][1];


      int A_col = sizeof(A[0]) / sizeof(double);
      int A_row = sizeof(A) / sizeof(A[0]);
      int B_col = sizeof(B[0]) / sizeof(double);
      int B_row = sizeof(B) / sizeof(B[0]);
      //cout << A_col << " " << A_row << " " << B_col << " " << B_row << "\n";

      double sum;
      for (int r = 0; r < A_row; r++) {
		    for (int c = 0; c < B_col; c++) {
			    sum = 0.0;
			    for (int k = 0; k < A_col; k++) {
				    sum += A[r][k] * B[k][c];
			    }
			    C[r][c] = sum;
		    }
	    }

      // // 결과 행렬 출력
      // for (int r = 0; r < A_row; r++) {
      //   for (int c = 0; c < B_col; c++) {
      //     cout << C[r][c] << " ";
      //   }
      //   cout << endl;
      // }

      pt_cv.setX(C[0][0]);
      pt_cv.setY(C[1][0]);
      pt_cv.setZ(C[2][0]);
      //cout << "[" << pt_cv.getX() << ", " << pt_cv.getY() << ", " << pt_cv.getZ() << "]" << "\n";
      

      pt_transformed.push_back(cv::Point3d(pt_cv.x(), pt_cv.y(), pt_cv.z()));
      // cout << pt_transformed[i] << endl;
      cv::Point2d uv;
      uv = cam_model_.project3dToPixel(pt_transformed[i]);
      // cout << uv.x << ", " << uv.y << "\n";

      if (uv.x >= 0 && uv.x <= IMAGE_WIDTH && uv.y >= 0 && uv.y <= IMAGE_HEIGHT) 
      {
        static const int RADIUS = 2; 
        cv::circle(cv_ptr->image, uv, RADIUS, CV_RGB(255, 0, 0), -1); //이미지에 투영
        cout << "draw circle\n";
      }
    }
    // cout << "projection count: " << cnt << "\n";
    pt_cv.setZero();
    pt_transformed.clear();
    cv::imshow(OPENCV_WINDOW, cv_ptr->image);
    int code = -1;
    code = cv::waitKey(1);
    //ROS_INFO_STREAM("code:  " <<code);
    //cv::waitKey(1);

    sensor_msgs::ImagePtr convertedMsg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cv_ptr->image).toImageMsg();
    image_pub_.publish(convertedMsg);
  }
  
  //라이다 콜백 함수
  void lidarCb(const sensor_msgs::PointCloud2ConstPtr &pointCloudMsg) {
    cloudHeader = pointCloudMsg->header.stamp;

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>);

    pcl::fromROSMsg(*pointCloudMsg, *cloud);

    // Create the  filtering object
    pcl::PassThrough<pcl::PointXYZI> pass_x;
    pass_x.setInputCloud(cloud);
    pass_x.setFilterFieldName("x");
    pass_x.setFilterLimits(0.0, 50); // 0 to 4.5 mts limitation
    //pass_x.setFilterLimits(-50.0, 50); // 0 to 4.5 mts limitation
    pass_x.filter(*cloud_filtered);

    objectPoints.clear();
    for (size_t i = 0; i < cloud->size(); ++i)
    {
      objectPoints.push_back(tf::Vector3(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z));
      // ROS_INFO_STREAM("X: " << objectPoints[i].x() << "  Y: "<< objectPoints[i].y() << "  Z: "<< objectPoints[i].z());
    }
  }
};

int main(int argc, char **argv) 
{
  ros::init(argc, argv, "from_3d_to_2d");
  ImageConverter ic;
  ros::spin();
  return 0;
}
