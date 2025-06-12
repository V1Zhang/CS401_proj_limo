#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np
from math import *
from geometry_msgs.msg import Pose

class lane_converter:
    def __init__(self):    
        # 创建cv_bridge，声明图像的发布者和订阅者
        self.image_pub = rospy.Publisher("lane_detect_image", Image, queue_size=1)
        self.target_pub = rospy.Publisher("lane_detect_pose", Pose,  queue_size=1)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/image", Image, self.callback)
    def callback(self,data):
        # 使用cv_bridge将ROS的图像数据转换成OpenCV的图像格式
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print e
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        lower_yellow = np.array([5, 80, 100])
        upper_yellow = np.array([28, 200, 255])
        
        kernel = np.ones((5,5),np.uint8)
        
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        #cv2.imshow("lane",mask)
        #cv2.waitKey()

        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
        color_x = mask[400,0:400]
        color_y = mask[400:480,300:340]
        color_sum = mask
        white_count_x = np.sum(color_x == 255)
        white_count_y = np.sum(color_y == 255)
        white_count_sum = np.sum(color_sum == 255)
        #print('white_count_x:', white_count_x)
        #print('white_count_y:', white_count_y)

        white_index_x = np.where(color_x == 255)
        white_index_y = np.where(color_y == 255)
        #print('white_index_x:',white_index_x)
        #print('white_index_y:',white_index_y)

        if white_count_y == 0:
            center_y = 240
        else :
            center_y = (white_index_y[0][white_count_y - 2] + white_index_y[0][0]) / 2
            center_y = center_y +340
        if white_count_x == 0:
            center_x = 195
        else:
            center_x = (white_index_x[0][white_count_x - 2] + white_index_x[0][0]) / 2
        
        
        objPose = Pose()
        objPose.position.x = center_x;
        objPose.position.y = center_y;
        objPose.position.z = white_count_sum;
        #objPose.orientation.y = white_count_y;
        self.target_pub.publish(objPose)
        #print(objPose)

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(mask, "mono8"))
        except CvBridgeError as e:
            print e

if __name__ == '__main__':
    try:
        # 初始化ros节点
        rospy.init_node("yolo_detect_lane", anonymous=True)
        rospy.loginfo("Starting yolo lane object")
        lane_converter()
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down yolo_lane_detect node."
        cv2.destroyAllWindows()

