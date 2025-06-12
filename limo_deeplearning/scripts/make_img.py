#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np
from math import *
from geometry_msgs.msg import Pose
import time
from yolo.msg import identify

class lane_converter:
    def __init__(self):    
        # 创建cv_bridge，声明图像的发布者和订阅者
        self.image_pub = rospy.Publisher("lane_detect_image", Image, queue_size=1)
        
        self.bridge = CvBridge()
        #self.identify = rospy.Subscriber("/identify_info", identify, queue_size=10)
        self.identify = rospy.Subscriber("/identify_info", identify, self.ses)
        #global same
        print(self.identify)
    def ses(self,identify):
        i = identify.same
        print(i)
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # set update frequence
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        ret,frame = cap.read()
        print(frame)
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
        except CvBridgeError as e:
            print (e)
        
    '''def callback(self,data):
        start = time.clock()
        print(start)       
        # 使用cv_bridge将ROS的图像数据转换成OpenCV的图像格式
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print (e)
        #save_path 
        cv2.imwrite('0.jpg',cv_image)
        #cv2.imwrite('..//scripts//test//0.jpg',cv_image)
        end = time.clock
        print(end)
        #print('Running time: %s Seconds'%(end-start))
        '''
if __name__ == '__main__':
    try:
        # 初始化ros节点
        rospy.init_node("make_image", anonymous=True)
        rospy.loginfo("Starting make_image")
        lane_converter()
        rospy.spin()
    except KeyboardInterrupt:
        print ("Shutting down lane_detect node.")
        cv2.destroyAllWindows()
