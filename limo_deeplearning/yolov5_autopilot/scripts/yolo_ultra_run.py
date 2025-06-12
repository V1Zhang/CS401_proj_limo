#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import time
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np
from math import *
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from limo_deeplearning.msg import traffic
from limo_deeplearning.msg import identify
 
class follow_lane:
    def __init__(self):
        #订阅YOLO识别结果
        self.identify_sub = rospy.Subscriber('/identify_info', identify, self.confirm)   
        #订阅位姿信息
        self.Pose_sub = rospy.Subscriber("lane_detect_pose", Pose, self.velctory)
        #发布速度指令
        self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=2)
    def confirm(self,identifies):
        clases = identifies.classes
        areas = identifies.area
        positions = list(identifies.position)        
        low_point = positions[3]

        global stop,stop_label,stop_man
        if low_point >=420:
            # 识别结果为行人（Ultraman）
            if clases <= 4 and clases >=2:
                stop = 1
                stop_man = 0
                rospy.loginfo("%s in front, Stop!",identifies.results)
        else:
            # 否则行人标志位自加1
            stop_man += 1
            if stop_man == 8:
                stop_man = 0
                stop = 0

    def velctory(self,Pose):        
        x = Pose.position.x
        y = Pose.position.y
        z = Pose.position.z 
        y_count = Pose.orientation.y
        global stop        
        print("stop:%d"%stop)
        
        vel = Twist()
        max_ang_vel = 0.8
        min_ang_vel = -0.8

        #  follow straight line
        if z <= 5 or stop == 1:
            #stop
            lin_vel = 0
            ang_vel = 0
            time.sleep(1)
            vel.linear.x  = lin_vel
            vel.angular.z = ang_vel
            
            self.vel_pub.publish(vel)
            rospy.loginfo(
                    "Publsh velocity command[{} m/s, {} rad/s]".format(
                        vel.linear.x, vel.angular.z))
        else  :
            # 直行在道路中央
            if x < 140 and x > 160:
                lin_vel = 0.19
                ang_vel = 0
            # 行驶在道路中间偏差太多调整
            elif x > 500 or x < 50:
                lin_vel = 0
                ang_vel = (1 - x / 150)
            # 行驶在道路中间偏差较小调整
            else:
                lin_vel = 0.19
                ang_vel = (1 - x / 150) * 0.25
            if x == 195 and y >= 360:
                lin_vel = 0.19
                ang_vel = -0.50
                
        if ang_vel >= max_ang_vel:
            ang_vel = max_ang_vel
        if ang_vel <= min_ang_vel:
            ang_vel = min_ang_vel

        vel.linear.x  = lin_vel
        vel.angular.z = ang_vel
        self.vel_pub.publish(vel)
                
        #rate.sleep()
        rospy.loginfo(
                    "Publsh velocity command[{} m/s, {} rad/s]".format(
                        vel.linear.x, vel.angular.z))
       

if __name__ == '__main__':
    try:
        # 初始化ros节点
        rospy.init_node("yolo_follow_lane", anonymous=True)
        rospy.loginfo("Starting yolo follow lane")
        global stop,stop_label,stop_man
        stop=stop_label=stop_man=0
        follow_lane()
        rospy.spin()
    except KeyboardInterrupt:
        print ("Shutting down yolo_follow_object node.")
        cv2.destroyAllWindows()











