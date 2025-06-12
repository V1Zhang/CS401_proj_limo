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

class follow_lane:
    def __init__(self):
        #订阅道路线位置
        self.Pose_sub = rospy.Subscriber("lane_detect_pose", Pose, self.velctory)
        #发布速度指令
        self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=2)

    def velctory(self,Pose):        
        x = Pose.position.x
        y = Pose.position.y
        z = Pose.position.z

        vel = Twist()
        #设置速度范围
        max_ang_vel = 0.8
        min_ang_vel = -0.8
        #当检测不到道路线时，停止
        if z <= 5 :
            lin_vel = 0
            ang_vel = 0
            vel.linear.x  = lin_vel
            vel.angular.z = ang_vel
            self.vel_pub.publish(vel)
            time.sleep(1)
            rospy.loginfo(
                    "Publsh velocity command[{} m/s, {} rad/s]".format(
                        vel.linear.x, vel.angular.z))
        else  : 
            #检测一条道路线时，道路中心在此范围直行
            if x<210 and x >190 :
                lin_vel = 0.19
                ang_vel = 0
            #偏差太大时，原地转向
            elif x>500 or x<50:
                lin_vel = 0
                ang_vel = (1-x/210)
            #偏差较小时微调车身位置
            else:
                lin_vel = 0.19
                ang_vel = (1-x/200)*0.7
            #转弯
            if x == 195 and y >=350:
                lin_vel = 0.19
                ang_vel = -0.60
        #设定转向速度范围
        if ang_vel >= max_ang_vel:
            ang_vel = max_ang_vel
        if ang_vel <= min_ang_vel:
            ang_vel = min_ang_vel
        #发布速度指令
        vel.linear.x  = lin_vel
        vel.angular.z = ang_vel
        self.vel_pub.publish(vel)
        rospy.loginfo(
                    "Publsh velocity command[{} m/s, {} rad/s]".format(
                        vel.linear.x, vel.angular.z))

if __name__ == '__main__':
    try:
        # 初始化ros节点
        rospy.init_node("follow_lane", anonymous=True)
        rospy.loginfo("Starting follow lane")
        follow_lane()
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down follow_object node.")
        cv2.destroyAllWindows()











