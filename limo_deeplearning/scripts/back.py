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

global over
over = 0
class follow_lane:
    def __init__(self):
        #订阅道路线位置
        self.Pose_sub = rospy.Subscriber("lane_detect_pose", Pose, self.velctory)
        #发布速度指令
        self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=2)
    def myhook(self):
        print "shutdown time!"

    def back(self,count):
        vel = Twist()
        global over
        lin_vel = 0
        ang_vel = 0
        vel.linear.x = lin_vel
        vel.angular.z = ang_vel
        self.vel_pub.publish(vel)
        rospy.loginfo(
            "Wait to back: command[{} m/s, {} rad/s]".format(
                vel.linear.x, vel.angular.z))
        time.sleep(1)
        while (count > 0):
            lin_vel = -0.4
            ang_vel = 1.1
            count = count - 1
            time.sleep(0.1)
            vel.linear.x = lin_vel
            vel.angular.z = ang_vel
            self.vel_pub.publish(vel)
            rospy.loginfo(
                "Start to back: command[{} m/s, {} rad/s]".format(
                    vel.linear.x, vel.angular.z))
        else:
            lin_vel = 0
            ang_vel = 0
            over = 1

    def velctory(self,Pose):        
        x = Pose.position.x
        y = Pose.position.y
        z = Pose.position.z
        global over
        vel = Twist()
        #设定角速度范围
        max_ang_vel = 0.8
        min_ang_vel = -2.0
        #如果识别不到道路信息或者倒车结束，停车
        if z <= 5 or over == 1:
            lin_vel = 0
            ang_vel = 0
            vel.linear.x  = lin_vel
            vel.angular.z = ang_vel
            self.vel_pub.publish(vel)
            time.sleep(0.3)
            rospy.loginfo(
                    "Publsh velocity command[{} m/s, {} rad/s]".format(
                        vel.linear.x, vel.angular.z))
        else  : 
            #识别到两条道路线直行
            if x<210 and x >190 :
                lin_vel = 0.19
                ang_vel = 0
            #识别到一条道路线直行
            elif x>90 and x<150:
                lin_vel = 0.19
                ang_vel = 0
            else:
                lin_vel = 0.19
                ang_vel = (1-x/200)*0.7
            #行驶至倒车点并开始倒车
            if x == 195 and y >=390:
                count = 17
                self.back(count)

        if ang_vel >= max_ang_vel:
            ang_vel = max_ang_vel
        if ang_vel <= min_ang_vel:
            ang_vel = min_ang_vel

        vel.linear.x  = lin_vel
        vel.angular.z = ang_vel
        #发布速度指令
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
        print "Shutting down follow_object node."
        cv2.destroyAllWindows()











