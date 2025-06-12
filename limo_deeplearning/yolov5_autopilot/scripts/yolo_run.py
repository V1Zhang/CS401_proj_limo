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
        #订阅红绿灯检测结果
        self.traffic_sub = rospy.Subscriber("traffic_light_mode", traffic, self.trafficcc)
        #订阅YOLO检测结果
        self.identify_sub = rospy.Subscriber('/identify_info', identify, self.confirm)   
        #订阅位姿信息
        self.Pose_sub = rospy.Subscriber("lane_detect_pose", Pose, self.velctory)
        #发布速度指令
        self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=2)
    def confirm(self,identifies):
        clases = identifies.classes
        areas = identifies.area
        positions = list(identifies.position)        
        accs = identifies.acc

        global stop,stop_label,stop_man,over,over_label
        #红绿灯
        if clases == 1 :
            low_point = positions[3]
            if low_point >= 350 and stop_label == 1:  #Determine the distance of traffic light and result
                stop = 1
                stop_man = 0
            print('low_point:%d'%low_point)
        #行人
        elif clases <= 4 and clases >=2:
            low_point = positions[3]          
            if low_point >=410:
                stop = 1
                stop_man = 0
                rospy.loginfo("%s in front, Stop!",identifies.results)
            print('low_point:',low_point)
            #print('stop:%d'%stop)
        # “STOP”结束停车
        elif clases == 14:
            if accs >=  0.88:
                over_label += 1
        # 右转标志
        elif clases == 0:
            if over_label >= 1 and accs >= 0.9:
                over = 1
            else:
                stop_man += 1     #记录行人消失后的识别次数，大于一定次数之后才会开始运动
                if stop_man >=8:
                    stop = 0
        elif clases > 4 and clases < 14 or clases > 14:
            stop_man += 1
            if stop_man >=8:
                stop = 0

    def trafficcc(self,traffic_mode):
        light1 = traffic_mode.name
        light2 = traffic_mode.number
        global stop,stop_label,stop_label_man
        
        if light1 == 'RED':
            stop_label = 1
            print('RED')
        elif light1 == 'GREEN':
            stop_label = 2
        elif light1 == 'YELLOW':
            stop_label = 3
        elif light2 == 2:
            stop_label = 1
            #print('two lights')
        
    def velctory(self,Pose):        
        x = Pose.position.x
        y = Pose.position.y
        z = Pose.position.z 
        y_count = Pose.orientation.y
        global stop,over,start
        
        print("stop:%d"%stop)
        vel = Twist()
        max_ang_vel = 0.8
        min_ang_vel = -0.8
        # 行驶出库
        if start == 1:
            while(start):
                lin_vel=0.32
                ang_vel=-0.91
                vel.linear.x  = lin_vel
                vel.angular.z = ang_vel
                rate.sleep()
                rospy.loginfo(
                            "Publsh velocity command[{} m/s, {} rad/s]".format(
                                vel.linear.x, vel.angular.z))
                vel_pub.publish(vel)
                time.sleep(0.1)   
                start = start+1
                print(start)
                if start >30:
                    start = -1            
                    rospy.loginfo("Start YOLO")
        #  follow straight line
        elif z <= 5 or stop == 1 or over == 1:
        #stop或者over结束
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
            # 在道路中间直行
            if x < 140 and x > 160:
                lin_vel = 0.19
                ang_vel = 0
            # 偏差过大时调整
            elif x > 500 or x < 50:
                lin_vel = 0
                ang_vel = (1 - x / 150)
            #偏差较小时调整
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
        vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        rate = rospy.Rate(50)
        global stop,stop_label,stop_man,start,over,over_label
        start = 1
        stop=stop_label=stop_man=over=over_label=0
        
        follow_lane()
        rospy.spin()
    except KeyboardInterrupt:
        print ("Shutting down yolo_follow_object node.")
        cv2.destroyAllWindows()











