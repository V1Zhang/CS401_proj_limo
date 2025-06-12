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
        #订阅位姿信息
        
        self.Pose_sub = rospy.Subscriber("lane_detect_pose", Pose, self.velctory)
        #发布速度指令
        self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=2)
        
       
    def velctory(self,Pose):        
        x = Pose.position.x
        y = Pose.position.y
        z = Pose.position.z 
        #y_count = Pose.orientation.y
        #rate = rospy.Rate(50) 
        vel = Twist()
        count = 0
        max_ang_vel = 0.8
        min_ang_vel = -0.8
        #  follow straight line
        if z <= 5:
            lin_vel = 0
            ang_vel = 0
        else : 
            
            if x<230 and x >200 :
                lin_vel = 0.19
                ang_vel = 0
            #elif x<360 and x>390:
                #lin_vel = 0.15
                #ang_vel = 0
            elif x>500 or x<50:
                lin_vel = 0
                ang_vel = (1-x/210)
            else:
                lin_vel = 0.19
                ang_vel = (1-x/210)*0.4

            if x == 210 and y >=350:                                
                
                c_count = 1
                if count > 0 :
                    for count in range(19):
                #while(count):
                        lin_vel = 0.21
                        ang_vel = -0.53
                        vel.linear.x  = lin_vel
                        vel.angular.z = ang_vel

                        time.sleep(0.1)
                        #if count > 0:
                            #count = count + 1
                        print('count:',count) 
                    #rospy.loginfo("Turn")
                        self.vel_pub.publish(vel)
                        rospy.loginfo(
                        "Publsh velocity command[{} m/s, {} rad/s]".format(
                            vel.linear.x, vel.angular.z))
                    #rate.sleep()
                    else :
                        c_count = 0
                    '''if count >23:
                        count = -2
                        print('count:',count)
                        #x = 220
                        lin_vel = 0
                        ang_vel = 0
                        vel.linear.x  = lin_vel
                        vel.angular.z = ang_vel
                        self.vel_pub.publish(vel)
                        rospy.loginfo("Stop")
                        time.sleep(2)
                        break
                    '''
                
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
        rospy.init_node("follow_lane")
        rospy.loginfo("Starting follow lane")
        follow_lane()
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down follow_object node."
        cv2.destroyAllWindows()











