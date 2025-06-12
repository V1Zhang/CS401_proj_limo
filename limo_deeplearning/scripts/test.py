#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import time
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

rospy.init_node("move_ord")
rospy.loginfo("Starting move ord")

restar = 0
vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
rate = rospy.Rate(50)
def pub_cmd():
    vel = Twist()
    vel.linear.x  = 0.3
    vel.angular.z = -0.68
    #time.sleep(2)
    rate.sleep()
    rospy.loginfo(
                    "Publsh velocity command[{} m/s, {} rad/s]".format(
                        vel.linear.x, vel.angular.z))
    return vel;
def cou(count):
    while(count):
        global restar
        vel = pub_cmd() 
        vel_pub.publish(vel)
        time.sleep(0.1)   
        count = count+1
        print(count)
        if count >18:
            count = -1
            restar = 1
            rospy.loginfo("Stop")
    
    
count = 1
cou(count)   
if (restar) :
    vel = Twist()
    vel.linear.x  = 0.3
    vel_pub.publish(vel)
    rospy.loginfo(
                    "Publsh velocity command[{} m/s, {} rad/s]".format(
                        vel.linear.x, vel.angular.z))
    print ('restar:',restar)
    #break
rospy.spin()





