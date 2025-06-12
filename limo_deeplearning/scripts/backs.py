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

image_pub = rospy.Publisher("lane_detect_image", Image, queue_size=1)
cap = cv2.VideoCapture(0)
# set image size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# set update frequence
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

while(1):
    ret, frame = cap.read()
    bridge = CvBridge()
    #print(frame)
    #gay = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    try:
        image_pub.publish(bridge.cv2_to_imgmsg(frame, "bgr8"))
    except CvBridgeError as e:
        print (e)
cap.release()
cv2.destroyAllWindows()

