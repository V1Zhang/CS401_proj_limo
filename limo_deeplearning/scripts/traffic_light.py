#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np
from math import *
from geometry_msgs.msg import Pose
from std_msgs.msg import Int16


class traffic_light:
    def __init__(self):    
        # 创建cv_bridge，声明图像的发布者和订阅者
        self.image_pub = rospy.Publisher("traffic_detect_image", Image, queue_size=1)
        self.light_mode_pub = rospy.Publisher("traffic_light_mode", Int16, queue_size=1)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)

    def callback(self,data):

        # 使用cv_bridge将ROS的图像数据转换成OpenCV的图像格式
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = hsv
        cimg = img

 
        # red range
        lower_red1 = np.array([0, 100, 220])   #[0, 130, 150]
        upper_red1 = np.array([20, 255, 255])   #[20, 255, 255]

        lower_red3 = np.array([160, 100, 210])   #[160, 220, 150]
        upper_red3 = np.array([180, 245, 255])    #[180, 255, 255]
        # green range
        lower_green1 = np.array([40, 200, 150])     #[40, 40, 210]
        upper_green1 = np.array([90, 255, 255])     #[90, 255, 255]
        lower_green2 = np.array([5, 5, 234])     #[40, 40, 210]
        upper_green2 = np.array([35, 10, 235])     #[90, 255, 255]
        # yellow range
        lower_yellow = np.array([20, 100, 180])
        upper_yellow = np.array([30, 255, 255])

        #二值化图像
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask3 = cv2.inRange(hsv, lower_red3, upper_red3)
        #融合两个区间的图像
        maskr = cv2.add(mask1,mask3)
        maskg1 = cv2.inRange(hsv, lower_green1, upper_green1)
        maskg2 = cv2.inRange(hsv, lower_green2, upper_green2)
        maskg = cv2.add(maskg1,maskg2)
        masky = cv2.inRange(hsv, lower_yellow, upper_yellow)
        # 填充
        kernel = np.ones((5,5),np.uint8)
        maskr = cv2.morphologyEx(maskr,cv2.MORPH_CLOSE,kernel)
        #滤波
        maskr = cv2.medianBlur(maskr, 5)
        masky = cv2.medianBlur(masky, 5)
        maskg = cv2.medianBlur(maskg, 5)

        size = img.shape
 
        # hough circle detect
        r_circles = cv2.HoughCircles(maskr, cv2.HOUGH_GRADIENT, 1, 80,
                                     param1=50, param2=10, minRadius=0, maxRadius=20)
         
        g_circles = cv2.HoughCircles(maskg, cv2.HOUGH_GRADIENT, 1, 60,
                                     param1=50, param2=10, minRadius=0, maxRadius=20)
         
        y_circles = cv2.HoughCircles(masky, cv2.HOUGH_GRADIENT, 1, 30,
                                     param1=50, param2=7, minRadius=0, maxRadius=20)
 
        # traffic light detect
        r = 5
        bound = 5.0 / 10
        light_mode = Int16()
        light_mode.data = 0  # 默认无灯

        if r_circles is not None:
            
            r_circles = np.uint16(np.around(r_circles))
         
            for i in r_circles[0, :]:
                if i[0] > size[1] or i[1] > size[0] or i[1] > size[0] * bound:
                    continue
         
                h, s = 0.0, 0.0
                for m in range(-r, r):
                    for n in range(-r, r):
         
                        if (i[1] + m) >= size[0] or (i[0] + n) >= size[1]:
                            continue
                        h += maskr[i[1] + m, i[0] + n]
                        s += 1
                if h / s > 50:
                    cv2.circle(cimg, (i[0], i[1]), i[2] + 10, (0, 255, 0), 2)
                    cv2.circle(maskr, (i[0], i[1]), i[2] + 30, (0, 255, 0), 2)
                    cv2.putText(cimg, 'RED', (i[0], i[1]), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    light_mode.data = 2  # 红灯
            # rospy.loginfo("The detect result is RED")
         
        if g_circles is not None:
            
            g_circles = np.uint16(np.around(g_circles))
         
            for i in g_circles[0, :]:
                if i[0] > size[1] or i[1] > size[0] or i[1] > size[0] * bound:
                    continue
         
                h, s = 0.0, 0.0
                for m in range(-r, r):
                    for n in range(-r, r):
         
                        if (i[1] + m) >= size[0] or (i[0] + n) >= size[1]:
                            continue
                        h += maskg[i[1] + m, i[0] + n]
                        s += 1
                if h / s > 50:
                    cv2.circle(cimg, (i[0], i[1]), i[2] + 10, (0, 255, 0), 2)
                    cv2.circle(maskg, (i[0], i[1]), i[2] + 30, (0, 255, 0), 2)
                    cv2.putText(cimg,'GREEN', (i[0], i[1]), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    light_mode.data = 0  # 绿灯
            rospy.loginfo("The detect result is GREEN")
         
        if y_circles is not None:
            y_circles = np.uint16(np.around(y_circles))
         
            for i in y_circles[0, :]:
                if i[0] > size[1] or i[1] > size[0] or i[1] > size[0] * bound:
                    continue
         
                h, s = 0.0, 0.0
                for m in range(-r, r):
                    for n in range(-r, r):
         
                        if (i[1] + m) >= size[0] or (i[0] + n) >= size[1]:
                            continue 
                        h += masky[i[1] + m, i[0] + n]
                        s += 1
                if h / s > 50:
                    cv2.circle(cimg, (i[0], i[1]), i[2] + 10, (128, 255, 0), 2)
                    cv2.circle(masky, (i[0], i[1]), i[2] + 30, (128, 255,0), 2)
                    cv2.putText(cimg, 'YELLOW', (i[0], i[1]), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    light_mode.data = 1  # 黄灯
            # rospy.loginfo("The detect result is YELLOW")

        # 发布红绿灯状态
        self.light_mode_pub.publish(light_mode)

        cimg = cv2.cvtColor(cimg, cv2.COLOR_HSV2BGR)
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cimg, "bgr8"))#"bgr8" "mono8"
        except CvBridgeError as e:
            print(e)

if __name__ == '__main__':
    try:
        # 初始化ros节点
        rospy.init_node("detect_traffic_light", anonymous=True)
        rospy.loginfo("Starting traffic light detect")
        traffic_light()
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down traffic_light node.")
        cv2.destroyAllWindows() 

