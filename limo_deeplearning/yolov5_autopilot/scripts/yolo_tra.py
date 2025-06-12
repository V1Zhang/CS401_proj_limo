#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np
from math import *
from limo_deeplearning.msg import identify
# from limo_deeplearning.msg import traffic
from std_msgs.msg import Int16
#from learning_topic.msg import Person

class traffic_light:
    def __init__(self):    
        # 创建cv_bridge，声明图像的发布者和订阅者
        self.image_pub = rospy.Publisher("traffic_detect_image", Image, queue_size=1)
        # self.light_mode_pub = rospy.Publisher("traffic_light_mode", traffic,  queue_size=1)
        self.light_mode_pub = rospy.Publisher("traffic_light_mode", Int16,  queue_size=1)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image", Image, self.callback)
        self.yolo_sub = rospy.Subscriber("identify_info", identify, self.yolodetect)
        
    def yolodetect(self,identifies):
        name = identifies.results        
        clases = identifies.classes
        areas = identifies.area
        positions = list(identifies.position)
        accs = identifies.acc
        global x0,y0,x1,y1
        #print(name)
        #如果识别结果为红绿灯，则获取顶点坐标
        if clases == 1:
            x0 = int(positions[0])
            y0 = int(positions[1])
            x1 = int(positions[2])
            y1 = int(positions[3])
        else:
        # 反之则识别特定区域，此区域为场地内，从而减小场地之外的干扰
            x0 = 400
            y0 = 0
            x1 = 640
            y1 = 300  
    
    def callback(self,data):
        # light_mode = traffic()
        # light_mode.name = 'empty'
        # light_mode.number = 0
        msg = Int16()
        msg.data = 0  # 默认无灯

        # 使用cv_bridge将ROS的图像数据转换成OpenCV的图像格式
        global x0,y0,x1,y1
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print (e)
        #识别指定区域
        cv_image = cv_image[y0:y1,x0:x1]
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = hsv

        cimg = img
        # color range
        lower_red1 = np.array([0, 15, 220])   #[0, 130, 150]
        upper_red1 = np.array([20, 30, 255])   #[20, 255, 255]
        lower_red3 = np.array([160, 150, 210])   #[160, 220, 150]
        upper_red3 = np.array([180, 245, 255])    #[180, 255, 255]

        lower_green1 = np.array([40, 140, 100])     #[40, 40, 210]
        upper_green1 = np.array([90, 255, 255])     #[90, 255, 255]
        lower_green2 = np.array([30, 2, 200])     #[40, 40, 210]
        upper_green2 = np.array([40, 25, 255])

        lower_yellow = np.array([28, 10, 230])
        upper_yellow = np.array([35, 20, 255])
        lower_yellow1 = np.array([8, 190, 180])
        upper_yellow1 = np.array([15, 225, 255])
        #中心区域阈值
        lower = np.array([25, 2, 230])
        upper = np.array([34, 20, 235])
        #二值化图像
        mask = cv2.inRange(hsv, lower, upper)
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask3 = cv2.inRange(hsv, lower_red3, upper_red3)
        maskr = cv2.add(mask1,mask3)
        maskr = cv2.add(maskr,mask)
        maskg1 = cv2.inRange(hsv, lower_green1, upper_green1)
        maskg2 = cv2.inRange(hsv, lower_green2, upper_green2)
        maskg = cv2.add(maskg1,maskg2)
        maskg = cv2.add(maskg,mask)

        masky1 = cv2.inRange(hsv, lower_yellow, upper_yellow)
        masky2 = cv2.inRange(hsv, lower_yellow1, upper_yellow1)

        masky = cv2.add(masky1,masky2)
        masky = cv2.add(masky,mask)
        
        kernel = np.ones((5,5),np.uint8)
        #填充图像
        maskr = cv2.morphologyEx(maskr,cv2.MORPH_CLOSE,kernel)
        masky = cv2.morphologyEx(masky,cv2.MORPH_CLOSE,kernel)
        maskg = cv2.morphologyEx(maskg,cv2.MORPH_CLOSE,kernel)
        #中值滤波
        maskr = cv2.medianBlur(maskr, 5)
        masky = cv2.medianBlur(masky, 5)
        maskg = cv2.medianBlur(maskg, 5)

        size = img.shape

        # hough circle detect
        r_circles = cv2.HoughCircles(maskr, cv2.HOUGH_GRADIENT, 1, 80,
                                     param1=50, param2=10, minRadius=5, maxRadius=20)
         
        g_circles = cv2.HoughCircles(maskg, cv2.HOUGH_GRADIENT, 1, 60,
                                     param1=50, param2=10, minRadius=5, maxRadius=20)
         
        y_circles = cv2.HoughCircles(masky, cv2.HOUGH_GRADIENT, 1, 30,
                                     param1=50, param2=7, minRadius=5, maxRadius=20)
 
        # traffic light detect
        r = 5
        bound = 6.0 / 10
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
                #添加标签
                if h / s > 50:
                    cv2.circle(cimg, (i[0], i[1]), i[2] + 10, (0, 255, 0), 2)
                    cv2.circle(maskr, (i[0], i[1]), i[2] + 30, (0, 255, 0), 2)
                    cv2.putText(cimg, 'RED', (i[0], i[1]), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        
            # light_mode.name = 'RED'
            # light_mode.number = 1
            msg.data = 2
            rospy.loginfo("The detect result is RED")
         
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
                # 添加标签
                if h / s > 50:
                    cv2.circle(cimg, (i[0], i[1]), i[2] + 10, (0, 255, 0), 2)
                    cv2.circle(maskg, (i[0], i[1]), i[2] + 30, (0, 255, 0), 2)
                    cv2.putText(cimg,'GREEN', (i[0], i[1]), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            # light_mode.name = 'GREEN'
            # light_mode.number = 1
            msg.data = 0
            rospy.loginfo("The detect result is GREEN")
         
        if y_circles is not None:
            
            if light_mode.name == 'empty':
                # light_mode.name = 'YELLOW'
                # light_mode.number = 1
                msg.data = 1
            else:
                # light_mode.number = 2
                msg.data = 2
            rospy.loginfo("The detect result is YELLOW")
            
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
                # 添加标签
                if h / s > 50:
                    cv2.circle(cimg, (i[0], i[1]), i[2] + 10, (128, 255, 0), 2)
                    cv2.circle(masky, (i[0], i[1]), i[2] + 30, (128, 255,0), 2)
                    cv2.putText(cimg, 'YELLOW', (i[0], i[1]), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        # 发布识别结果
        # self.light_mode_pub.publish(light_mode)
        self.light_mode_pub.publish(msg)
        # rospy.loginfo("Publsh light_mode message[%s,%d]", light_mode.name,light_mode.number)
        rospy.loginfo("Publsh light_mode message[%d]", msg.data)
        cimg = cv2.cvtColor(cimg, cv2.COLOR_HSV2BGR)
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cimg, "bgr8"))#"bgr8" "mono8"
        except CvBridgeError as e:
            print (e)

if __name__ == '__main__':
    try:
        # 初始化ros节点
        rospy.init_node("yolo_traffic_ight", anonymous=True)
        rospy.loginfo("Starting yolo traffic ight")
        global x0,y0,x1,y1
        x0=y0=x1=y1=0
        traffic_light()
        rospy.spin()
    except KeyboardInterrupt:
        print ("Shutting down yolo_traffic_light node.")
        cv2.destroyAllWindows() 

