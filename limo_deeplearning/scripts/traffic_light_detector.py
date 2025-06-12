#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np
from limo_deeplearning.msg import traffic

class TrafficLightDetector:
    def __init__(self):    
        # 创建cv_bridge
        self.bridge = CvBridge()
        
        # 订阅相机图像
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        
        # 发布处理后的图像
        self.image_pub = rospy.Publisher("traffic_light_image", Image, queue_size=1)
        
        # 发布红绿灯状态
        self.traffic_pub = rospy.Publisher("traffic_light_mode", traffic, queue_size=1)
        
        # 初始化状态消息
        self.traffic_msg = traffic()
        self.traffic_msg.name = "UNKNOWN"
        self.traffic_msg.position = [0, 0, 0]
        self.traffic_msg.area = 0.0

    def detect_color(self, hsv_img):
        # 红色范围
        red_min1 = np.array([0, 5, 150])
        red_max1 = np.array([8, 255, 255])
        red_min2 = np.array([175, 5, 150])
        red_max2 = np.array([180, 255, 255])

        # 黄色范围
        yellow_min = np.array([20, 5, 150])
        yellow_max = np.array([30, 255, 255])

        # 绿色范围
        green_min = np.array([35, 5, 150])
        green_max = np.array([90, 255, 255])

        # 创建掩码
        red_mask = cv2.inRange(hsv_img, red_min1, red_max1) + \
                  cv2.inRange(hsv_img, red_min2, red_max2)
        yellow_mask = cv2.inRange(hsv_img, yellow_min, yellow_max)
        green_mask = cv2.inRange(hsv_img, green_min, green_max)

        # 中值滤波去噪
        red_mask = cv2.medianBlur(red_mask, 5)
        yellow_mask = cv2.medianBlur(yellow_mask, 5)
        green_mask = cv2.medianBlur(green_mask, 5)

        # 计算非零像素数量
        red_count = cv2.countNonZero(red_mask)
        yellow_count = cv2.countNonZero(yellow_mask)
        green_count = cv2.countNonZero(green_mask)

        # 返回检测结果
        max_count = max(red_count, yellow_count, green_count)
        if max_count > 60:
            if max_count == red_count:
                return "RED", red_mask
            elif max_count == yellow_count:
                return "YELLOW", yellow_mask
            elif max_count == green_count:
                return "GREEN", green_mask
        return "UNKNOWN", None

    def callback(self, data):
        try:
            # 转换图像格式
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            
            # 检测红绿灯
            color, mask = self.detect_color(hsv_image)
            
            # 更新状态消息
            self.traffic_msg.name = color
            self.traffic_msg.header.stamp = rospy.Time.now()
            
            # 如果检测到红绿灯，计算位置和面积
            if mask is not None:
                # 查找轮廓
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # 找到最大的轮廓
                    max_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(max_contour)
                    if area > 100:  # 面积阈值
                        M = cv2.moments(max_contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            self.traffic_msg.position = [cx, cy, 0]
                            self.traffic_msg.area = area
                            
                            # 在图像上标记
                            cv2.circle(cv_image, (cx, cy), 5, (0, 255, 0), -1)
                            cv2.putText(cv_image, color, (cx-20, cy-20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 发布处理后的图像
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
            
            # 发布红绿灯状态
            self.traffic_pub.publish(self.traffic_msg)
            
            # 输出日志
            rospy.loginfo("Traffic light detected: %s", color)
            
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", e)

if __name__ == '__main__':
    try:
        # 初始化节点
        rospy.init_node("traffic_light_detector", anonymous=True)
        rospy.loginfo("Starting traffic light detector...")
        
        # 创建检测器实例
        detector = TrafficLightDetector()
        
        # 保持节点运行
        rospy.spin()
        
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down traffic light detector node.")
        cv2.destroyAllWindows() 