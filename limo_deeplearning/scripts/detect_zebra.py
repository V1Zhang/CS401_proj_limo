#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
import numpy as np

class ZebraDetector:
    def __init__(self):
        self.zebra_pub = rospy.Publisher("zebra_detected", Bool, queue_size=1)
        self.bridge = CvBridge()
        
        # 简化的检测参数
        self.white_threshold = 250 # 白色阈值
        self.min_white_pixels = 3300  # 下部区域最少白色像素数量
        
        # 调试信息相关
        self.frame_count = 0
        self.last_detection = False
        self.detection_count = 0
        self.last_white_pixels = 0
        
        # 订阅相机话题
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        
        # rospy.loginfo("=== 斑马线检测器启动 ===")
        # rospy.loginfo("白色阈值: %d", self.white_threshold)
        # rospy.loginfo("最少白色像素: %d", self.min_white_pixels)
        # rospy.loginfo("订阅话题: /camera/color/image_raw")
        # rospy.loginfo("等待图像数据...")
    
    def callback(self, data):
        self.frame_count += 1
        
        # 每50帧输出一次状态信息
        # if self.frame_count % 50 == 0:
            # rospy.loginfo("📊 已处理 %d 帧图像，检测到斑马线 %d 次，最近白色像素: %d", 
                        #  self.frame_count, self.detection_count, self.last_white_pixels)
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            rospy.logerr("图像转换失败: %s", e)
            return
        
        # 简单的斑马线检测
        zebra_detected = self.detect_zebra(cv_image)
        
        # 检测状态变化时输出信息
        if zebra_detected != self.last_detection:
            if zebra_detected:
                self.detection_count += 1
                rospy.logwarn("🦓 检测到斑马线！[第%d次] 白色像素: %d，发送停车信号...", 
                             self.detection_count, self.last_white_pixels)
            else:
                rospy.loginfo("✅ 斑马线消失，恢复正常检测")
            self.last_detection = zebra_detected
        
        self.zebra_pub.publish(Bool(zebra_detected))
    
    def detect_zebra(self, image):
        """改进的斑马线检测 - 通过比较上下区域白色像素比例来区分斑马线和墙壁"""
        try:
            # 转灰度
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 获取图像尺寸
            height, width = gray.shape
            
            # 创建二值化图像
            _, binary = cv2.threshold(gray, self.white_threshold, 255, cv2.THRESH_BINARY)
            
            # 计算上下区域的白色像素数量
            upper_region = binary[0:int(height*0.7), :]  # 上70%区域
            lower_region = binary[int(height*0.7):, :]   # 下30%区域
            
            upper_white_pixels = cv2.countNonZero(upper_region)
            lower_white_pixels = cv2.countNonZero(lower_region)
            
            # 计算上下区域的白色像素比例
            upper_ratio = upper_white_pixels / (upper_region.shape[0] * upper_region.shape[1])
            lower_ratio = lower_white_pixels / (lower_region.shape[0] * lower_region.shape[1])
            
            # 判断条件：
            # 1. 下部区域白色像素数量超过阈值
            # 2. 上部区域白色像素比例要小于下部区域的一定比例
            is_zebra = (lower_white_pixels > self.min_white_pixels and 
                       upper_ratio < lower_ratio * 0.45)  # 上部区域白色比例要小于下部区域的30%
            
            # 调试信息
            # if self.frame_count % 50 == 0:
                # rospy.loginfo("🔍 检测信息:")
                # rospy.loginfo("- 上部区域白色像素: %d (比例: %.2f%%)", 
                            #  upper_white_pixels, upper_ratio * 100)
                # rospy.loginfo("- 下部区域白色像素: %d (比例: %.2f%%)", 
                            #  lower_white_pixels, lower_ratio * 100)
            
            # 可视化调试（可选）
            # if is_zebra:
            #     debug_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            #     # 绘制上下区域分界线
            #     cv2.line(debug_image, (0, int(height*0.7)), (width, int(height*0.7)), (0, 255, 0), 2)
            #     cv2.imshow("Zebra Detection", debug_image)
            #     cv2.waitKey(1)
            
            return is_zebra
            
        except Exception as e:
            rospy.logerr("检测错误: %s", e)
            return False

if __name__ == '__main__':
    try:
        rospy.init_node("zebra_detector", log_level=rospy.INFO)
        rospy.loginfo("正在启动斑马线检测节点...")
        detector = ZebraDetector()
        rospy.loginfo("斑马线检测节点运行中，按Ctrl+C退出")
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("用户中断，斑马线检测节点正常关闭")
    except Exception as e:
        rospy.logerr("节点启动失败: %s", e)
