#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <limo_deeplearning/msg/traffic.h>

class TrafficControl {
private:
    ros::NodeHandle nh_;
    ros::Subscriber traffic_sub_;
    ros::Publisher cmd_vel_pub_;
    std::string traffic_light_state_;
    double current_vel_;

public:
    TrafficControl() : current_vel_(0.0) {
        // 订阅红绿灯检测结果
        traffic_sub_ = nh_.subscribe("traffic_light_mode", 1, &TrafficControl::trafficCallback, this);
        // 发布速度控制命令
        cmd_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("cmd_vel", 1);
    }

    void trafficCallback(const limo_deeplearning::msg::traffic::ConstPtr& msg) {
        traffic_light_state_ = msg->name;
        ROS_INFO("Traffic light state: %s", traffic_light_state_.c_str());
        
        geometry_msgs::Twist cmd_vel;
        
        if (traffic_light_state_ == "RED") {
            // 红灯：逐渐减速到停止
            if (current_vel_ > 0.1) {
                current_vel_ -= 0.1;  // 每次减少0.1m/s
            } else {
                current_vel_ = 0.0;   // 完全停止
            }
            cmd_vel.linear.x = current_vel_;
            cmd_vel.angular.z = 0.0;  // 保持直线行驶
            ROS_WARN("Red light detected! Slowing down...");
        }
        else if (traffic_light_state_ == "YELLOW") {
            // 黄灯：缓慢减速
            if (current_vel_ > 0.05) {
                current_vel_ -= 0.05;  // 每次减少0.05m/s
            }
            cmd_vel.linear.x = current_vel_;
            cmd_vel.angular.z = 0.0;   // 保持直线行驶
            ROS_WARN("Yellow light detected! Caution...");
        }
        else {
            // 绿灯：恢复正常速度
            current_vel_ = 0.5;  // 设置正常行驶速度
            cmd_vel.linear.x = current_vel_;
            cmd_vel.angular.z = 0.0;
            ROS_INFO("Green light detected! Proceeding...");
        }

        cmd_vel_pub_.publish(cmd_vel);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "traffic_control");
    TrafficControl traffic_control;
    ros::spin();
    return 0;
} 