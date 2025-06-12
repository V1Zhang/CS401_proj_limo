#include "pure_pursuit.h"
#include <iostream>
#include <fstream>
#include <string>

PurePursuit::PurePursuit() {
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    std::string scan_topic, pose_topic, cmd_vel_topic, path_topic;
    private_nh.param("scan_topic", scan_topic, std::string("scan"));
    private_nh.param("pose_topic", pose_topic, std::string("pose"));

    private_nh.param("path_topic", path_topic, std::string("target_path"));
    private_nh.param("cmd_vel_topic", cmd_vel_topic, std::string("cmd_vel"));

    private_nh.param("obstacle_check_x_range", obstacle_check_x_range_, 0.5);
    private_nh.param("obstacle_check_y_range", obstacle_check_y_range_, 0.3);

    private_nh.param("position_control_tolerance", position_control_tolerance_, 0.01);
    private_nh.param("angle_control_tolerance", angle_control_tolerance_, 0.3);

    private_nh.param("max_lin_vel", max_lin_vel_, 0.5);
    private_nh.param("max_ang_vel", max_ang_vel_, M_PI);

    private_nh.param("position_kp", position_kp_, 1.0);
    private_nh.param("angle_kp", angle_kp_, 5.0);

    private_nh.param("look_ahead_dist", look_ahead_dist_, 1.0);
    private_nh.param("zebra_cooldown_duration", zebra_cooldown_duration_, 5.0);

    cmd_vel_pub_ = nh.advertise<geometry_msgs::Twist>(cmd_vel_topic, 1);
    next_waypoint_pub_ = nh.advertise<geometry_msgs::PoseStamped>("next_waypoint", 1);
    target_path_pub_ = nh.advertise<nav_msgs::Path>("target_path", 1, true);

    scan_sub_ = nh.subscribe(scan_topic, 1, &PurePursuit::scanCallback, this);
    zebra_sub_ = nh.subscribe("zebra_detected", 1, &PurePursuit::zebraCallback, this);
    traffic_mode_sub_ = nh.subscribe("/traffic_light_mode", 1, &PurePursuit::trafficModeCallback, this);

    // 初始化斑马线冷却时间为很久以前，以便第一次检测能正常工作
    last_zebra_pass_time_ = ros::Time(0);

    nav_msgs::Path path_msg;
    if (loadTargetPath("/home/agilex/target_path.txt", path_msg)) {
        makePath(path_msg);
        target_path_pub_.publish(path_msg);
        state_ = START;
    }
    else {
        ROS_ERROR("Failed to load path file!");
    }
}

bool PurePursuit::loadTargetPath(const std::string& path_file, nav_msgs::Path& path_msg) {
   std::ifstream stream(path_file);
   if (!stream) {
       ROS_ERROR("could not open %s", path_file.c_str());
       return false;
   }

   std::string str;
   while (std::getline(stream, str)) {
       std::stringstream ss(str);
       double x, y;
       ss >> x >> y;

       geometry_msgs::PoseStamped pose_msg;
       pose_msg.header.stamp = ros::Time::now();
       pose_msg.header.frame_id = "map";

       pose_msg.pose.position.x = x;
       pose_msg.pose.position.y = y;
       pose_msg.pose.position.z = 0.0;

       path_msg.header = pose_msg.header;
       path_msg.poses.push_back(pose_msg);
   }

   return true;
}

double PurePursuit::normalizeAngle(double angle) {
    while (angle >= M_PI) {
        angle -= 2 * M_PI;
    }
    while (angle <= -M_PI) {
        angle += 2 * M_PI;
    }
    return angle;
}

bool PurePursuit::getPose() {
    tf::StampedTransform tf;
    try {
        tf_listener_.waitForTransform("map", "base_link", ros::Time(0), ros::Duration(1.0));
        tf_listener_.lookupTransform("map", "base_link", ros::Time(0), tf);
    }
    catch (tf::TransformException e) {
        ROS_ERROR("%s", e.what());
        return false;
    }

    geometry_msgs::TransformStamped tf_msg;
    tf::transformStampedTFToMsg(tf, tf_msg);
    cur_pose_(0) = tf_msg.transform.translation.x;
    cur_pose_(1) = tf_msg.transform.translation.y;
    cur_pose_(2) = tf::getYaw(tf_msg.transform.rotation);

    return true;
}

void PurePursuit::detectOobstacle(const sensor_msgs::LaserScan& scan) {
    obstacle_distance_ = std::numeric_limits<double>::max();
    int ranges = scan.ranges.size();
    for (int i = 0; i < ranges; ++i) {
        if (scan.ranges[i] < scan.range_min)
            continue;

        double angle = scan.angle_min + i * scan.angle_increment;
        double x = scan.ranges[i] * cos(angle);
        double y = scan.ranges[i] * sin(angle);

        if (x > 0 && fabs(y) < obstacle_check_y_range_) {
            if(x < obstacle_distance_) {
                obstacle_distance_ = x;
            }
        }
    }
}

bool PurePursuit::checkHeading(const WayPoint& p) {
    double error = normalizeAngle(p.heading - cur_pose_(2));
    static int check_cnt = 0;

    if (fabs(error) < angle_control_tolerance_) {
        check_cnt++;
        if (check_cnt > 10) {
            check_cnt = 0;
            return true;
        }
        else {
            return false;
        }
    }
    else {
        check_cnt = 0;
        return false;
    }
}

bool PurePursuit::checkPosition(const WayPoint& p) {
    double error = cos(cur_pose_(2)) * (p.position(0) - cur_pose_(0)) +
                   sin(cur_pose_(2)) * (p.position(1) - cur_pose_(1));
    static int check_cnt = 0;

    if (fabs(error) < position_control_tolerance_) {
        check_cnt++;
        if (check_cnt > 10) {
            check_cnt = 0;
            return true;
        }
        else {
            return false;
        }
    }
    else {
        check_cnt = 0;
        return false;
    }
}

double PurePursuit::calculateAngVel(const WayPoint& p) {
    double error = normalizeAngle(p.heading - cur_pose_(2));
    double ang_vel = error * angle_kp_;
    if (ang_vel > 0) {
        ang_vel = std::min(ang_vel, max_ang_vel_);
    }
    else {
        ang_vel = std::max(ang_vel, -max_ang_vel_);
    }

    return ang_vel;
}

void PurePursuit::calculateVel(const WayPoint& closest_p, const WayPoint& next_p, double& lin_vel, double& ang_vel) {
    double dx = cos(cur_pose_(2)) * (next_p.position(0) - cur_pose_(0)) +
                sin(cur_pose_(2)) * (next_p.position(1) - cur_pose_(1));
    double dy = -sin(cur_pose_(2)) * (next_p.position(0) - cur_pose_(0)) +
                cos(cur_pose_(2)) * (next_p.position(1) - cur_pose_(1));
    double ds = sqrt(dx * dx + dy * dy);

    if (obstacle_distance_ < dx && obstacle_distance_ < obstacle_check_x_range_) {
        lin_vel = -0.3;
        ang_vel = -0.5;
        ROS_WARN_THROTTLE(1.0, "Obstacle detected!");
        return;
    }

    lin_vel = max_lin_vel_;
    if (fabs(dy) < 1e-3) {
        ang_vel = 0.0;
    }
    else {
        double rho = ds * ds / (2 * dy);
        ang_vel = lin_vel / rho;
    }

    if (fabs(normalizeAngle(closest_p.heading - next_p.heading)) < 1e-3) {
        ang_vel = angle_kp_ * ang_vel;
    }

    if (ang_vel > 0) {
        ang_vel = std::min(ang_vel, max_ang_vel_);
    }
    else {
        ang_vel = std::max(ang_vel, -max_ang_vel_);
    }
}

void PurePursuit::makePath(const nav_msgs::Path& path_msg) {
    path_.clear();

    int n = path_msg.poses.size();
    for (int i = 0; i < n - 1; ++i) {
        Eigen::Vector2d position(path_msg.poses[i].pose.position.x,
                                 path_msg.poses[i].pose.position.y);

        double dx = path_msg.poses[i + 1].pose.position.x - position(0);
        double dy = path_msg.poses[i + 1].pose.position.y - position(1);
        double theta = atan2(dy, dx);

        WayPoint p(position, theta);
        path_.push_back(p);
    }

    Eigen::Vector2d position(path_msg.poses[n - 1].pose.position.x,
                             path_msg.poses[n - 1].pose.position.y);
    double theta = path_.getEndPoint().heading;
    WayPoint p(position, theta);
    path_.push_back(p);
}

void PurePursuit::scanCallback(const sensor_msgs::LaserScanConstPtr& scan_msg) {
    if (state_ == IDLE)
        return;
    detectOobstacle(*scan_msg);
}

void PurePursuit::zebraCallback(const std_msgs::BoolConstPtr& zebra_msg) {
    // 检查是否在冷却期内
    ros::Duration cooldown_elapsed = ros::Time::now() - last_zebra_pass_time_;
    bool in_cooldown = cooldown_elapsed.toSec() < zebra_cooldown_duration_;
    
    if (zebra_msg->data && state_ == TRACK && !is_zebra_stopping_ && !in_cooldown) {
        ROS_INFO("Zebra crossing detected! Stopping for 1 second...");
        state_ = ZEBRA_STOP;
        zebra_stop_time_ = ros::Time::now();
        is_zebra_stopping_ = true;
        zebra_detected_ = true;
    } else if (zebra_msg->data && in_cooldown) {
        ROS_DEBUG("Zebra crossing detected but still in cooldown period (%.1f/%.1f sec)", 
                  cooldown_elapsed.toSec(), zebra_cooldown_duration_);
    }
}

void PurePursuit::track() {
    double lin_vel = 0, ang_vel = 0;
    static WayPoint closest_waypoint, next_waypoint;

    if (!getPose()) {
        return;
    }

    switch (state_) {
        case IDLE: {
            break;
        }
        case START: {
            WayPoint start_point = path_.getStartPoint();
            if (checkHeading(start_point)) {
                closest_waypoint = path_.getStartPoint();
                next_waypoint = path_.findNextPoint(closest_waypoint, look_ahead_dist_);
                state_ = TRACK;
            }
            else {
                ang_vel = calculateAngVel(start_point);
            }
            break;
        }
        case TRACK: {
            if (next_waypoint.id == path_.getEndPoint().id) {
                closest_waypoint = path_.getStartPoint();
                next_waypoint = path_.getStartPoint();
                calculateVel(closest_waypoint, next_waypoint, lin_vel, ang_vel);
            }
            else  {
                closest_waypoint = path_.findClosestPoint(cur_pose_, closest_waypoint.id);
                next_waypoint = path_.findNextPoint(closest_waypoint, look_ahead_dist_);
                calculateVel(closest_waypoint, next_waypoint, lin_vel, ang_vel);
            }
            break;
        }
        case ZEBRA_STOP: {
            // 停车1秒钟
            ros::Duration elapsed = ros::Time::now() - zebra_stop_time_;
            if (elapsed.toSec() >= 1.0) {
                ROS_INFO("Resuming navigation after zebra crossing stop. Starting %.1f second cooldown...", zebra_cooldown_duration_);
                state_ = TRACK;
                is_zebra_stopping_ = false;
                zebra_detected_ = false;
                last_zebra_pass_time_ = ros::Time::now(); // 设置冷却开始时间
            }
            // 在停车期间，速度保持为0
            lin_vel = 0.0;
            ang_vel = 0.0;
            break;
        }
    }
    switch (traffic_light_state_) {
        case 2:
            lin_vel = 0.0;
            ang_vel = 0.0;
            ROS_INFO_THROTTLE(1.0, "Red light detected: STOPPING.");
            break;
        case 1:
            lin_vel *= 0.1;
            ang_vel *= 0.1;
            ROS_INFO_THROTTLE(1.0, "Yellow light detected: SLOWING DOWN.");
            break;  
        case 0:
        default:
            break;
    }


    publishCmdVel(lin_vel, ang_vel);
    publishNextWayPoint(next_waypoint);
}

void PurePursuit::publishNextWayPoint(const WayPoint& p) {
    geometry_msgs::PoseStamped pose_msg;
    pose_msg.header.stamp = ros::Time::now();
    pose_msg.header.frame_id = "map";
    tf::Quaternion q;
    q.setRPY(0.0, 0.0, p.heading);
    pose_msg.pose.orientation.x = q.x();
    pose_msg.pose.orientation.y = q.y();
    pose_msg.pose.orientation.z = q.z();
    pose_msg.pose.orientation.w = q.w();
    pose_msg.pose.position.x = p.position(0);
    pose_msg.pose.position.y = p.position(1);
    pose_msg.pose.position.z = 0.0;

    next_waypoint_pub_.publish(pose_msg);
}

void PurePursuit::publishCmdVel(double lin_vel, double ang_vel) {
    geometry_msgs::Twist vel_msg;

    vel_msg.linear.x = lin_vel;
    vel_msg.linear.y = 0.0;
    vel_msg.linear.z = 0.0;
    vel_msg.angular.x = 0.0;
    vel_msg.angular.y = 0.0;
    vel_msg.angular.z = ang_vel;

    cmd_vel_pub_.publish(vel_msg);
}

void PurePursuit::run() {
    ros::Rate rate(50);

    while (ros::ok()) {
        track();
        rate.sleep();
        ros::spinOnce();
    }
}

void PurePursuit::trafficModeCallback(const std_msgs::Int16& msg) {
    traffic_light_state_ = msg.data;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "pure_pursuit");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    PurePursuit p;
    p.run();

    return 0;
}
