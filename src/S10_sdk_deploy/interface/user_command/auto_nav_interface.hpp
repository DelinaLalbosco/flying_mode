#pragma once

#include "user_command_interface.h"
#include "custom_types.h"
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>

using namespace interface;
using namespace types;

class AutoNavInterface : public UserCommandInterface
{
private:
    std::atomic<bool> running_{false};
    std::thread spin_thread_;
    rclcpp::Node::SharedPtr node_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr nav_cmd_sub_;

    mutable std::mutex cmd_mutex_;
    float fwd_ = 0.0f, side_ = 0.0f, yaw_ = 0.0f;

    void nav_cmd_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(cmd_mutex_);
        fwd_  = static_cast<float>(msg->linear.x);
        side_ = static_cast<float>(msg->linear.y);
        yaw_  = static_cast<float>(msg->angular.z);
    }

    void spin_loop()
    {
        while (running_)
        {
            rclcpp::spin_some(node_);

            if (msfb_->GetCurrentState() == uint8_t(RobotMotionState::WaitingForStand)) {
                usr_cmd_->target_mode = uint8_t(RobotMotionState::StandingUp);
            } else if (msfb_->GetCurrentState() == uint8_t(RobotMotionState::StandingUp)) {
                usr_cmd_->target_mode = uint8_t(RobotMotionState::RLControlMode);
            }

            float fwd, side, yaw;
            {
                std::lock_guard<std::mutex> lock(cmd_mutex_);
                fwd = fwd_; side = side_; yaw = yaw_;
            }

            usr_cmd_->forward_vel_scale  = fwd;
            usr_cmd_->side_vel_scale     = side;
            usr_cmd_->turnning_vel_scale = yaw;
            usr_cmd_->time_stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count() / 1000.0;

            std::this_thread::sleep_for(std::chrono::milliseconds(4));
        }
    }

public:
    AutoNavInterface(RobotName robot_name) : UserCommandInterface(robot_name)
    {
        std::cout << "[AutoNavInterface] Ready!\n";
        std::memset(usr_cmd_, 0, sizeof(UserCommand));
        node_ = std::make_shared<rclcpp::Node>("auto_nav_interface");
        nav_cmd_sub_ = node_->create_subscription<geometry_msgs::msg::Twist>(
            "/AUTO_NAV_CMD", 10,
            std::bind(&AutoNavInterface::nav_cmd_callback, this, std::placeholders::_1));
    }

    ~AutoNavInterface() { Stop(); }

    void Start() override
    {
        if (running_) return;
        running_ = true;
        spin_thread_ = std::thread(&AutoNavInterface::spin_loop, this);
    }

    void Stop() override
    {
        running_ = false;
        if (spin_thread_.joinable()) spin_thread_.join();
        usr_cmd_->forward_vel_scale = usr_cmd_->side_vel_scale = usr_cmd_->turnning_vel_scale = 0.0f;
    }

    UserCommand* GetUserCommand() override { return usr_cmd_; }
};
