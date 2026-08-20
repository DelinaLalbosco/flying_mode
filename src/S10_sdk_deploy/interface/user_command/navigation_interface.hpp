#pragma once

#include "user_command_interface.h"
#include "custom_types.h"

#include <thread>
#include <atomic>
#include <mutex>
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace interface;
using namespace types;

class NavigationInterface : public UserCommandInterface
{
private:

    std::atomic<bool> running_{false};
    std::thread navigation_thread_;

    mutable std::mutex navigation_mutex_;

    // ============================================================
    // Navigation parameters
    // ============================================================

    float max_forward_ = 0.35f;
    float max_side_    = 0.0f;
    float max_yaw_     = 0.8f;

    float turn_gain_ = 1.5f;

    // Start slowing down when heading error is larger than this
    float heading_slowdown_angle_ = 45.0f * M_PI / 180.0f;

    // Stop forward movement when heading error is larger than this
    float heading_stop_angle_ = 90.0f * M_PI / 180.0f;

    // Distance at which waypoint is considered reached
    float waypoint_reach_radius_ = 0.30f;

    // ============================================================
    // Robot pose
    // ============================================================

    float robot_x_ = 0.0f;
    float robot_y_ = 0.0f;
    float robot_yaw_ = 0.0f;

    // ============================================================
    // Waypoints
    //
    // Change these to your GOAI waypoints.
    // x, y
    // ============================================================

    std::vector<std::pair<float, float>> waypoints_ = {

        { 0.0f,  -1.725f },
        { -0.7125f, 11.655f },
        { -8.73f, 11.8425f },
        { -10.0f, 0.0f }

    };

    size_t current_waypoint_ = 0;

    // ============================================================
    // Utility
    // ============================================================

    static float normalize_angle(float angle)
    {
        while (angle > static_cast<float>(M_PI))
            angle -= 2.0f * static_cast<float>(M_PI);

        while (angle < -static_cast<float>(M_PI))
            angle += 2.0f * static_cast<float>(M_PI);

        return angle;
    }

    static float clamp(float value, float min_value, float max_value)
    {
        return std::max(min_value, std::min(value, max_value));
    }

    float get_distance_to_waypoint(float target_x, float target_y)
    {
        const float dx = target_x - robot_x_;
        const float dy = target_y - robot_y_;

        return std::sqrt(dx * dx + dy * dy);
    }

    // ============================================================
    // Navigation command generation
    // ============================================================

    void compute_navigation_command(
        float& forward,
        float& side,
        float& yaw)
    {
        forward = 0.0f;
        side = 0.0f;
        yaw = 0.0f;

        std::lock_guard<std::mutex> lock(navigation_mutex_);

        // No more waypoints
        if (current_waypoint_ >= waypoints_.size())
        {
            return;
        }

        const float target_x =
            waypoints_[current_waypoint_].first;

        const float target_y =
            waypoints_[current_waypoint_].second;

        const float dx = target_x - robot_x_;
        const float dy = target_y - robot_y_;

        const float distance =
            std::sqrt(dx * dx + dy * dy);

        // --------------------------------------------------------
        // Waypoint reached
        // --------------------------------------------------------

        if (distance < waypoint_reach_radius_)
        {
            std::cout
                << "\n[NAV] Waypoint "
                << current_waypoint_
                << " reached"
                << std::endl;

            current_waypoint_++;

            if (current_waypoint_ < waypoints_.size())
            {
                std::cout
                    << "[NAV] Next waypoint: "
                    << current_waypoint_
                    << " -> ("
                    << waypoints_[current_waypoint_].first
                    << ", "
                    << waypoints_[current_waypoint_].second
                    << ")"
                    << std::endl;
            }
            else
            {
                std::cout
                    << "[NAV] All waypoints reached!"
                    << std::endl;
            }

            return;
        }

        // --------------------------------------------------------
        // Desired heading
        // --------------------------------------------------------

        const float target_yaw =
            std::atan2(dy, dx);

        // --------------------------------------------------------
        // Heading error
        // --------------------------------------------------------

        const float heading_error =
            normalize_angle(target_yaw - robot_yaw_);

        // --------------------------------------------------------
        // Yaw command
        // --------------------------------------------------------

        yaw =
            clamp(
                turn_gain_ * heading_error,
                -max_yaw_,
                max_yaw_
            );

        // --------------------------------------------------------
        // Forward velocity
        // --------------------------------------------------------

        const float abs_heading_error =
            std::fabs(heading_error);

        if (abs_heading_error >= heading_stop_angle_)
        {
            // Robot is facing almost completely away
            // from target. Rotate first.
            forward = 0.0f;
        }
        else if (abs_heading_error > heading_slowdown_angle_)
        {
            // Reduce forward speed while turning.
            float scale =
                1.0f -
                (abs_heading_error - heading_slowdown_angle_) /
                (heading_stop_angle_ - heading_slowdown_angle_);

            scale = clamp(scale, 0.0f, 1.0f);

            forward = max_forward_ * scale;
        }
        else
        {
            // Normal forward movement.
            forward = max_forward_;
        }

        // No lateral movement for now.
        side = 0.0f;

        // --------------------------------------------------------
        // Print navigation information
        // --------------------------------------------------------

        std::cout
            << "\r[NAV] WP=" << current_waypoint_
            << " Pos=("
            << robot_x_
            << ", "
            << robot_y_
            << ")"
            << " Target=("
            << target_x
            << ", "
            << target_y
            << ")"
            << " Dist="
            << distance
            << " HeadingErr="
            << heading_error * 180.0f / static_cast<float>(M_PI)
            << " Forward="
            << forward
            << " Yaw="
            << yaw
            << std::flush;
    }

    // ============================================================
    // Navigation thread
    // ============================================================

    void navigation_loop()
    {
        std::cout
            << "\n[NavigationInterface] Navigation thread started"
            << std::endl;

        while (running_)
        {
            usr_cmd_->time_stamp = GetCurrentTimeStamp();

            float forward = 0.0f;
            float side = 0.0f;
            float yaw = 0.0f;

            // Only send walking commands in RL mode
            if (msfb_ != nullptr &&
                msfb_->GetCurrentState() ==
                    RobotMotionState::RLControlMode)
            {
                compute_navigation_command(
                    forward,
                    side,
                    yaw
                );
            }

            usr_cmd_->forward_vel_scale = forward;
            usr_cmd_->side_vel_scale = side;
            usr_cmd_->turnning_vel_scale = yaw;

            std::this_thread::sleep_for(
                std::chrono::milliseconds(10)
            );
        }

        // Safety: stop robot commands
        usr_cmd_->forward_vel_scale = 0.0f;
        usr_cmd_->side_vel_scale = 0.0f;
        usr_cmd_->turnning_vel_scale = 0.0f;

        std::cout
            << "\n[NavigationInterface] Navigation thread stopped"
            << std::endl;
    }

    double GetCurrentTimeStamp()
    {
        static auto start =
            std::chrono::steady_clock::now();

        auto now =
            std::chrono::steady_clock::now();

        return std::chrono::duration<double, std::milli>(
            now - start
        ).count();
    }

public:

    NavigationInterface(RobotName robot_name)
        : UserCommandInterface(robot_name)
    {
        std::cout
            << "[NavigationInterface] Initialized"
            << std::endl;

        std::memset(
            usr_cmd_,
            0,
            sizeof(UserCommand)
        );
    }

    ~NavigationInterface()
    {
        Stop();
    }

    // ============================================================
    // Start
    // ============================================================

    void Start() override
    {
        if (running_)
            return;

        running_ = true;

        navigation_thread_ =
            std::thread(
                &NavigationInterface::navigation_loop,
                this
            );
    }

    // ============================================================
    // Stop
    // ============================================================

    void Stop() override
    {
        running_ = false;

        if (navigation_thread_.joinable())
        {
            navigation_thread_.join();
        }

        std::lock_guard<std::mutex> lock(
            navigation_mutex_
        );

        usr_cmd_->forward_vel_scale = 0.0f;
        usr_cmd_->side_vel_scale = 0.0f;
        usr_cmd_->turnning_vel_scale = 0.0f;
    }

    // ============================================================
    // Required by UserCommandInterface
    // ============================================================

    UserCommand* GetUserCommand() override
    {
        return usr_cmd_;
    }

    // ============================================================
    // Set robot pose
    //
    // This will be called from MuJoCo / ROS2 later.
    // ============================================================

    void SetRobotPose(
        float x,
        float y,
        float yaw)
    {
        std::lock_guard<std::mutex> lock(
            navigation_mutex_
        );

        robot_x_ = x;
        robot_y_ = y;
        robot_yaw_ = normalize_angle(yaw);
    }

    // ============================================================
    // Set waypoints
    // ============================================================

    void SetWaypoints(
        const std::vector<std::pair<float, float>>& waypoints)
    {
        std::lock_guard<std::mutex> lock(
            navigation_mutex_
        );

        waypoints_ = waypoints;
        current_waypoint_ = 0;

        std::cout
            << "\n[NAV] Loaded "
            << waypoints_.size()
            << " waypoints"
            << std::endl;
    }

    // ============================================================
    // Set maximum velocities
    // ============================================================

    void SetMaxVelocities(
        float forward,
        float side,
        float yaw)
    {
        max_forward_ = std::fabs(forward);
        max_side_ = std::fabs(side);
        max_yaw_ = std::fabs(yaw);
    }

    // ============================================================
    // Reset navigation
    // ============================================================

    void ResetNavigation()
    {
        std::lock_guard<std::mutex> lock(
            navigation_mutex_
        );

        current_waypoint_ = 0;

        std::cout
            << "[NAV] Navigation reset"
            << std::endl;
    }
};