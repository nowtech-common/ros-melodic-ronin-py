#!/usr/bin/env python3
# license removed for brevity
import math
import rospy
import std_msgs.msg
from sensor_msgs.msg import Imu

# TODO output ground truth
def talker(aArgs):
    pub = rospy.Publisher('ronin_imu', Imu, queue_size=10)
    rospy.init_node('circle_test_source', anonymous=True)
    rate = rospy.Rate(aArgs.sampling_freq) # Hz
    step = 0
    imu = Imu()
    imu.orientation_covariance = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    imu.angular_velocity_covariance = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    imu.linear_acceleration_covariance = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    imu.orientation.x = 0.0
    imu.orientation.y = 0.0
    imu.orientation.z = 0.0
    imu.orientation.w = 0.0
    imu.angular_velocity.x = 0.0
    imu.angular_velocity.y = 0.0
    imu.linear_acceleration.z = 0.0
    while not rospy.is_shutdown():
        imu.header.seq = step
        imu.header.stamp = rospy.Time.now()
        angle = 2.0 * math.pi * aArgs.envelope_freq / aArgs.sampling_freq * step
        omega = aArgs.omega_max * (1.0 - math.cos(angle)) / 2.0
        imu.angular_velocity.z = omega
        imu.linear_acceleration.x = math.pi * aArgs.envelope_freq * aArgs.omega_max * math.sin(angle) * aArgs.radius
        imu.linear_acceleration.y = omega * omega * aArgs.radius
        pub.publish(imu)
        step = step + 1
        rate.sleep()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--radius', type=float, default=1.0)
    parser.add_argument('--omega_max', type=float, default=1.0)
    parser.add_argument('--sampling_freq', type=int, default=200)
    parser.add_argument('--envelope_freq', type=float, default=0.5 / math.pi)

    args = parser.parse_args()

    try:
        talker(args)
    except rospy.ROSInterruptException:
        pass
