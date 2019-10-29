#!/usr/bin/env python3
# license removed for brevity
import rospy
from sensor_msgs.msg import Imu

def talker():
    pub = rospy.Publisher('ronin_imu', String, queue_size=10)
    rospy.init_node('circle_test_source', anonymous=True)
    rate = rospy.Rate(200) # hz
    while not rospy.is_shutdown():
# TODO replace with real logic
        hello_str = "hello world %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
