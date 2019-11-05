#!/usr/bin/env python3
#
# GNU General Public License v3.0
#
# This ROS node is based on this work:
# [Yan, H., Herath, S. and Furukawa, Y. (2019). RoNIN: Robust Neural Inertial Navigation in the Wild: Benchmark, Evaluations, and New Methods. [online] arXiv.org. Available at: https://arxiv.org/abs/1905.12853](https://arxiv.org/abs/1905.12853)
#

import sys
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

import os
import time
from os import path as osp

import numpy as np
import torch
import json
from scipy.interpolate import interp1d
from model_resnet1d import *

gPublisher = None
gReply = None
gDevice = None
gNetwork = None
gSlidingWindow = None
gDecimateCounter = None

def imuCallback(aRequest, aArgs):
    global gPublisher, gReply, gDevice, gNetwork, gSlidingWindow, gDecimateCounter

    stepSize = aArgs[0]
    windowSize = aArgs[1]

    gSlidingWindow = np.roll(gSlidingWindow, -1)
    gSlidingWindow[0, windowSize - 1] = aRequest.angular_velocity.x
    gSlidingWindow[1, windowSize - 1] = aRequest.angular_velocity.y
    gSlidingWindow[2, windowSize - 1] = aRequest.angular_velocity.z
    gSlidingWindow[3, windowSize - 1] = aRequest.linear_acceleration.x
    gSlidingWindow[4, windowSize - 1] = aRequest.linear_acceleration.y
    gSlidingWindow[5, windowSize - 1] = aRequest.linear_acceleration.z

    gDecimateCounter = gDecimateCounter + 1
    if gDecimateCounter == stepSize:
        tensor = torch.unsqueeze(torch.from_numpy(gSlidingWindow).float(), 0)
        result = gNetwork(tensor.to(gDevice)).cpu().detach().numpy()  # run on a [gyro0.x gyro1.x ... gyro199.x] ... [acc0.z, acc1.z ... acc199.z] chunk by calling ResNet1D.forward. For each iteration, 10 (step_size) oldest elements fall out and the same amount come in
        gReply.header = aRequest.header
        gReply.pose.pose.position.x = result[0][0]
        gReply.pose.pose.position.y = result[0][1]
        gPublisher.publish(gReply)
        gDecimateCounter = 0

    
def initRonin(aArgs):
    global gDevice, gNetwork

    if not torch.cuda.is_available() or aArgs["cpu"]:
        devName = 'cpu'
        gDevice = torch.device(devName)
        checkpoint = torch.load(aArgs["model_path"], map_location=lambda storage, location: storage)
    else:
        devName = 'cuda:0'
        gDevice = torch.device(devName)
        checkpoint = torch.load(args.model_path)

    fcConfig = {'fc_dim': 512, 'in_dim': args["window_size"] // 32 + 1, 'dropout': 0.5, 'trans_planes': 128}
    gNetwork = ResNet1D(6, 2, BasicBlock1D, [2, 2, 2, 2],
                           base_plane=64, output_block=FCOutputModule, kernel_size=3, **fcConfig)
    gNetwork.load_state_dict(checkpoint['model_state_dict'])
    gNetwork.eval().to(gDevice)
    rospy.loginfo('Model %s loaded to device %s.', args["model_path"], devName)


def initRos(aArgs):
    global gPublisher, gReply, gSlidingWindow, gDecimateCounter

    rospy.init_node('ros_melodic_ronin', anonymous=True)

    gPublisher = rospy.Publisher('ronin_odo', Odometry, queue_size=10)
    gReply = Odometry()
    gReply.pose.pose.position.z = 0.0
    gReply.pose.pose.orientation.x = 0.0
    gReply.pose.pose.orientation.y = 0.0
    gReply.pose.pose.orientation.z = 0.0
    gReply.pose.pose.orientation.w = 0.0
    gReply.pose.covariance = np.zeros(36)
    gReply.twist.twist.linear.x = 0.0
    gReply.twist.twist.linear.y = 0.0
    gReply.twist.twist.linear.z = 0.0
    gReply.twist.twist.angular.x = 0.0
    gReply.twist.twist.angular.y = 0.0
    gReply.twist.twist.angular.z = 0.0
    gReply.twist.covariance = np.zeros(36)

    gSlidingWindow = np.zeros((6, aArgs["window_size"]))
    gDecimateCounter = 0
    rospy.Subscriber("ronin_imu", Imu, imuCallback, callback_args=(aArgs["step_size"], aArgs["window_size"]))


def ronin(aArgs):
    initRonin(aArgs)
    initRos(aArgs)
    rospy.spin()


if __name__ == '__main__':
    args = {}
    args["step_size"] = rospy.get_param('/ros_melodic_ronin/step_size')
    args["window_size"] = rospy.get_param('/ros_melodic_ronin/window_size')
    args["cpu"] = rospy.get_param('/ros_melodic_ronin/cpu')
    args["model_path"] = rospy.get_param('/ros_melodic_ronin/model_path') # /home/balazs/munka/nowtech/repos/nowtechnologies/ronin/models/ronin_resnet/checkpoint_gsn_latest.pt
    ronin(args)
