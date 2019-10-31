#!/usr/bin/env python3
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

# TODO check RoNIN test data ranges and guess measurement unit
# TODO check if acc contains gravity
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

    if args.out_dir is not None and not osp.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    if not torch.cuda.is_available() or args.cpu:
        devName = 'cpu'
        gDevice = torch.device(devName)
        checkpoint = torch.load(args.model_path, map_location=lambda storage, location: storage) # checkpoint variable contains the model
    else:
        devName = 'cuda:0'
        gDevice = torch.device(devName)
        checkpoint = torch.load(args.model_path)

    fcConfig = {'fc_dim': 512, 'in_dim': args.window_size // 32 + 1, 'dropout': 0.5, 'trans_planes': 128}
    gNetwork = ResNet1D(6, 2, BasicBlock1D, [2, 2, 2, 2],
                           base_plane=64, output_block=FCOutputModule, kernel_size=3, **fcConfig)
    gNetwork.load_state_dict(checkpoint['model_state_dict']) #Copies parameters and buffers from state_dict into this module and its descendants.
    gNetwork.eval().to(gDevice)                               # Sets the module in evaluation mode. This is equivalent with self.train(False).
    #  This method modifies the module in-place. the desired device of the parameters and buffers in this module
    rospy.loginfo('Model %s loaded to device %s.', args.model_path, devName)


def initRos(aArgs):
    global gPublisher, gReply, gSlidingWindow, gDecimateCounter

    rospy.init_node('ronin', anonymous=True)

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

    gSlidingWindow = np.zeros((6, aArgs.window_size))
    gDecimateCounter = 0
    rospy.Subscriber("ronin_imu", Imu, imuCallback, callback_args=(aArgs.step_size, aArgs.window_size))


def ronin(aArgs):
    initRonin(aArgs)
    initRos(aArgs)
    rospy.spin()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_ori_error', type=float, default=20.0)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--window_size', type=int, default=200)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--show_plot', action='store_true')
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--model_path', type=str, default='/home/balazs/munka/nowtech/repos/nowtechnologies/ronin/models/ronin_resnet/checkpoint_gsn_latest.pt')

    args = parser.parse_args()

    np.set_printoptions(formatter={'all': lambda x: '{:.6f}'.format(x)})
    ronin(args)
