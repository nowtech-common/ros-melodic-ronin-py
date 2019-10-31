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

# TODO check RoNIN test data ranges and guess measurement unit
# TODO check if acc contains gravity
def imuCallback(aRequest, aArgs):
    stepSize = aArgs[0]
    windowSize = aArgs[1]
    publisher = aArgs[2]
    reply = aArgs[3]
    network = aArgs[4]
    slidingWindow = aArgs[5]
    decimateCounter = aArgs[6]

    slidingWindow = np.roll(slidingWindow, -1)
    slidingWindow[0, windowSize - 1] = aRequest.angular_velocity.x
    slidingWindow[1, windowSize - 1] = aRequest.angular_velocity.y
    slidingWindow[2, windowSize - 1] = aRequest.angular_velocity.z
    slidingWindow[3, windowSize - 1] = aRequest.linear_acceleration.x
    slidingWindow[4, windowSize - 1] = aRequest.linear_acceleration.y
    slidingWindow[5, windowSize - 1] = aRequest.linear_acceleration.z

    decimateCounter = decimateCounter + 1
    if decimateCounter == stepSize:
        result = network(slidingWindow.to(device)).cpu().detach().numpy()  # run on a [gyro0.x gyro1.x ... gyro199.x] ... [acc0.z, acc1.z ... acc199.z] chunk by calling ResNet1D.forward. For each iteration, 10 (step_size) oldest elements fall out and the same amount come in
        print(result)
        reply.header = aRequest.header
        reply.pose.pose.position.x = result[0]
        reply.pose.pose.position.y = result[1]
        publisher.publish(reply)
        decimateCounter = 0

    
def initRonin(aArgs):
    if args.out_dir is not None and not osp.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    if not torch.cuda.is_available() or args.cpu:
        device = torch.device('cpu')
        checkpoint = torch.load(args.model_path, map_location=lambda storage, location: storage) # checkpoint variable contains the model
    else:
        device = torch.device('cuda:0')
        checkpoint = torch.load(args.model_path)

    fcConfig = {'fc_dim': 512, 'in_dim': args.window_size // 32 + 1, 'dropout': 0.5, 'trans_planes': 128}
    network = ResNet1D(6, 2, BasicBlock1D, [2, 2, 2, 2],
                           base_plane=64, output_block=FCOutputModule, kernel_size=3, **fcConfig)
    network.load_state_dict(checkpoint['model_state_dict']) #Copies parameters and buffers from state_dict into this module and its descendants.
    network.eval().to(device)                               # Sets the module in evaluation mode. This is equivalent with self.train(False).
    #  This method modifies the module in-place. the desired device of the parameters and buffers in this module
    print('Model {} loaded to device {}.'.format(args.model_path, device))
    return network


def initRos(aArgs, aNetwork):
    rospy.init_node('ronin', anonymous=True)

    odometryPub = rospy.Publisher('ronin_odo', Odometry, queue_size=10)
    reply = Odometry()
    reply.pose.pose.position.z = 0.0
    reply.pose.pose.orientation.x = 0.0
    reply.pose.pose.orientation.y = 0.0
    reply.pose.pose.orientation.z = 0.0
    reply.pose.pose.orientation.w = 0.0
    reply.pose.covariance = np.zeros(36)
    reply.twist.twist.linear.x = 0.0
    reply.twist.twist.linear.y = 0.0
    reply.twist.twist.linear.z = 0.0
    reply.twist.twist.angular.x = 0.0
    reply.twist.twist.angular.y = 0.0
    reply.twist.twist.angular.z = 0.0
    reply.twist.covariance = np.zeros(36)

    slidingWindow = np.zeros((6, aArgs.window_size))
    decimateCounter = 0
    rospy.Subscriber("ronin_imu", Imu, imuCallback, callback_args=(aArgs.step_size, aArgs.window_size, odometryPub, reply, aNetwork, slidingWindow, decimateCounter))


def ronin(aArgs):
    network = initRonin(aArgs)
    initRos(aArgs, network)
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
