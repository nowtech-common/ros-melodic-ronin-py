#!/usr/bin/env python3
import sys
import rospy
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

gOdometryPub = None

# TODO check RoNIN test data ranges and guess measurement unit
# TODO check if acc contains gravity
def imuCallback(aRequest):
    rospy.loginfo(rospy.get_caller_id() + "I heard %d", aRequest.data.header.seq)
    reply = Odometry
    reply.header = aRequest.data.header
    # TODO replace with RoNIN processing
    reply.pose.pose.position.x = aRequest.data.angular_velocity.x
    reply.pose.pose.position.y = aRequest.data.linear_acceleration.y
    reply.pose.pose.position.z = 0.0
    gOdometryPub.publish(reply)
    
def ronin(aArgs):
    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('ronin', anonymous=True)

    gOdometryPub = rospy.Publisher('ronin_odo', Odometry, queue_size=10)
    rospy.Subscriber("ronin_imu", Imu, imuCallback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_list', type=str)
    parser.add_argument('--val_list', type=str, default=None)
    parser.add_argument('--test_list', type=str, default=None)
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--root_dir', type=str, default=None, help='Path to data directory')
    parser.add_argument('--cache_path', type=str, default=None, help='Path to cache folder to store processed data')
    parser.add_argument('--dataset', type=str, default='ronin', choices=['ronin', 'ridi'])
    parser.add_argument('--max_ori_error', type=float, default=20.0)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--window_size', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-04)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--run_ekf', action='store_true')
    parser.add_argument('--fast_test', action='store_true')
    parser.add_argument('--show_plot', action='store_true')

    parser.add_argument('--continue_from', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--feature_sigma', type=float, default=0.00001)
    parser.add_argument('--target_sigma', type=float, default=0.00001)

    args = parser.parse_args()

    np.set_printoptions(formatter={'all': lambda x: '{:.6f}'.format(x)})
    ronin(args)
