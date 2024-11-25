#!/usr/bin/env python3
import numpy as np

def rot_z(yaw):
    return np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ])

def quaternion_to_yaw(q):
    return np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))