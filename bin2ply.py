import numpy as np
import os

from plyfile import PlyData, PlyElement
import numpy as np
import torch.nn as nn
import torch
point = np.fromfile('/data15/DISCOVER_winter2024/zhengj2401/PVG/000001.bin', dtype=np.float32).reshape(-1, 4)
# print("point pos mean: ", np.mean(point, axis=0))
# print("point pos range: ", np.max(point, axis=0) - np.min(point, axis=0))
point_xyz_world = point[:, :3]



def save_ply(points, filename):

    vertex = np.array([(points[i][0], points[i][1], points[i][2]) for i in range(len(points))],
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_element = PlyElement.describe(vertex, 'vertex')

 
    ply_data = PlyData([vertex_element], text=True)
    ply_data.write(filename)

save_ply(point_xyz_world, '/data15/DISCOVER_winter2024/zhengj2401/PVG/000001.ply')