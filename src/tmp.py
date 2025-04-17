## DATA

# Height -> AHN

import open3d as o3d
import numpy as np
import laspy as lp
import os
# os.chdir(r'/tudelft.net/staff-umbrella/EDT Veluwe/testbed/AHN/')
os.chdir(r'/home/mahmoudahmed/Downloads')

# tr_pcd = lp.read("C_27DN1.LAZ")
tr_pcd = lp.read("33CN2_13.LAZ")

print([dimension.name for dimension in tr_pcd.point_format.dimensions])
# print(np.max(tr_pcd.green))
points = np.vstack((tr_pcd.x, tr_pcd.y, tr_pcd.z)).transpose()
colors = np.vstack((tr_pcd.red, tr_pcd.green, tr_pcd.blue)).transpose()

point_clouds = o3d.geometry.PointCloud()
point_clouds.points = o3d.utility.Vector3dVector(points)
point_clouds.colors = o3d.utility.Vector3dVector(colors/255)

o3d.visualization.draw_geometries([point_clouds])

# RGB & NIR -> Arial

import rasterio
import rasterio.features
import rasterio.wrap

os.chdir(r'/tudelft.net/staff-umbrella/EDT Veluwe/testbed/bleedmareiaal/')



# Training / Validation df
