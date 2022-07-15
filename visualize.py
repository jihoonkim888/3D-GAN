from simple_3dviz import Lines
from simple_3dviz import Mesh
from simple_3dviz.window import show
from src import binvox_rw
import numpy as np


with open('/home/jihoon/diss/data/shapenet-chair-binvox/256/model_0.binvox', 'rb') as f:
    real__voxels = binvox_rw.read_as_3d_array(f).data

with open('/home/jihoon/diss/data/output_data.npy', 'rb') as f:
    up_voxels = np.load(f)

for i in range(up_voxels.shape[0]):
    with open(f'/home/jihoon/diss/data/shapenet-chair-binvox/64/model_{i}.binvox', 'rb') as f:
        real_data_64 = binvox_rw.read_as_3d_array(f).data
    with open(f'/home/jihoon/diss/data/shapenet-chair-binvox/256/model_{i}.binvox', 'rb') as f:
        real_data_256 = binvox_rw.read_as_3d_array(f).data
    up_data = (up_voxels[i][0] > 0.3).astype(bool)
    print('real 64')
    m = Mesh.from_voxel_grid(
        real_data_64, colors=(0.8, 0, 0))
    show(m)
    print('real 256')
    m = Mesh.from_voxel_grid(
        real_data_256, colors=(0.8, 0, 0))
    show(m)
    print('upscaled')
    m = Mesh.from_voxel_grid(
        up_data, colors=(0, 0.8, 0))
    show(m)

# It is also possible to visualize the voxel grid with boundaries by creating
# a Lines renderable object
# l = Lines.from_voxel_grid(voxels, colors=(0, 0, 0.), width=0.01)
# show([m, l])
