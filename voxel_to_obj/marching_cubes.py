import argparse
import numpy as np
import mcubes
import os
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file_path', type=str,
                    required=True, help='file path to .npy file')
parser.add_argument('-s', '--save_path', type=str, required=True,
                    help='directory to save obj files. A new directory will be created if it does not exist')
parser.add_argument('-t', '--threshold', type=float, required=False,
                    help='threshold to filter out low values in the array. The default value is ' + str(threshold))
parser.add_argument('-sm', '--smoothen', type=bool, required=False,
                    help='Whether to smoothen the mesh or not')


if __name__ == '__main__':
    args = parser.parse_args()
    file_path = args.file_path
    save_path = args.save_path
    threshold = args.threshold if args.threshold else 0.3
    smoothen = args.smoothen if args.smoothen else False

    # open the .npy file and read the numpy array
    with open(file_path, 'rb') as f:
        data = np.load(f)

    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)
        print('directory', save_path, 'created to save obj files')

    assert len(
        data.shape) == 5, 'The numpy array should be 5D with shape (batch, channel, dim, dim, dim)'
    print(data.shape[0], 'shapes detected')

    for i in tqdm(range(data.shape[0])):
        u = data[i][0]
        u = np.rot90(u, k=3, axes=(1, 2))
        u = np.rot90(u, k=1, axes=(0, 2))
        u[u < threshold] = 0
        n = str(i).zfill(4)
        if smoothen:
            u = mcubes.smooth(u)  # smoothen the binary array
        vertices, triangles = mcubes.marching_cubes(u, 0)
        obj_filename = os.path.join(save_path, 'model_'+str(n)+'.obj')
        mcubes.export_obj(vertices, triangles, obj_filename)
        # print(obj_filename, 'exported')

    print('Done!')
