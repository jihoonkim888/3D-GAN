# Fine-scale Synthetic Shape Generation with Upsampling Network

![Airplane models generated using 3D-GAN](https://user-images.githubusercontent.com/68455855/190845159-ffd73ed7-ce47-4062-89ab-b699f2ca4740.png)
![Upsampled models](https://user-images.githubusercontent.com/68455855/190845160-4442ebed-a851-4e19-ad73-710d875daca2.png)


This repository contains the code implementation of the 3D-GAN and the up-convolutional network that are used to generate synthetic samples and upsample low-resolution samples, respectively.

## Root
- train_3D_GAN.py: trains 3D-GAN used to generate synthetic shape samples
- train_upscaler.py: trains the proposed up-convolutonal network to upsample low-resolution shape samples
- main.py: generates low-resolution synthetic samples, upsample them and output 3D voxel plots of the original and upsampled shape samples
- main_128.py: generates high-resolution synthetic samples and output their 3D voxel plots
- main_real.py: plots real low and high-resolution shape samples used to train the 3D-GAN and the up-convolutional network
- main_real_ups.py: upsamples real low-resolution shape samples and output their 3D voxel plots

## Subfolders
The repository contains two subfolders, obj_to_voxel and src.
### obj_to_voxel
- binvox: program that converts a mesh (.obj) file in into an occupancy grid (.binvox)
- voxelize_obj.py: converts all the mesh files in a folder into .binvox files. it should be able to convert meshes in any formats that binvox supports.
- Dockerfile: A script to build a docker image that can run binvox and voxelize_obj.py

### src
- binvox_rw.py: A Python script that reads .binvox files as arrays (numpy)
- GAN.py: The source code of the 3D-GAN used to generate shape samples
- upscaler.py: The source code of the up-convolutional network used to upsample shape samples



You will need an NVIDIA GPU that supports CUDA with at least 4GB of RAM to run the low-res 3D-GAN or the up-convolutional network, and 8GB of VRAM to run the high-resolution 3D-GAN.

The docker image used to train and test can be found [here](nvcr.io/nvidia/pytorch:22.08-py3).
For voxelisation of mesh files, please use the Dockerfile in the subfolder.

### References
- binvox https://www.patrickmin.com/binvox/
- binvox_rw.py https://github.com/dimatura/binvox-rw-py
