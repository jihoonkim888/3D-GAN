import os
from tqdm.auto import tqdm
import subprocess
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--obj_folder', type=str, required=True,
                    help='Location of obj files')
parser.add_argument('--save_folder', type=str, required=True,
                    help='Folder to save binvox files')
parser.add_argument('--out_dim', nargs='+', required=True,
                    help='Output dimension of binvox files. Can be a single or multiple int for multiple output res')

args = parser.parse_args()
print(args)

# find the paths of all obj files


def find_obj(obj_folder):
    lst_files = []
    for root, _, file in os.walk(obj_folder):
        for i in file:
            if i.endswith('.obj'):
                lst_files.append(os.path.join(root, i))
    return lst_files


# convert them into binvox files
def convert(lst_files, out_dim):
    if isinstance(out_dim, list):
        for res in out_dim:
            print('resolution:', res)
            # convert obj into binvox
            for file in tqdm(lst_files):
                full_path = os.path.join(file)
                # -e option gives better results somehow, especially with thin parts of an object
                cmd = f'./binvox -d {res} -rotx -cb -aw -e {full_path}'
                _ = subprocess.check_output(cmd, shell=True)

            lst_binvox = find_binvox(args.obj_folder)
            save_res_folder = os.path.join(args.save_folder, res)
            os.makedirs(save_res_folder, exist_ok=True)
            # save binvox files in a separate folder
            for i, file in enumerate(lst_binvox):
                os.system(f'mv {file} {save_res_folder}')
                original_filename = os.path.join(
                    save_res_folder, 'model_normalized.binvox')
                enum_filename = os.path.join(
                    save_res_folder, f'model_{str(i).zfill(4)}.binvox')
                os.system(f'mv {original_filename} {enum_filename}')
            print(f'binvox files of res {res} saved to {save_res_folder}')

    else:
        print('--out_dim must be int or a list of int')
        raise TypeError


def find_binvox(obj_folder):
    # binvox program saves binvox files where obj files were, so move them to a new folder
    # find the paths of binvox files
    lst_binvox = []
    for root, _, file in os.walk(obj_folder):
        for i in file:
            if i.endswith('.binvox'):
                lst_binvox.append(os.path.join(root, i))
    return lst_binvox


def remove_residual_binvox(obj_folder):
    lst_binvox = find_binvox(args.obj_folder)
    if len(lst_binvox) > 0:
        print(len(lst_binvox), 'residual binvox files found from source')
        for i in lst_binvox:
            os.system(f'rm {i}')
        print('Residual binvox files deleted')
    return


if __name__ == '__main__':
    remove_residual_binvox(args.obj_folder)
    os.makedirs(args.save_folder, exist_ok=True)
    lst_obj = find_obj(args.obj_folder)
    print(len(lst_obj), 'obj files found from', args.obj_folder)
    convert(lst_obj, args.out_dim)
    print('Done!')
