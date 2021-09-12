import os
import numpy as np
import os
import sys
import glob
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils import data
from torch.utils.data import Dataset
import torch
import igl

from scipy.linalg import expm, norm


def M(axis, theta):
    return expm(np.cross(np.eye(3), axis / norm(axis) * theta))


# Input: lists path
# Output: (src_p, tgt_p, R, T, src_p_nei, tgt_p_nei, tgt_box, src_normal, tgt_normal, )
# Options (In the final convergence stage, we can consider adding some randomness to improve the robustness of the network and enhance the richness of data.)


# In order to better facilitate the data stream processing of each framework, we have generated a unified data stream processing.
class Dataset_2021_8_29(Dataset):
    def __init__(self,
                 points_files_src_sample,
                 points_files_tar_sample,
                 DCP_True=False,
                 FMR_True=False):

        self.points_files_src_sample = points_files_src_sample
        self.points_files_tar_sample = points_files_tar_sample
        self.randg = np.random.RandomState(0)
        self.DCP_True = DCP_True
        self.FMR_True = FMR_True

    # random the vertex and random the normals!
    def transform_R_T(self, points, R, T):
        return points @ R + T

    def random_data(self, data, rotation_range=30):
        R = M(
            np.random.rand(3) - 0.5, rotation_range * np.pi / 180.0 *
            (np.random.rand(1) - 0.5)).astype(np.float32)
        T = np.zeros([1, 3]).astype(np.float32)
        points_tar_sample = self.transform_R_T(data['points_tar_sample'], R, T)
        normals_tar = self.transform_R_T(data['normals_ref'], R, 0 * T)
        points_based_neighs_tar = self.transform_R_T(
            data['points_based_neighs_tar'].reshape(-1, 3), R, T)

        points_src_sample = self.transform_R_T(
            data['points_src_sample'] @ data['R'] + data['T'], R, T)
        normals_src = self.transform_R_T(data['normals_src'] @ data['R'], R,
                                         0 * T)
        points_based_neighs_src = self.transform_R_T(
            data['points_based_neighs_src'].reshape(-1, 3) @ data['R'] +
            data['T'], R, T)

        # update the src and ref!
        data['points_src_sample'] = (points_src_sample -
                                     data['T']) @ data['R'].transpose(1, 0)

        data['points_based_neighs_src'] = (
            points_based_neighs_src - data['T']) @ data['R'].transpose(1, 0)
        data['normals_src'] = normals_src @ data['R'].transpose(1, 0)

        # make a new transformed!
        data['points_tar_sample'] = points_tar_sample
        data['normals_ref'] = normals_tar
        data['points_based_neighs_tar'] = points_based_neighs_tar
        data['center'] = data['centers'] @ R + T
        data['tar_box'] = data['tar_box'] @ R + T
        return data

    def __getitem__(self, index):

        # sth input to the network;
        points_file_src_sample = self.points_files_src_sample[index]
        V_src_sample, _ = igl.read_triangle_mesh(points_file_src_sample)

        points_file_tar_sample = self.points_files_src_sample[index]
        V_tar_sample, _ = igl.read_triangle_mesh(points_file_tar_sample)

        normals_src, _ = igl.read_triangle_mesh(
            points_file_src_sample.replace("sample", "sample_normals", 1))
        points_file_tar_sample = self.points_files_tar_sample[index]
        V_tar_sample, _ = igl.read_triangle_mesh(points_file_tar_sample)
        normals_tar, _ = igl.read_triangle_mesh(
            points_file_tar_sample.replace("sample", "sample_normals", 1))

        # sth optimized the parameters of the networks;
        path_points_based_neighs_src = points_file_src_sample.replace(
            '.obj', '_neigh.bin', 1)
        path_points_based_neighs_tar = points_file_tar_sample.replace(
            '.obj', '_neigh.bin', 1)
        # point_neighs: (B, N*k, 3)
        points_based_neighs_src = np.fromfile(path_points_based_neighs_src,
                                              np.float32).reshape(
                                                  -1, 3).astype(np.float64)
        points_based_neighs_tar = np.fromfile(path_points_based_neighs_tar,
                                              np.float32).reshape(
                                                  -1, 3).astype(np.float64)
        centers_tar = V_tar_sample.mean(0)
        centers_src = V_src_sample.mean(0)
        V_tar_sample = V_tar_sample - centers_tar
        tar_box = igl.bounding_box(V_tar_sample)[0].astype(np.float32)

        V_src_sample = V_src_sample - centers_src
        points_based_neighs_src = points_based_neighs_src.copy() - centers_src
        points_based_neighs_tar = points_based_neighs_tar.copy() - centers_tar
        path_gt_transform = self.points_files_tar_sample[index].replace(
            'tar_sample', 'transform', 1).replace('.obj', '.bin', 1)
        # print(np.fromfile(path_gt_transform, np.float64).shape)
        # print(path_gt_transform)
        # print("********************************************8")
        gt_transform = np.fromfile(path_gt_transform,
                                   np.float64).astype(np.float64).reshape(
                                       3, 4)
        rotation = gt_transform[:3, :3].transpose(0, 1)
        translation = gt_transform[:3, 3]
        translation += -centers_tar + centers_src @ rotation
        centers = V_tar_sample.mean(0)
        # scale the data;
        transformed = np.eye(4, 4)
        transformed[:3, :3] = rotation
        # print(centers, center1)
        t = 1.0

        transformed[:3, 3] = -rotation @ translation / t
        data = {
            'points_tar_sample':
            V_tar_sample.astype(np.float32).transpose(0, 1) / t,
            "points_src_sample":
            V_src_sample.astype(np.float32).transpose(0, 1) / t,
            "normals_tar":
            normals_tar.astype(np.float32),
            "normals_src":
            normals_src.astype(np.float32),
            "tar_box":
            tar_box.astype(np.float32) / t,
            "centers":
            centers.astype(np.float32) / t,
            'R':
            rotation.astype(np.float32).transpose(0, 1),
            'T':
            translation.astype(np.float32) / t,
            'R_inv':
            rotation.astype(np.float32),
            'T_inv':
            -rotation.astype(np.float32) @ translation.astype(np.float32) / t,
            'points_based_neighs_src':
            points_based_neighs_src.transpose(0, 1).astype(np.float32) / t,
            'points_based_neighs_tar':
            points_based_neighs_tar.transpose(0, 1).astype(np.float32) / t,
            'igt':
            transformed.astype(np.float32)
        }
        # considerate the data flow, we need transpose the data;
        if self.DCP_True is True:
            data['points_tar_sample'] = data['points_tar_sample'].transpose(
                1, 0)
            data['points_src_sample'] = data['points_src_sample'].transpose(
                1, 0)
            data['points_based_neighs_src'] = data[
                'points_based_neighs_src'].transpose(1, 0)
            data['points_based_neighs_tar'] = data[
                'points_based_neighs_tar'].transpose(1, 0)
            data['R'] = data['R'].transpose(1, 0)
            data['R_inv'] = data['R_inv'].transpose(1, 0)
            data['igt'][:3, :3] = data['igt'][:3, :3].transpose(1, 0)
        if self.FMR_True is True:
            len = data['points_src_sample'].shape[
                0] if data['points_src_sample'].shape[0] < data[
                    'points_tar_sample'].shape[0] else data[
                        'points_tar_sample'].shape[0]
            data['points_tar_sample'] = data['points_tar_sample'][:len, ]
            data['points_src_sample'] = data['points_src_sample'][:len, ]
        return data

    def __len__(self):
        return len(self.points_files_tar_sample)


# Choose which datasets you use, please change the path;
# For Human datasets: split into the train datasets and test datasets;
#
def generate_datasets_human(DCP=False, FMR=False):
    data_path = "/data1/dengzhi/Human_dataset"
    num_mesh_start = 0
    num_mesh_end = 110
    num_view_start = 0
    num_view_end = 50

    path_points_src_sample = []
    path_points_ref_sample = []
    for mesh_idx in range(num_mesh_start, num_mesh_end):
        for view_idx in range(num_view_start, num_view_end):
            tp_src = os.path.join(
                data_path,
                "src_sample_" + str(mesh_idx) + "_" + str(view_idx) + ".obj")
            tp_tar = os.path.join(
                data_path,
                "tar_sample_" + str(mesh_idx) + "_" + str(view_idx) + ".obj")
            path_points_src_sample.append(tp_src)
            path_points_ref_sample.append(tp_tar)

    train_data_set = Dataset_2021_8_29(path_points_src_sample[:4],
                                       path_points_ref_sample[:4],
                                       DCP_True=DCP,
                                       FMR_True=FMR)
    test_data_set = Dataset_2021_8_29(path_points_src_sample[5000:5000 + 2],
                                      path_points_ref_sample[5000:5000 + 2],
                                      DCP_True=DCP,
                                      FMR_True=FMR)
    print("this is the length of the train dataset{:4f}, test dataset".format(
        train_data_set.__len__(), test_data_set.__len__()))
    train_data_loader = torch.utils.data.DataLoader(
        train_data_set,
        batch_size=4,
        shuffle=True,
        num_workers=1,
        drop_last=True,
        worker_init_fn=np.random.seed(np.random.get_state()[1][0]))
    test_data_loader = torch.utils.data.DataLoader(
        train_data_set,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=True,
        worker_init_fn=np.random.seed(np.random.get_state()[1][0]))
    return train_data_loader, test_data_loader


def generate_datasets_airplane(DCP=False, FMR=False):
    data_path = "/data1/dengzhi/Airplane"
    num_mesh_start = 0
    num_mesh_end = 625
    num_view_start = 0
    num_view_end = 18

    path_points_src_sample = []
    path_points_ref_sample = []
    for mesh_idx in range(num_mesh_start, num_mesh_end):
        for view_idx in range(num_view_start, num_view_end):
            tp_src = os.path.join(
                data_path,
                "src_sample_" + str(mesh_idx) + "_" + str(view_idx) + ".obj")
            tp_tar = os.path.join(
                data_path,
                "tar_sample_" + str(mesh_idx) + "_" + str(view_idx) + ".obj")
            path_points_src_sample.append(tp_src)
            path_points_ref_sample.append(tp_tar)

    train_data_set = Dataset_2021_8_29(path_points_src_sample[:4],
                                       path_points_ref_sample[:4],
                                       DCP_True=DCP,
                                       FMR_True=FMR)
    test_data_set = Dataset_2021_8_29(path_points_src_sample[5000:5000 + 2],
                                      path_points_ref_sample[5000:5000 + 2],
                                      DCP_True=DCP,
                                      FMR_True=FMR)
    print("this is the length of the train dataset{:4f}, test dataset".format(
        train_data_set.__len__(), test_data_set.__len__()))
    train_data_loader = torch.utils.data.DataLoader(
        train_data_set,
        batch_size=4,
        shuffle=True,
        num_workers=1,
        drop_last=True,
        worker_init_fn=np.random.seed(np.random.get_state()[1][0]))
    test_data_loader = torch.utils.data.DataLoader(
        train_data_set,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=True,
        worker_init_fn=np.random.seed(np.random.get_state()[1][0]))
    return train_data_loader, test_data_loader


def generate_datasets_airplane(DCP=False, FMR=False):
    data_path = "/data1/dengzhi/Airplane"
    num_mesh_start = 0
    num_mesh_end = 625
    num_view_start = 0
    num_view_end = 18

    path_points_src_sample = []
    path_points_ref_sample = []
    for mesh_idx in range(num_mesh_start, num_mesh_end):
        for view_idx in range(num_view_start, num_view_end):
            tp_src = os.path.join(
                data_path,
                "src_sample_" + str(mesh_idx) + "_" + str(view_idx) + ".obj")
            tp_tar = os.path.join(
                data_path,
                "tar_sample_" + str(mesh_idx) + "_" + str(view_idx) + ".obj")
            path_points_src_sample.append(tp_src)
            path_points_ref_sample.append(tp_tar)

    train_data_set = Dataset_2021_8_29(path_points_src_sample[:4],
                                       path_points_ref_sample[:4],
                                       DCP_True=DCP,
                                       FMR_True=FMR)
    test_data_set = Dataset_2021_8_29(path_points_src_sample[5000:5000 + 2],
                                      path_points_ref_sample[5000:5000 + 2],
                                      DCP_True=DCP,
                                      FMR_True=FMR)
    print("this is the length of the train dataset{:4f}, test dataset".format(
        train_data_set.__len__(), test_data_set.__len__()))
    train_data_loader = torch.utils.data.DataLoader(
        train_data_set,
        batch_size=4,
        shuffle=True,
        num_workers=1,
        drop_last=True,
        worker_init_fn=np.random.seed(np.random.get_state()[1][0]))
    test_data_loader = torch.utils.data.DataLoader(
        train_data_set,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=True,
        worker_init_fn=np.random.seed(np.random.get_state()[1][0]))
    return train_data_loader, test_data_loader


def generate_datasets_real(DCP=False, FMR=False):
    data_path = "/data1/dengzhi/Real"

    path_points_src_sample = []
    path_points_ref_sample = []
    for i in range(0, 4):
        path_points_ref_sample.append(
            os.path.join(data_path,
                         str(i) + "_tar_sample.obj"))
        path_points_src_sample.append(
            os.path.join(data_path,
                         str(i) + "_src_sample.obj"))

    train_data_set = Dataset_2021_8_29(path_points_src_sample[:4],
                                       path_points_ref_sample[:4],
                                       DCP_True=DCP,
                                       FMR_True=FMR)
    test_data_set = Dataset_2021_8_29(path_points_src_sample[5000:5000 + 2],
                                      path_points_ref_sample[5000:5000 + 2],
                                      DCP_True=DCP,
                                      FMR_True=FMR)
    print("this is the length of the train dataset{:4f}, test dataset".format(
        train_data_set.__len__(), test_data_set.__len__()))
    train_data_loader = torch.utils.data.DataLoader(
        train_data_set,
        batch_size=4,
        shuffle=True,
        num_workers=1,
        drop_last=True,
        worker_init_fn=np.random.seed(np.random.get_state()[1][0]))
    test_data_loader = torch.utils.data.DataLoader(
        train_data_set,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=True,
        worker_init_fn=np.random.seed(np.random.get_state()[1][0]))
    return train_data_loader, test_data_loader
