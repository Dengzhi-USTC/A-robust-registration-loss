import numpy as np
from scipy.spatial import KDTree
import igl
import os
import openmesh as om
from sklearn.neighbors import KDTree


def generate_neighs(points, k):
    dis_neighs = []
    kdt1 = KDTree(points, leafsize=15)
    dis1, idx1 = kdt1.query(points, k + 1)
    points_neighs = np.concatenate([
        points[idx1[:, 0]].reshape(-1, 1, 3), points[idx1[:, 1]].reshape(
            -1, 1, 3), points[idx1[:, 2]].reshape(-1, 1, 3)
    ], -2)
    points_neighs1 = np.concatenate([
        points[idx1[:, 0]].reshape(-1, 1, 3), points[idx1[:, 1]].reshape(
            -1, 1, 3), points[idx1[:, 3]].reshape(-1, 1, 3)
    ], -2)
    points_neighs2 = np.concatenate([
        points[idx1[:, 0]].reshape(-1, 1, 3), points[idx1[:, 2]].reshape(
            -1, 1, 3), points[idx1[:, 3]].reshape(-1, 1, 3)
    ], -2)

    points_neighs = np.concatenate(
        [points_neighs, points_neighs1, points_neighs2], 0)
    dis_neighs = np.concatenate(
        [dis1[:, x + 1].reshape(-1, 1) for x in range(dis1.shape[1] - 1)], -1)

    dis = np.mean(dis_neighs)
    return points_neighs, dis


# data_list: contained *.obj, we replace the data_list
def Generate_point_based_datasets(data_list):
    for _, tp_data_path in enumerate(data_list):
        tp_data_path_neigh = tp_data_path.replace('.ply', '_neigh.bin', 1)
        tp_data_path_sample = tp_data_path.replace('.ply', '.obj', 1)

        tp_data_path_neigh_obj = tp_data_path.replace('.ply', '_neigh.obj', 1)
        tp_data_path_radiu = tp_data_path.replace('.ply', '_radiu.bin', 1)
        mesh = om.read_trimesh(tp_data_path)
        vertices = mesh.points()
        # index = np.random.choice(np.arange(vertices.shape[0]),
        #                          size=5000,
        #                          replace=False)
        # vertices = vertices[index, :]
        points_neigh, points_dis = generate_neighs(vertices, 3)
        points_neigh = points_neigh.astype(np.float32)
        points_dis = points_dis.astype(np.float32)
        points_neigh.tofile(tp_data_path_neigh)
        points_neigh_obj = points_neigh.reshape(-1, 3)
        F = np.zeros([1, 3], dtype=np.int32)
        igl.write_obj(tp_data_path_neigh_obj, points_neigh_obj, F)
        igl.write_obj(tp_data_path_sample, vertices, F)
        points_dis.tofile(tp_data_path_radiu)


# data_path: the file list of your data.
data_path = "/disk_ssd/dengzhi/registration/model/bunny/reconstruction/random_rot_src_70"
data_list_src = [os.path.join(data_path, x) for x in os.listdir(data_path)]
data_list_src = [x for x in data_list_src if 'ply' in x]

Generate_point_based_datasets(data_list_src)