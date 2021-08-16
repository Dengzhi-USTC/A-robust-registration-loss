# Generate the dataset pairs!
from sklearn.neighbors import KDTree
import trimesh
import igl
import os
import math
import numpy as np

from data_utils import Scale_data, Render_data, make_rotate, R_T_compose, Make_sample_data


# Step1.
def Generate_best_connected_area(vertices, faces, normals):
    meshexport = trimesh.Trimesh(vertices, faces, normals)
    connected_comp = meshexport.split(only_watertight=False)
    max_area = 0
    max_comp = None
    for comp in connected_comp:
        if comp.area > max_area:
            max_area = comp.area
            max_comp = comp
    meshexport = max_comp
    return {
        'mesh': meshexport,
        'verts': meshexport.vertices,
        'faces': meshexport.faces
    }


# we use the data_path_files_list to illustrate this!
# we ignore the files_name;
def Scale_datasets(source_path, target_scale_path):
    if os.path.exists(source_path) is False:
        print("Input is wrong!")
        exit(0)
    if os.path.exists(target_scale_path) is False:
        print("Create new file")
    raw_mesh_files = os.listdir(source_path)
    for i, mesh_name in enumerate(raw_mesh_files):
        mesh = trimesh.load_mesh(os.path.join(source_path, mesh_name))

        results = Generate_best_connected_area(mesh.vertices, mesh.faces,
                                               mesh.face_normals)
        # mesh.export(os.path.join(target_path_connected, str(i) + ".obj"))
        V, F = results['verts'], results['faces']
        scale = (V.max(0) - V.min(0)).max()
        center = V.mean(0)
        V = Scale_data(V, center, scale)
        igl.write_triangle_mesh(
            os.path.join(target_scale_path,
                         str(i) + ".obj"), V, F)


# Step2
#
def Create_datasets_multiview(num=0,
                              num_views=50,
                              ori_path='scale_datasets_path',
                              target_path=None):
    Render_dataset = Render_data()
    # ori_path = os.path.join(local_path, "Human/human_scale_dataset")
    # target_path = os.path.join(local_path, "Human/Human_datasets_new")
    sub_paths = os.listdir(ori_path)
    sub_paths.sort()
    mesh_paths = [os.path.join(ori_path, x) for x in sub_paths]
    mesh_target_sim_paths = [
        os.path.join(target_path,
                     str(x) + "target.obj") for x in range(len(sub_paths))
    ]

    mesh_lists = [trimesh.load_mesh(path) for path in mesh_paths]
    total_mesh_lists = []
    camera_poses = []
    target_points_paths = []
    target_depth_mesh_paths = []
    target_paths = []
    for mesh_idx in range(len(mesh_lists)):
        for view_idx in range(num_views):
            total_mesh_lists.append(mesh_lists[mesh_idx])
            x_y_z_angle = (np.random.rand(3) - 0.5) * 150
            R0 = make_rotate(math.radians(x_y_z_angle[0]),
                             math.radians(x_y_z_angle[1]),
                             math.radians(x_y_z_angle[2]))

            T = R0 @ np.array([0, 0, 18.358])
            camera_poses.append(R_T_compose(R0, T))
            target_paths.append(
                os.path.join(
                    target_path,
                    str(num) + "_" + str(mesh_idx) + "_" + str(view_idx) +
                    ".exr"))
            target_points_paths.append(
                os.path.join(
                    target_path,
                    str(num) + "_" + str(mesh_idx) + "_" + str(view_idx) +
                    ".obj"))

            target_depth_mesh_paths.append(
                os.path.join(
                    target_path,
                    str(num) + "_" + str(mesh_idx) + "_" + str(view_idx) +
                    "depth_map.obj"))
    Render_dataset.render_data(total_mesh_lists,
                               camera_poses,
                               target_paths,
                               target_points_paths=target_points_paths,
                               partial_mesh_paths=target_depth_mesh_paths)
    for path, target_path in zip(mesh_paths, mesh_target_sim_paths):
        V, F = igl.read_triangle_mesh(path)
        igl.write_triangle_mesh(target_path, V, F)


def Decimate_meshes(source_paths, target_paths):
    for path, target_path in zip(source_paths, target_paths):
        os.system("meshlabserver -i %s -o %s -s %s" %
                  (path, target_path,
                   os.path.join("/home/jack/v_100_data_b/registration",
                                "Decimate.mlx")))


# We simplify the mesh to generate more real data!
def Decimate_data(num=0, target_path="target_path_view_obj"):
    source_paths_partial = []
    target_paths_partial = []
    # target_paths_partial_hull = []
    num_views = 50
    num_mesh = 100
    for mesh_idx in range(0, num_mesh):
        for view_idx in range(num_views):
            source_paths_partial.append(
                os.path.join(
                    target_path,
                    str(num) + "_" + str(mesh_idx) + "_" + str(view_idx) +
                    "depth_map.obj"))
            target_paths_partial.append(
                os.path.join(
                    target_path,
                    str(num) + "_" + str(mesh_idx) + "_" + str(view_idx) +
                    "depth_map_sim.obj"))
    Decimate_meshes(source_paths_partial, target_paths_partial)


#


# Step3:
# make rotation for data!
# rotate [-45, 45]
# translation [-1, 1]
# src_mesh_idx_view_idx
def Make_rotate_data(num0,
                     num1,
                     target_path='target_path_obj_num_sim',
                     new_target_path="generate the data pairs"):

    if os.path.exists(new_target_path) is False:
        os.makedirs(new_target_path)
    # target_path = os.path.join(local_path, "Human/datasets")
    # new_target_path = os.path.join(local_path, "Human/datasets1")
    num_views = 50
    num_mesh = 100
    for mesh_idx in range(0, num_mesh):
        for view_idx in range(num_views):
            source_paths_partial = os.path.join(
                target_path,
                str(num0) + "_" + str(mesh_idx) + "_" + str(view_idx) +
                "depth_map_sim.obj")
            target_paths_partial = os.path.join(
                target_path,
                str(num1) + "_" + str(mesh_idx) + "_" + str(view_idx) +
                "depth_map_sim.obj")
            V_src, F_src = igl.read_triangle_mesh(source_paths_partial)
            V_tar, F_tar = igl.read_triangle_mesh(target_paths_partial)

            # rotate previous? why for data preparessing!
            x_y_z_angle_raw = (np.random.rand(3) - 0.5) * 20
            R0_raw = make_rotate(math.radians(x_y_z_angle_raw[0]),
                                 math.radians(x_y_z_angle_raw[1]),
                                 math.radians(x_y_z_angle_raw[2]))

            V_src = V_src @ R0_raw
            V_tar = V_tar @ R0_raw
            #
            x_y_z_angle_src = (np.random.rand(3) - 0.5) * 0
            R0_src = make_rotate(math.radians(x_y_z_angle_src[0]),
                                 math.radians(x_y_z_angle_src[1]),
                                 math.radians(x_y_z_angle_src[2]))
            T0_src = np.random.rand(3) * 0
            x_y_z_angle_tar = (np.random.rand(3) - 0.5) * 90
            R0_tar = make_rotate(math.radians(x_y_z_angle_tar[0]),
                                 math.radians(x_y_z_angle_tar[1]),
                                 math.radians(x_y_z_angle_tar[2]))
            T0_tar = np.random.rand(3) * 1
            new_V_src = V_src @ R0_src + T0_src
            new_V_tar = V_tar @ R0_tar + T0_tar

            igl.write_triangle_mesh(
                os.path.join(
                    new_target_path,
                    "src_" + str(mesh_idx) + "_" + str(view_idx) + ".obj"),
                new_V_src, F_src)
            igl.write_triangle_mesh(
                os.path.join(
                    new_target_path,
                    "tar_" + str(mesh_idx) + "_" + str(view_idx) + ".obj"),
                new_V_tar, F_tar)

            new_R0_tar = R0_src.transpose(1, 0) @ R0_tar
            new_T0_tar = -T0_src @ new_R0_tar + T0_tar
            new_R0_tar.tofile(
                os.path.join(
                    new_target_path,
                    "ta_R_" + str(mesh_idx) + "_" + str(view_idx) + ".bin"))
            new_T0_tar.tofile(
                os.path.join(
                    new_target_path,
                    "ta_T_" + str(mesh_idx) + "_" + str(view_idx) + ".bin"))


# Step4
# generate the sample_pts;
def Sample_mesh(new_taget_path):

    sub_files = os.listdir(new_taget_path)
    src_paths = [
        os.path.join(new_taget_path, x) for x in sub_files if "src" in x
    ]

    tar_paths = [
        os.path.join(new_taget_path, x) for x in sub_files if "tar" in x
    ]
    src_sample_paths = [
        os.path.join(new_taget_path, x.replace("src", "src_sample"))
        for x in sub_files if "src" in x
    ]
    tar_sample_paths = [
        os.path.join(new_taget_path, x.replace("tar", "tar_sample"))
        for x in sub_files if "tar" in x
    ]
    # Pay attention current is for partial datasets!
    Make_sample_data(src_paths, src_sample_paths, num=1024)
    Make_sample_data(tar_paths, tar_sample_paths, num=1024)


def Consistency_Faces(src_mesh_list,
                      tar_mesh_list,
                      Num_faces,
                      Num_vertices=3000):
    for src_mesh_path, tar_mesh_path in zip(src_mesh_list, tar_mesh_list):
        V_src, F_src = igl.read_triangle_mesh(src_mesh_path)
        if F_src.shape[0] > Num_faces:
            Num_faces = F_src.shape[0] + 1
        if V_src.shape[0] > Num_vertices:
            Num_vertices = V_src.shape[0] + 1

    for src_mesh_path, tar_mesh_path in zip(src_mesh_list, tar_mesh_list):
        V_src, F_src = igl.read_triangle_mesh(src_mesh_path)
        if F_src.shape[0] < Num_faces:
            F_src = np.concatenate([
                F_src,
                np.ones([Num_faces - F_src.shape[0], 3]).astype(np.int32)
            ], 0)
        if V_src.shape[0] < Num_vertices:
            V_src = np.concatenate([
                V_src,
                np.zeros([Num_vertices - V_src.shape[0], 3]).astype(np.float32)
            ], 0)
        F_tar = F_src
        V_tar = V_src
        igl.write_triangle_mesh(tar_mesh_path, V_tar, F_tar)


def Generate_consistency_faces(new_target_path):

    sub_files = os.listdir(new_target_path)
    src_paths = [
        os.path.join(new_target_path, x) for x in sub_files
        if "src" in x and "sample" not in x
    ]
    tar_paths = [
        os.path.join(new_target_path, x) for x in sub_files
        if "tar" in x and "sample" not in x
    ]
    src_paths_consistent = [
        os.path.join(new_target_path, x.replace("src", "src_consistent"))
        for x in sub_files if "src" in x and "sample" not in x
    ]
    tar_paths_consistent = [
        os.path.join(new_target_path, x.replace("tar", "tar_consistent"))
        for x in sub_files if "tar" in x and "sample" not in x
    ]
    Consistency_Faces(src_mesh_list=src_paths,
                      tar_mesh_list=src_paths_consistent,
                      Num_faces=5000)
    Consistency_Faces(src_mesh_list=tar_paths,
                      tar_mesh_list=tar_paths_consistent,
                      Num_faces=5000)


# generate the overlap radio to know the data!
# Step6.
# Cal the overlap radio
# vertices1 and vertices2 are registrated well!
def Cal_overlap_radio(vertices1, vertices2):
    kdt1 = KDTree(vertices1, leaf_size=30, metric='euclidean')
    kdt2 = KDTree(vertices2, leaf_size=30, metric='euclidean')
    dis1, idx1 = kdt1(vertices2)
    dis2, idx2 = kdt2(vertices1)
    error = (np.mean(dis1) + np.mean(dis2)) / 0.5
    dis1_I = dis1 <= error
    dis2_I = dis2 <= error
    inter_radio = (np.sum(dis1_I) + np.sum(dis2_I)) / (vertices1.shape[0] +
                                                       vertices2.shape[0])

    inter_radio1 = np.sum(dis1_I) / vertices1.shape[0]

    inter_radio2 = np.sum(dis2_I) / vertices2.shape[0]
    inter_ra = {
        'inter_radio': inter_radio,
        'inter_radio1': inter_radio1,
        'inter_radio2': inter_radio2
    }
    return inter_ra


# Scale datasets
# source path: the path of the raw mesh files.
# target_scale_path: the path of the scale raw mesh files.
# new_target_path: the path of the intermediate data
# target_path_pairs: the path of the final target_pairs
source_path = ""
target_scale_path = ""
new_target_path = ""
target_path_pairs = ""
Scale_datasets(source_path, target_scale_path)  #

Create_datasets_multiview(num=0,
                          num_views=50,
                          ori_path=target_scale_path,
                          target_path=new_target_path)
Create_datasets_multiview(num=1,
                          num_views=50,
                          ori_path=target_scale_path,
                          target_path=new_target_path)
# # for local
Decimate_data(num=0, target_path=new_target_path)  # #
Decimate_data(num=1, target_path=new_target_path)  # #

Make_rotate_data(0,
                 1,
                 target_path=new_target_path,
                 new_target_path=target_path_pairs)
Sample_mesh(target_path_pairs)
Generate_consistency_faces(target_path_pairs)