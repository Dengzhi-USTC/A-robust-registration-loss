from loss import Sample_neighs
import igl
import numpy as np

"Generate your own dataset, we only need add the neighs points in your dataset."

src_obj_path_lists = []
tar_obj_path_lists = []
src_neigh_obj_path_lists = []
tar_neigh_obj_path_lists = []

num_sample = 5000, num_neigh = 3, device = 'cuda:0'

for i in range(len(src_neigh_obj_path_lists)):
    tp_src_obj_path = src_obj_path_lists[i]
    tp_tar_obj_path = tar_obj_path_lists[i]
    tp_src_neigh_obj_path = src_neigh_obj_path_lists[i]
    tp_tar_neigh_obj_path = tar_neigh_obj_path_lists[i]

    src_points, _ = igl.read_triangle_mesh(tp_src_obj_path)
    tar_points, _ = igl.read_triangle_mesh(tp_tar_obj_path)

    src_neigh_points = Sample_neighs(src_points,
                                     num_sample=num_sample,
                                     num_neigh=num_neigh,
                                     device=device)
    tar_neigh_points = Sample_neighs(tar_points,
                                     num_sample=num_sample,
                                     num_neigh=num_neigh,
                                     device=device)

    tp_F = np.zeros([1, 3], np.int32)
    igl.write_triangle(tp_src_neigh_obj_path, src_neigh_points, tp_F)
    igl.write_triangle(tp_tar_neigh_obj_path, tar_neigh_points, tp_F)
