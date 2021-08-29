# tide our the loss!
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from LieAlgebra import *
import utils as utils

torch.pi = torch.acos(torch.zeros(1)).item() * 2
'''
For our metric:
Our metric is not an accurate calculation of the expectations designed in the paper. In the implementation, based on the consideration of parallelization, we have made a certain degree of approximation.
1.We set the maximum number of intersection points and the minimum number of intersection points between the line and the point cloud.
2.Instead of using a sparse point cloud to calculate the intersection point, we use a dense-based point cloud.
(And according to our experience, the denser the point cloud, the better the result, approximate the underline surface)
# If the density of the point cloud is not uniform, it may also produce unfavorable results
'''


def Welsch1(x, c):
    return 1 - torch.exp(-((x / c)) / 2.0)


# data:(B, N, 9) idx:(B, l)
def Batch_index_select(data, idx):
    return torch.cat(
        [torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(data, idx)],
        0)


def Select_tensors(data, idx_batch, idx_faces):
    data = torch.index_select(data, 0, idx_batch)
    data = Batch_index_select(data, idx_faces)
    return data


# calculate the distance map!
def compute_sqrdis_map_2(points_x, points_y):
    '''
    points_x : batchsize * M * 3
    points_y : batchsize * N * 3
    output   : batchsize * M * N
    '''
    thisbatchsize = points_x.size()[0]
    pn_x = points_x.size()[1]
    pn_y = points_y.size()[1]
    batch_points_x = points_x.reshape(thisbatchsize, pn_x, 1,
                                      3).expand(-1, -1, pn_y, -1)
    batch_points_y = points_y.reshape(thisbatchsize, 1, pn_y,
                                      3).expand(-1, pn_x, -1, -1)
    dis = torch.sum((batch_points_x - batch_points_y)**2, -1)
    return dis


# For raw point cloud!
'''
Input:
    point_neis:(B, n_num, nnei*3)
    line:(B, num_sample, 6)
    the nnei set 3 in our experiments!
Output: 
points: expand points
norm_d: weights
label_intersection_1: intersections
'''


def cal_intersection_batch2_points_with_line(point_neis, line):
    if len(point_neis.shape) != 3 or len(line.shape) != 3:
        print("Input is wrong")
        exit(0)
    nf = point_neis.shape[1]
    nl = line.shape[1]
    nnei = point_neis.shape[2] // 3
    line = line.unsqueeze(2)
    line = line.expand(-1, -1, nf, -1)
    points1 = point_neis.view(-1, nf, nnei, 3)
    points = points1.unsqueeze(1)
    points = points.expand(-1, nl, -1, -1, -1)
    line_pts = line[:, :, :, 3:].view(-1, nl, nf, 1,
                                      3).expand(-1, -1, -1, nnei, -1)
    line_dirs = line[:, :, :, :3].view(-1, nl, nf, 1,
                                       3).expand(-1, -1, -1, nnei, -1)
    AC = (points - line_pts)
    Proj_AC = torch.sum(AC * line_dirs, -1)**2

    d_AC = torch.sum(AC * AC, -1)
    d = torch.sqrt(d_AC - Proj_AC + 2e-4)
    if torch.sum(torch.isnan(d)) > 0:
        print("Exit the systerm")
        exit(0)
    norm_d = d / torch.sum(d, -1, keepdim=True)
    # the delta is local adaptive! better than paper.
    delta = torch.mean(
        torch.cat([
            torch.sqrt(
                torch.sum((point_neis[:, :, 3 * i:3 * (i + 1)] -
                           point_neis[:, :, :3])**2, -1).unsqueeze(2))
            for i in range(1, nnei)
        ] + [
            torch.sqrt(
                torch.sum((point_neis[:, :, 3:6] - point_neis[:, :, 6:9])**2,
                          -1)).unsqueeze(2)
        ], -1), -1)
    delta_batch = delta.unsqueeze(1).expand(-1, nl, -1).unsqueeze(-1)
    # generate the intersections!
    label_intersection_1 = torch.sum(
        torch.cat([
            d[:, :, :, i:i + 1] < delta_batch * 1.731 / 2 for i in range(nnei)
        ], -1), -1) == nnei
    return points.reshape(-1, nf, nnei * 3), norm_d.reshape(
        -1, nf, nnei).detach(), label_intersection_1


def cal_loss_intersection_batch_m_n_median_pts_lines(
        m, n, points_informations_1, points_informations_2,
        label_intersection_sum_2_2, nface1, nface2):
    nnei = 3
    #(Length, 2): represents the (b, l)
    idx_label_intersection_sum_2_2 = label_intersection_sum_2_2.nonzero(
    ).reshape(-1)
    # When, we don't have those situation, we need to ignore this.
    if idx_label_intersection_sum_2_2.shape[0] == 0:
        return None
    label_intersection_1_sum_idx_faces_2 = torch.index_select(
        points_informations_1[2].view(-1, nface1), 0,
        idx_label_intersection_sum_2_2).nonzero()[:, 1].reshape(-1, m)

    label_intersection_2_sum_idx_faces_2 = torch.index_select(
        points_informations_2[2].view(-1, nface2), 0,
        idx_label_intersection_sum_2_2).nonzero()[:, 1].reshape(-1, n)

    barycenteric_intersections_weights_1 = torch.cat([
        Select_tensors(points_informations_1[1][:, :, i:i + 1],
                       idx_label_intersection_sum_2_2,
                       label_intersection_1_sum_idx_faces_2)
        for i in range(nnei)
    ], -1)

    barycenteric_intersections_weights_2 = torch.cat([
        Select_tensors(points_informations_2[1][:, :, i:i + 1],
                       idx_label_intersection_sum_2_2,
                       label_intersection_2_sum_idx_faces_2)
        for i in range(nnei)
    ], -1)

    points1_1_batch = Select_tensors(points_informations_1[0],
                                     idx_label_intersection_sum_2_2,
                                     label_intersection_1_sum_idx_faces_2)
    points2_2_batch = Select_tensors(points_informations_2[0],
                                     idx_label_intersection_sum_2_2,
                                     label_intersection_2_sum_idx_faces_2)

    # # we generate the intersections based on the points with lines!
    points_intersections_1_batch = torch.mean(
        torch.cat([(barycenteric_intersections_weights_1[:, :, i:i + 1] *
                    points1_1_batch[:, :, 3 * i:3 * (i + 1)]).unsqueeze(-1)
                   for i in range(nnei)], -1), -1)

    points_intersections_2_batch = torch.mean(
        torch.cat([(barycenteric_intersections_weights_2[:, :, i:i + 1] *
                    points2_2_batch[:, :, 3 * i:3 * (i + 1)]).unsqueeze(-1)
                   for i in range(nnei)], -1), -1)

    distance_map = compute_sqrdis_map_2(points_intersections_1_batch,
                                        points_intersections_2_batch)
    return distance_map


def cal_loss_intersection_batch_whole_median_pts_lines(s_m,
                                                       s_n,
                                                       e_m,
                                                       e_n,
                                                       points1,
                                                       points2,
                                                       line,
                                                       device='cpu'):

    num1 = points1.shape[1]
    num2 = points2.shape[1]
    points_informations_1 = cal_intersection_batch2_points_with_line(
        point_neis=points1, line=line)
    points_informations_2 = cal_intersection_batch2_points_with_line(
        point_neis=points2, line=line)
    label_intersection_1_sum = torch.sum(points_informations_1[2], -1)
    label_intersection_2_sum = torch.sum(points_informations_2[2], -1)
    loss = torch.FloatTensor([0]).to(device)
    Flag = 0
    distance_map_list = {str(i): [] for i in range(points1.shape[0])}
    distance_map_list_vector = {str(i): [] for i in range(points1.shape[0])}
    weights_k_j_list = {str(i): [] for i in range(points1.shape[0])}
    for k in range(s_m, e_m):
        for j in range(s_n, e_n):
            # j = k
            label_intersection = (label_intersection_1_sum
                                  == k) * (label_intersection_2_sum == j)
            batch_intersection_lines = torch.sum(label_intersection, -1)

            distance_map = cal_loss_intersection_batch_m_n_median_pts_lines(
                k,
                j,
                points_informations_1,
                points_informations_2,
                label_intersection.reshape(-1),
                nface1=num1,
                nface2=num2)
            if distance_map is not None:
                t = 0
                for i1 in range(batch_intersection_lines.shape[0]):
                    distance_map_list_vector[str(i1)].append(
                        distance_map[t:t +
                                     batch_intersection_lines[i1]].reshape(
                                         1, -1))
                    t = t + batch_intersection_lines[i1]
                weights_k_j = torch.exp(torch.FloatTensor([-0.5 * abs(k - j)
                                                           ])).to(device)
                weights_k_j_list[str(i1)].append(weights_k_j)

                distance_map_list[str(i1)].append(distance_map)
                Flag += 1
    if Flag > 0:
        for ib in range(points1.shape[0]):
            median = (torch.median(
                torch.cat(distance_map_list_vector[str(ib)], -1))).detach()
            for i in range(len(distance_map_list[str(ib)])):
                sqrdis = Welsch1(distance_map_list[str(ib)][i], 1 * median)
                loss += weights_k_j_list[str(ib)][i] * (
                    torch.mean(torch.min(sqrdis, 2)[0]) +
                    torch.mean(torch.min(sqrdis, 1)[0]))
        return loss / (i + 1)
    else:
        return None, None, None


# For Chamfer & Welsch-Chamfer
def chamfer_dist(points_x, points_y):
    '''
    points_x : batchsize * M * 3
    points_y : batchsize * N * 3
    output   : batchsize * M, batchsize * N
    '''
    thisbatchsize = points_x.size()[0]
    sqrdis = compute_sqrdis_map_2(points_x, points_y)
    dist1 = sqrdis.min(dim=2)[0].view(thisbatchsize, -1)
    dist2 = sqrdis.min(dim=1)[0].view(thisbatchsize, -1)
    # idx1 = torch.argmin(sqrdis, dim=2).reshape(-1)
    # idx2 = torch.argmin(sqrdis, dim=1).reshape(-1)
    tp_dist1 = dist1.reshape(-1)
    tp_dist2 = dist2.reshape(-1)
    tp_dist = torch.cat([tp_dist1, tp_dist2], 0)
    loss = torch.mean(tp_dist)
    return loss


# Random lines
# Regarding the random generation of straight lines, we also have some strategies to assist in generating more effective straight intersected lines.
# generate the bounding box
# (B, V, 3)
# (B, 8, 3)
'''
The simplified version of the intersection method is used to judge the rough intersection of a straight line and a point cloud; May use a tree-based structure to make it more efficient!
'''


def cal_intersection_batch2_step1(points, line):
    if len(points.shape) != 3 or len(line.shape) != 3:
        print("Input is wrong")
        exit(0)
    nf = points.shape[1]
    nl = line.shape[1]
    line = line.unsqueeze(2)
    line = line.expand(-1, -1, nf, -1)
    normals = torch.cross(points[:, :, 3:6] - points[:, :, :3],
                          points[:, :, 6:9] - points[:, :, :3],
                          dim=-1)
    S = torch.norm(normals, p=2, dim=-1).detach()
    normals = torch.nn.functional.normalize(normals, p=2, dim=-1).detach()

    normals = normals.unsqueeze(1)

    normals = normals.expand(-1, nl, -1, -1)
    S = S.unsqueeze(1)
    S = S.expand(-1, nl, -1)

    points = points.unsqueeze(1)
    points = points.expand(-1, nl, -1, -1)

    #(B, N*num)
    param_t1 = (torch.sum(normals *
                          (points[:, :, :, :3] - line[:, :, :, 3:]), -1) /
                (torch.sum(normals * line[:, :, :, :3], -1) + 1e-12))

    param_t1 = param_t1.unsqueeze(3)
    param_t1 = param_t1.expand(-1, -1, -1, 3)
    intersection1 = param_t1 * line[:, :, :, :3] + line[:, :, :, 3:]
    center2A_1 = (intersection1 - points[:, :, :, :3]).detach()
    center2B_1 = (intersection1 - points[:, :, :, 3:6]).detach()
    center2C_1 = (intersection1 - points[:, :, :, 6:9]).detach()
    return center2A_1, center2B_1, center2C_1, S


def cal_intersection_batch2_step2(center2A_1, center2B_1, center2C_1, S):
    barycenteric_A_1 = torch.norm(torch.cross(center2B_1, center2C_1, dim=-1),
                                  p=2,
                                  dim=-1)
    barycenteric_B_1 = torch.norm(torch.cross(center2C_1, center2A_1, dim=-1),
                                  p=2,
                                  dim=-1)
    barycenteric_C_1 = torch.norm(torch.cross(center2A_1, center2B_1, dim=-1),
                                  p=2,
                                  dim=-1)
    # (B, Nl, Bf)
    label_intersection_1 = (barycenteric_A_1 > 0) * (barycenteric_B_1 > 0) * (
        barycenteric_C_1 >
        0) * (barycenteric_A_1 + barycenteric_B_1 + barycenteric_C_1 <= S)
    return torch.sum(label_intersection_1, -1)


def cal_intersection_batch2_rand_lines(points, line):
    center2A_1, center2B_1, center2C_1, S = cal_intersection_batch2_step1(
        points, line)
    return cal_intersection_batch2_step2(center2A_1, center2B_1, center2C_1, S)


def generate_bbox(vertices):
    minV = torch.min(vertices, dim=1)[0]
    maxV = torch.max(vertices, dim=1)[0]
    bbox = torch.zeros(minV.shape[0], 8, 3)
    bbox[:, 0, :] = maxV

    bbox[:, 1, :2] = maxV[:, :2]
    bbox[:, 1, 2] = minV[:, 2]

    bbox[:, 2, 0] = maxV[:, 0]
    bbox[:, 2, 1] = minV[:, 1]
    bbox[:, 2, 2] = maxV[:, 2]

    bbox[:, 3, 0] = maxV[:, 0]
    bbox[:, 3, 1:] = minV[:, 1:]

    bbox[:, 4, 0] = minV[:, 0]
    bbox[:, 4, 1:] = maxV[:, 1:]

    bbox[:, 5, 0] = minV[:, 0]
    bbox[:, 5, 1] = maxV[:, 1]
    bbox[:, 5, 2] = minV[:, 2]

    bbox[:, 6, :2] = minV[:, :2]
    bbox[:, 6, 2] = maxV[:, 2]
    bbox[:, 7, :] = minV
    return bbox


def generate_mesh_by_bbox(bbox, device='cpu'):
    faces = torch.from_numpy(
        np.array(
            [[2, 0, 6], [0, 4, 6], [5, 4, 0], [5, 0, 1], [6, 4, 5], [5, 7, 6],
             [3, 0, 2], [1, 0, 3], [3, 2, 6], [6, 7, 3], [5, 1, 3], [3, 7, 5]],
            dtype=np.int64)).unsqueeze(0).expand(bbox.shape[0], -1,
                                                 -1).to(device)
    batch_fvs = utils.makefacevertices(bbox, faces)
    return batch_fvs


def generate_lines(lines, new_lines, label, counter):
    for i in range(lines.shape[0]):
        label_i = label[i].nonzero()
        if counter[str(i)] > lines.shape[1]:
            continue
        counter[str(i)] += label_i.shape[0]

        if counter[str(i)] > lines.shape[1]:
            idx = lines.shape[1] - counter[str(i)] + label_i.shape[0]
            lines[i, counter[str(i)] -
                  label_i.shape[0]:, :] = new_lines[i][label_i[:idx]].reshape(
                      -1, 6)
        else:
            lines[i, counter[str(i)] - label_i.
                  shape[0]:counter[str(i)], :] = new_lines[i][label_i].reshape(
                      -1, 6)
    return lines, counter


def Random_uniform_distribution_lines_batch_efficient(r,
                                                      centers,
                                                      N,
                                                      device='cpu'):
    thisbs = r.shape[0]
    if len(r.shape) == 2:
        r = r.expand(-1, N).unsqueeze(2)
    else:
        r = r.unsqueeze(1).expand(-1, N).unsqueeze(2)

    alpha1 = (torch.rand(thisbs, N) * 2 * torch.pi).to(device).unsqueeze(2)
    u = (torch.rand(thisbs, N) * 2 - 1.0).to(device).unsqueeze(2)
    Q_u_a1 = torch.cat([
        r * torch.sqrt(1 - u * u) * torch.cos(alpha1),
        r * torch.sin(alpha1) * torch.sqrt(1 - u * u), r * u
    ], -1)

    alpha2 = (torch.rand(thisbs, N) * 2 * torch.pi).to(device).unsqueeze(2)
    u = (torch.rand(thisbs, N) * 2 - 1.0).to(device).unsqueeze(2)
    Q_u_a2 = torch.cat([
        r * torch.sqrt(1 - u * u) * torch.cos(alpha2),
        r * torch.sin(alpha2) * torch.sqrt(1 - u * u), r * u
    ], -1)

    direction = (Q_u_a2 - Q_u_a1)
    x0 = (Q_u_a1 + centers.reshape(-1, 3).unsqueeze(1).expand(-1, N, -1))
    direction = torch.nn.functional.normalize(direction, p=2, dim=-1)
    lines = torch.cat([direction, x0], -1)
    return lines


def Random_uniform_distribution_lines_batch_efficient_resample(
        r, centers, N, vertices1, vertices2, device='cpu'):
    # we also can apply more times;
    # we re sample at last 10 times!
    bbox1 = generate_bbox(vertices1).to(device)
    bbox2 = generate_bbox(vertices2).to(device)
    batch_fvs1 = generate_mesh_by_bbox(bbox1, device)
    batch_fvs2 = generate_mesh_by_bbox(bbox2, device)
    lines = torch.zeros(r.shape[0], N, 6).to(device)
    counter = {str(i): 0 for i in range(r.shape[0])}
    for i in range(10):
        new_lines = Random_uniform_distribution_lines_batch_efficient(
            r, centers, N, device)
        label1 = cal_intersection_batch2_rand_lines(batch_fvs1, new_lines)
        label2 = cal_intersection_batch2_rand_lines(batch_fvs2, new_lines)
        label = label1 * label2
        lines, counter = generate_lines(lines, new_lines, label, counter)
    return lines


# Optimized single case;
# Optimized based on a lie algebraic transformation
class Reconstruction_point(nn.Module):
    def __init__(self, rotation=None, translation=None):
        super(Reconstruction_point, self).__init__()
        if rotation is None or translation is None:
            tp = np.random.randn(3)
            tp = tp / np.linalg.norm(tp)
            tp_translation = np.random.randn(3) * 0.001
            self.parameters_ = nn.Parameter(
                torch.from_numpy(
                    np.concatenate([0.001 * tp, tp_translation],
                                   0).astype(np.float32)))
        else:
            Trans = torch.zeros(4, 4)
            Trans[:3, :3] = rotation.reshape(3, 3)
            Trans[:3, 3] = translation.reshape(3)
            tp = torch.rand(6) * 0.6
            self.parameters_ = nn.Parameter(se3.log(Trans).reshape(-1) + tp)

    def Transform(self):
        return se3.exp3(self.parameters_)

    def forward(self, points, points_neighbors):
        R, T = self.Transform()
        update_points = points @ R + T.reshape(1, 1, 3)
        points_neighbors = points_neighbors @ R + T.reshape(1, 1, 3)

        return update_points.reshape(-1, 3), points_neighbors.reshape(-1, 9)


# sample_points from the raw_points
# generate the neighbour points;
import torch
from scipy.spatial import KDTree
from sklearn.neighbors import KDTree


def Sample_neighs(points, num_sample=5000, num_neigh=3, device='cpu'):
    if points.shape[0] < num_sample:
        num_sample = points.shape[0]
    points_tensors = torch.from_numpy(points).to(device).unsqueeze(0)
    sample_points = (utils.Sample_points(points_tensors,
                                         num_sample)).cpu().squeeze().numpy()
    kdt1 = KDTree(points)
    _, idx1 = kdt1.query(sample_points, num_neigh)
    points_neighs = np.concatenate([
        points[idx1[:, 0]].reshape(-1, 3), points[idx1[:, 1]].reshape(-1, 3),
        points[idx1[:, 2]].reshape(-1, 3)
    ], -1)
    return points_neighs.reshape(-1, 3)