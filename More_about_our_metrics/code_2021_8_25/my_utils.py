# some tools aid helping the training process and data process!
from functools import partial
from numpy.core.fromnumeric import nonzero
import openmesh as om
import numpy as np
import os
import torch
from scipy.spatial.transform import Rotation


def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
                         dim=1).reshape(B, 3, 3)
    return rotMat


def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_dcm(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')


def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


def Batch_index_select(data, idx):
    return torch.cat(
        [torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(data, idx)],
        0)


# for this operator, we can't batch, we need for it!
def makefacevertices(vertices, faces):
    if len(vertices.shape) == 2:
        vertices = vertices.reshape(1, vertices.shape[0], vertices.shape[1])
    if len(faces.shape) == 2:
        faces = faces.reshape(1, faces.shape[0], faces.shape[1])
    # update_faces_points = torch.cat([
    #     torch.index_select(vertices, 1, faces[0, :, 0]),
    #     torch.index_select(vertices, 1, faces[0, :, 1]),
    #     torch.index_select(vertices, 1, faces[0, :, 2])
    # ], -1)
    update_faces_points = torch.cat([
        Batch_index_select(vertices, faces[:, :, 0]),
        Batch_index_select(vertices, faces[:, :, 1]),
        Batch_index_select(vertices, faces[:, :, 2])
    ], -1)
    return update_faces_points


# When we generate the data, we store the (x, y, z) in the depth map!
# generate the mesh from the depth map, mainly considerate the connection!
def generate_depth_mesh(img, mask, idx=0):
    idx_ = np.nonzero(mask)
    mesh = om.TriMesh()
    if not idx_:
        return mesh
    idx_ = zip(idx_[0], idx_[1])
    idx = set()

    for x, y in idx_:
        idx.add((x, y))
    vh = dict()
    for i in range(0, mask.shape[0], 1):
        for j in range(0, mask.shape[1], 1):
            T1 = (i, j) in idx and (i, j + 1) in idx and (
                i + 1, j) in idx and (i + 1, j + 1) in idx
            T2 = ((i, j) in idx and (i - 1, j) in idx and
                  (i, j - 1) in idx) and (i - 1, j - 1) in idx
            T3 = (i, j) in idx and (i - 1, j) in idx and (
                i - 1, j + 1) in idx and (i, j + 1) in idx
            T4 = (i, j) in idx and (i + 1, j) in idx and (
                i + 1, j - 1) in idx and (i, j - 1) in idx
            if (T1 or T2 or T3 or T4) and mask[i, j] > 0 and (np.fabs(
                    4 * img[i, j, 2] - img[i - 1, j, 2] - img[i + 1, j, 2] -
                    img[i, j - 1, 2] - img[i, j + 1, 2])) < 0.1 and (
                        np.fabs(4 * img[i, j, 1] - img[i - 1, j, 1] -
                                img[i + 1, j, 1] - img[i, j - 1, 1] -
                                img[i, j + 1, 1])
                    ) < 0.1 and (np.fabs(4 * img[i, j, 0] - img[i - 1, j, 0] -
                                         img[i + 1, j, 0] - img[i, j - 1, 0] -
                                         img[i, j + 1, 0])) < 0.1:
                vh0 = mesh.add_vertex(
                    [img[i, j, 0], img[i, j, 1], img[i, j, 2]])
                vh[(i, j)] = vh0
    for i in range(0, mask.shape[0], 1):
        for j in range(0, mask.shape[1], 1):
            T1 = (i, j) in vh and (i + 1, j) in vh and (i + 1, j + 1) in vh
            T2 = (i, j) in vh and (i + 1, j + 1) in vh and (i, j + 1) in vh
            if T1:
                mesh.add_face(vh[(i, j)], vh[(i + 1, j)], vh[(i + 1, j + 1)])
            if T2:
                mesh.add_face(vh[(i, j)], vh[(i + 1, j + 1)], vh[(i, j + 1)])
    return mesh


import pyrender
import trimesh
import pyexr
# For offscreenrender we need set the enviroment!
import os, cv2, math, igl

os.environ['PYOPENGL_PLATFORM'] = 'egl'


def make_rotate(rx, ry, rz):
    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3, 3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3, 3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3, 3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz, Ry), Rx)
    return R


def R_T_compose(R, T):
    pose = np.eye(4, 4)
    pose[:3, :3] = R
    pose[:3, 3] = T
    return pose


# Generate the data!
class Render_data():
    def __init__(self):
        super().__init__()

        self.resolution = 800
        site = np.ones([self.resolution, self.resolution])
        self.X = site.nonzero()[0].reshape(self.resolution, self.resolution)
        self.Y = site.nonzero()[1].reshape(self.resolution, self.resolution)
        self.FOV = np.pi / 3.
        self.znear = 0.5
        self.w = 2 * self.znear * np.tan(self.FOV / 2.)
        self.camera = pyrender.PerspectiveCamera(yfov=self.FOV,
                                                 aspectRatio=1.0,
                                                 znear=self.znear)
        self.resolution_half = self.resolution / 2
        self.temp_mask = np.zeros_like(self.X)
        self.temp_mask[3:self.resolution - 3, 3:self.resolution - 3] = 1.0

    def render(self, mesh, camera_pose):
        scenes = pyrender.Scene()
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scenes.add(mesh)
        # print(camera_pose)
        scenes.add(self.camera, pose=camera_pose)
        light = pyrender.SpotLight(color=np.ones(3),
                                   intensity=3.0,
                                   innerConeAngle=np.pi / 16.0,
                                   outerConeAngle=np.pi / 6.0)
        scenes.add(light, pose=camera_pose)
        r = pyrender.OffscreenRenderer(self.resolution, self.resolution)
        _, depth = r.render(scenes)

        # depth 2 camera coordinates
        # Pay attention to X and Y!
        mask = (depth != 0).reshape(self.resolution, self.resolution, 1)
        mask_3 = np.concatenate([mask, mask, mask], -1)
        x_p = self.w / 2. * (self.Y -
                             self.resolution_half) / self.resolution_half
        y_p = self.w / 2. * (self.resolution_half -
                             self.X) / self.resolution_half

        X = (x_p * depth / self.znear).reshape(self.resolution,
                                               self.resolution, 1)
        Y = (y_p * depth / self.znear).reshape(self.resolution,
                                               self.resolution, 1)
        Z = depth.reshape(self.resolution, self.resolution, 1)

        cameras_points = np.concatenate([X, Y, -Z], -1).reshape(-1, 3)

        # R = make_rotate(0, 0, math.radians(theta))
        # depth 2 world coordinates!

        # considerate it into points in the image coordiantes and connected into a mesh!
        space_points_ = (camera_pose[:3, :3] @ cameras_points.transpose()
                         ).transpose() + camera_pose[:3, 3]
        # space_points_ = (cameras_points @ camera_pose[:3, :3].transpose()
        #                  ) - camera_pose[3, :3]
        space_points = (
            space_points_.reshape(self.resolution, self.resolution, 3) *
            mask_3)
        idx = mask.nonzero()
        space_points_3 = space_points[idx[0], idx[1], :]
        return space_points, mask, space_points_3

    # considerate more case!
    def render_data(self,
                    mesh_lists,
                    camera_poses,
                    target_paths,
                    write_points=True,
                    target_points_paths=None,
                    partial_mesh_paths=None):
        for i, (mesh, camera_pose, target_path) in enumerate(
                zip(mesh_lists, camera_poses, target_paths)):
            print(i)
            depth_points_image, mask, points_cloud = self.render(
                mesh, camera_pose)
            if target_path is not None:
                pyexr.write(target_path, depth_points_image)
                if write_points:
                    F = np.array([0, 0, 0]).astype(np.int32).reshape(-1, 3)
                    # igl.write_triangle_mesh(target_points_paths[i],
                    #                         points_cloud.reshape(-1, 3), F)
            if partial_mesh_paths is not None:
                # print(self.temp_mask.shape,mask.shape)
                mesh = generate_depth_mesh(
                    depth_points_image,
                    mask.reshape(self.resolution, self.resolution) *
                    self.temp_mask)
                om.write_mesh(partial_mesh_paths[i], mesh)
        return None
        


import openmesh as om


# Generate the data!
class Render_data_gt_correspondence():
    def __init__(self):
        super().__init__()

        self.resolution = 800
        site = np.ones([self.resolution, self.resolution])
        self.X = site.nonzero()[0].reshape(self.resolution, self.resolution)
        self.Y = site.nonzero()[1].reshape(self.resolution, self.resolution)
        self.FOV = np.pi / 3.
        self.znear = 0.5
        self.w = 2 * self.znear * np.tan(self.FOV / 2.)
        self.camera = pyrender.PerspectiveCamera(yfov=self.FOV,
                                                 aspectRatio=1.0,
                                                 znear=self.znear)
        self.resolution_half = self.resolution / 2

    def render_depth(self, mesh, camera_pose):
        scenes = pyrender.Scene()
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scenes.add(mesh)
        # print(camera_pose)
        scenes.add(self.camera, pose=camera_pose)
        light = pyrender.SpotLight(color=np.ones(3),
                                   intensity=3.0,
                                   innerConeAngle=np.pi / 16.0,
                                   outerConeAngle=np.pi / 6.0)
        scenes.add(light, pose=camera_pose)
        r = pyrender.OffscreenRenderer(self.resolution, self.resolution)
        _, depth = r.render(scenes)

        # depth 2 camera coordinates
        # Pay attention to X and Y!
        mask = (depth != 0).reshape(self.resolution, self.resolution, 1)
        mask_3 = np.concatenate([mask, mask, mask], -1)
        x_p = self.w / 2. * (self.Y -
                             self.resolution_half) / self.resolution_half
        y_p = self.w / 2. * (self.resolution_half -
                             self.X) / self.resolution_half

        X = (x_p * depth / self.znear).reshape(self.resolution,
                                               self.resolution, 1)
        Y = (y_p * depth / self.znear).reshape(self.resolution,
                                               self.resolution, 1)
        Z = depth.reshape(self.resolution, self.resolution, 1)

        cameras_points = np.concatenate([X, Y, -Z], -1).reshape(-1, 3)

        # R = make_rotate(0, 0, math.radians(theta))
        # depth 2 world coordinates!

        # considerate it into points in the image coordiantes and connected into a mesh!
        space_points_ = (camera_pose[:3, :3] @ cameras_points.transpose()
                         ).transpose() + camera_pose[:3, 3]
        # space_points_ = (cameras_points @ camera_pose[:3, :3].transpose()
        #                  ) - camera_pose[3, :3]
        space_points = (
            space_points_.reshape(self.resolution, self.resolution, 3) *
            mask_3)
        # idx = mask.nonzero()
        # space_points_3 = space_points[idx[0], idx[1], :]
        return depth, space_points, mask

    # generate the original mesh from the depth and mesh!
    # use the
    # store the points information which is visible!
    def generate_gt_p_idx(self, mesh, depth_map, camera_pose):
        vertices = mesh.points()
        # idx = []
        camera_point = ((
            (vertices - camera_pose[:3, 3])) @ camera_pose[:3, :3])
        camera_point[:, 2] = -camera_point[:, 2]
        x_p = camera_point[:, 0] * self.znear / (camera_point[:, 2])
        y_p = camera_point[:, 1] * self.znear / (camera_point[:, 2])
        # print(x_p.max(), y_p.max())

        X = ((self.resolution_half -
              2 * self.resolution_half * y_p / self.w).reshape(-1, 1)).astype(
                  np.int32)
        Y = ((self.resolution_half +
              2 * self.resolution_half * x_p / self.w).reshape(-1, 1)).astype(
                  np.int32)
        # idx = np.concatenate([X, Y], -1).astype(np.int32)
        # print(idx.shape)
        # print(depth_map.shape)
        # print(depth_map[X, Y].shape)
        depth_map_idx = depth_map[X, Y]
        # print("this is the ")
        # print(camera_point[:, 2].min())

        # depth_map_nonzero_x, depth_map_nonzero_y = depth_map.nonzero(
        # )[0], depth_map.nonzero()[1]
        # print(depth_map[depth_map_nonzero_x, depth_map_nonzero_y].min())
        # print(depth_map_idx.min())
        idx_True = camera_point[:, 2] < depth_map_idx.reshape(-1) + 0.8

        # generate the delete mesh!
        # idx_nonzero = idx_True.nonzero()[0]

        # print("this is the nonzero of the idx{:0d}".format(
        #     idx_nonzero.shape[0]))
        # V_new = vertices[idx_nonzero, :]
        idx_nonzero = (idx_True - 1).nonzero()[0]

        # ori_v_2_new_v = np.ones(vertices.shape[0]) * -1
        # idx_new = np.linspace(0, idx_nonzero.shape[0], idx_nonzero.shape[0])
        # ori_v_2_new_v[idx_nonzero] = idx_new

        # faces = mesh.faces
        # f1 = ori_v_2_new_v[faces[:, 0]]
        # f2 = ori_v_2_new_v[faces[:, 1]]
        # f3 = ori_v_2_new_v[faces[:, 2]]

        # f = ((f1 - f2) * (f1 - f3) * (f2 - f3)).nonzero()[0]

        # f_save = faces[f]
        # face_new = np.concatenate([
        #     ori_v_2_new_v[f_save[:, 0]].reshape(
        #         -1, 1), ori_v_2_new_v[f_save[:, 1]].reshape(-1, 1),
        #     ori_v_2_new_v[f_save[:, 2]].reshape(-1, 1)
        # ], -1)

        # return V_new, face_new.reshape(-1, 3).astype(np.int32)

        # Use the openmesh to delete the mesh!
        for idx in idx_nonzero:
            mesh.delete_vertex(mesh.vertex_handle(idx))
        mesh.garbage_collection()
        return mesh.points(), mesh

    # considerate more case!
    def render_data(self,
                    mesh_lists,
                    mesh_openmesh_list,
                    camera_poses,
                    target_paths,
                    write_points=True,
                    target_points_paths=None,
                    partial_mesh_paths=None):
        for i, (mesh, mesh_openmesh, camera_pose, target_path) in enumerate(
                zip(mesh_lists, mesh_openmesh_list, camera_poses,
                    target_paths)):
            # print(i)
            depth_map, depth_points_image, mask = self.render_depth(
                mesh, camera_pose)
            vertices, mesh_om = self.generate_gt_p_idx(mesh_openmesh,
                                                       depth_map, camera_pose)
            if target_path is not None:
                if write_points:
                    F = np.array([0, 0, 0]).astype(np.int32).reshape(-1, 3)
                    igl.write_triangle_mesh(target_points_paths[i],
                                            vertices.reshape(-1, 3), F)
            if partial_mesh_paths is not None:
                # igl.write_triangle_mesh(partial_mesh_paths[i], vertices, faces)
                om.write_mesh(partial_mesh_paths[i], mesh_om)
                mesh = generate_depth_mesh(depth_points_image, mask)
                om.write_mesh(
                    partial_mesh_paths[i].replace('visible', 'depth_map', 1),
                    mesh)
        return None


def Scale_data(vertices_source, center_src, scale_src):
    vertices_source = (vertices_source - center_src) / scale_src * 5
    return vertices_source


def Rotation2anxis(R):
    theta = np.arccos((np.trace((R)) - 1.0) / 2.0)
    anxis = 1 / 2 / np.sin(theta) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    anxis = theta * anxis.copy() / np.linalg.norm(anxis, 2)
    return anxis


def anxis2Rotation(anxis_angle):
    I = np.identity(3)
    theta = np.linalg.norm(anxis_angle)
    anxis_angle = anxis_angle / theta
    M = np.zeros([3, 3])
    M[0, 1] = -anxis_angle[2]
    M[0, 2] = anxis_angle[1]
    M[1, 0] = anxis_angle[2]
    M[1, 2] = -anxis_angle[0]
    M[2, 0] = -anxis_angle[1]
    M[2, 1] = anxis_angle[0]
    return I + np.sin(theta) * M + (1 - np.cos(theta)) * M @ M


# prepare for the input!
def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(
        B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B, ), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid)**2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(
        1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius**2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    torch.cuda.empty_cache()
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, C]
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points],
                               dim=-1)  # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def Sample_points(points, npoints):
    idx = farthest_point_sample(points, npoints)
    points = points.reshape(points.shape[1], 3)
    idx = idx.reshape(-1)
    new_points = torch.index_select(points, 0, idx)
    return new_points


def Sample_points_normals(points, normals, npoints):
    idx = farthest_point_sample(points, npoints)
    points = points.reshape(points.shape[1], 3)
    idx = idx.reshape(-1)
    new_points = torch.index_select(points, 0, idx)
    new_normals = torch.index_select(normals, 0, idx)
    return new_points, new_normals


# src_paths: source mesh of the mesh_files
# tar_paths: sample_points of the target mesh_files.

import trimesh


# We need delete the data contained the nan!
def Make_sample_data(src_paths, tar_paths, sample_normal=True, num=1024):
    # for i in range(len(src_paths)):
    i = 0
    while i < len(src_paths):
        # V, F = igl.read_triangle_mesh(src_paths[i])
        # normals = igl.per_vertex_normals(V, F)
        mesh = trimesh.load_mesh(src_paths[i])
        V, F, normals = mesh.vertices, mesh.faces, mesh.vertex_normals
        normals.reshape(-1, normals.shape[0], 3).astype(np.float32)
        V = V.reshape(-1, V.shape[0], 3).astype(np.float32)
        tensor_vertices = torch.from_numpy(V)
        tensor_normals = torch.from_numpy(normals)
        new_points, new_normals = (Sample_points_normals(
            tensor_vertices, tensor_normals, num))
        new_points = new_points.cpu().numpy()
        new_normals = new_normals.cpu().numpy()
        if np.sum(np.isnan(new_normals)) > 0:
            continue
        F0 = np.zeros([3]).reshape(1, 3).astype(np.int32)
        igl.write_triangle_mesh(tar_paths[i], new_points, F0)
        igl.write_triangle_mesh(
            tar_paths[i].replace("sample", "sample_normals", 1), new_normals,
            F0)
        i = i + 1


import igl

# test()
#
fx = 5.40021232e+02
fy = 5.70342205e+02
cx = 3.20000000e+02
cy = 240.0
factor = 1
# height = 100
# width = 100

import cv2 as cv
import trimesh


# def Real_depth_2_depth_obj():
class Real_depth_2_depth_obj(object):
    def __init__(self, height, width, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.factor = factor
        self.height = height
        self.width = width
        self.x = np.concatenate([
            np.linspace(0, self.width, self.width).reshape(1, -1)
            for i in range(self.height)
        ], 0).reshape(self.height, self.width, 1)
        self.y = np.concatenate([
            np.linspace(0, self.height, self.height).reshape(-1, 1)
            for i in range(self.width)
        ], 1).reshape(self.height, self.width, 1)

        self.temp_mask = np.zeros_like(self.x)
        self.temp_mask[3:self.height - 3, 3:self.width - 3, 0] = 1.0
        # np.linspace()
    def depth2depthobj(self, depth):
        depthobj_z = depth.reshape(self.height, self.width, 1)
        depthobj_x = (self.x - self.cx) * depthobj_z / self.fx
        depthobj_y = (self.y - self.cy) * depthobj_z / self.fy
        mask = np.zeros_like(depthobj_z)
        # print(depthobj_z.max())
        # if depthobj_z.nonzero()[0].max() > 479 or depthobj_z.nonzero()[0].min() < 1:

        mask[(depthobj_z > 0).nonzero()] = 1.0
        mask = mask * self.temp_mask
        # print(mask.shape)
        depthobj_x1 = (depthobj_x * mask)
        depthobj_y1 = (depthobj_y * mask)
        depthobj_z1 = (depthobj_z * mask)
        depthobj = np.concatenate([depthobj_x1, depthobj_y1, depthobj_z1],
                                  2) / 1000
        return depthobj, mask

    def sample_normals(self, v, f, tar_paths, num=2048):
        # mesh = trimesh.load_mesh(src_paths[i])
        mesh = trimesh.Trimesh(vertices=v, faces=f)

        V, F, normals = mesh.vertices, mesh.faces, mesh.vertex_normals
        # random randomly chosse the idx!
        index = np.random.choice(np.arange(V.shape[0]), size=15000)
        V = V[index, :]
        normals = normals[index, :]
        normals.reshape(-1, normals.shape[0], 3).astype(np.float32)
        V = V.reshape(-1, V.shape[0], 3).astype(np.float32)
        tensor_vertices = torch.from_numpy(V)
        tensor_normals = torch.from_numpy(normals)
        new_points, new_normals = (Sample_points_normals(
            tensor_vertices, tensor_normals, num))
        new_points = new_points.cpu().numpy()
        new_normals = new_normals.cpu().numpy()
        # print(i)
        if np.sum(np.isnan(new_normals)) > 0:
            return
        F0 = np.zeros([3]).reshape(1, 3).astype(np.int32)
        igl.write_triangle_mesh(tar_paths, new_points, F0)
        igl.write_triangle_mesh(
            tar_paths.replace("sample", "sample_normals", 1), new_normals, F0)

    def generate_data(self, src_paths, tar_paths):
        poses = []
        for idx, (src_path, tar_path) in enumerate(zip(src_paths, tar_paths)):
            depth = cv.imread(src_path, cv.IMREAD_UNCHANGED)
            depthobj, mask = self.depth2depthobj(depth)
            depth_mesh = generate_depth_mesh(depthobj, mask)
            # igl.write_triangle_mesh(tar_path, )
            # om.write_mesh(tar_path, depth_mesh)
            depth_V = depth_mesh.points()
            depth_F = depth_mesh.face_vertex_indices()

            F = np.zeros([1, 3]).astype(np.int32)
            # igl.write_triangle_mesh(tar_path.replace(".obj", "points.obj", 1),
            #                         depthobj.reshape(-1, 3), F)
            pose = np.loadtxt(src_path.replace('depth.png', 'pose.txt', 1))
            ori_V = depthobj.reshape(
                -1, 3) @ pose[:3, :3].transpose() + pose[:3, 3]
            self.sample_normals(depth_V, depth_F,
                                tar_path.replace(".obj", "_sample.obj")),
            # igl.write_triangle_mesh(
            #     tar_path.replace(".obj", "points_gt.obj", 1), ori_V, F)
            poses.append(pose)
            print("this is the {:0d}sample".format(idx))
        return poses


# def Real_depth_2_depth_obj():
class Real_depth_2_depth_obj_original(object):
    def __init__(self, height, width, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.factor = factor
        self.height = height
        self.width = width
        self.x = np.concatenate([
            np.linspace(0, self.width, self.width).reshape(1, -1)
            for i in range(self.height)
        ], 0).reshape(self.height, self.width, 1)
        self.y = np.concatenate([
            np.linspace(0, self.height, self.height).reshape(-1, 1)
            for i in range(self.width)
        ], 1).reshape(self.height, self.width, 1)

        self.temp_mask = np.zeros_like(self.x)
        self.temp_mask[3:self.height - 3, 3:self.width - 3, 0] = 1.0
        # np.linspace()
    def depth2depthobj(self, depth):
        depthobj_z = depth.reshape(self.height, self.width, 1)
        depthobj_x = (self.x - self.cx) * depthobj_z / self.fx
        depthobj_y = (self.y - self.cy) * depthobj_z / self.fy
        mask = np.zeros_like(depthobj_z)
        # print(depthobj_z.max())
        # if depthobj_z.nonzero()[0].max() > 479 or depthobj_z.nonzero()[0].min() < 1:

        mask[(depthobj_z > 0).nonzero()] = 1.0
        mask = mask * self.temp_mask
        # print(mask.shape)
        depthobj_x1 = (depthobj_x * mask)
        depthobj_y1 = (depthobj_y * mask)
        depthobj_z1 = (depthobj_z * mask)
        depthobj = np.concatenate([depthobj_x1, depthobj_y1, depthobj_z1],
                                  2) / 1000
        return depthobj, mask

    def sample_normals(self, v, f, tar_paths):
        # mesh = trimesh.load_mesh(src_paths[i])
        mesh = trimesh.Trimesh(vertices=v, faces=f)

        V, F, normals = mesh.vertices, mesh.faces, mesh.vertex_normals
        # random randomly chosse the idx!
        index = np.random.choice(np.arange(V.shape[0]), size=20000)
        V = V[index, :]
        normals = normal[index, :]
        normals.reshape(-1, normals.shape[0], 3).astype(np.float32)
        V = V.reshape(-1, V.shape[0], 3).astype(np.float32)
        tensor_vertices = torch.from_numpy(V)
        tensor_normals = torch.from_numpy(normals)
        new_points, new_normals = (Sample_points_normals(
            tensor_vertices, tensor_normals, num))
        new_points = new_points.cpu().numpy()
        new_normals = new_normals.cpu().numpy()
        # print(i)
        if np.sum(np.isnan(new_normals)) > 0:
            return
        F0 = np.zeros([3]).reshape(1, 3).astype(np.int32)
        igl.write_triangle_mesh(tar_paths, new_points, F0)
        igl.write_triangle_mesh(
            tar_paths.replace("sample", "sample_normals", 1), new_normals, F0)
        i = i + 1

    def generate_data(self, src_paths, tar_paths):
        poses = []
        for idx, (src_path, tar_path) in enumerate(zip(src_paths, tar_paths)):
            depth = cv.imread(src_path, cv.IMREAD_UNCHANGED)
            depthobj, mask = self.depth2depthobj(depth)
            depth_mesh = generate_depth_mesh(depthobj, mask)
            # igl.write_triangle_mesh(tar_path, )
            om.write_mesh(tar_path, depth_mesh)

            F = np.zeros([1, 3]).astype(np.int32)
            # igl.write_triangle_mesh(tar_path.replace(".obj", "points.obj", 1),
            #                         depthobj.reshape(-1, 3), F)
            pose = np.loadtxt(src_path.replace('depth.png', 'pose.txt', 1))
            ori_V = depthobj.reshape(
                -1, 3) @ pose[:3, :3].transpose() + pose[:3, 3]
            self.sample_normals(ori_V),
            # igl.write_triangle_mesh(
            #     tar_path.replace(".obj", "points_gt.obj", 1), ori_V, F)
            poses.append(pose)
            print("this is the {:0d}sample".format(idx))
        return poses
