import openmesh as om
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import pyrender
import trimesh
import pyexr
# For offscreenrender we need set the enviroment!
import os, cv2, math, igl

os.environ['PYOPENGL_PLATFORM'] = 'egl'

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


def Scale_data(vertices_source, center_src, scale_src):
    vertices_source = (vertices_source - center_src) / scale_src * 5
    return vertices_source


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