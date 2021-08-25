# we test the svd module
# we use the two mesh, to considerate the double intersection loss!
import torch
import igl
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
from loss_2021_8_23 import cal_loss_intersection_batch_whole_median_pts_lines, cal_loss_intersection_batch_whole_median_new_svd
from loss_2021_8_23 import Random_uniform_distribution_lines_batch_efficient_resample
from loss_2021_8_23 import chamfer_dist, chamfer_dist_welsch
import sys
import argparse


# the data:{src_pts, tgt_pts, R, T}
# tgt_pts = src_pts@R + T
# We use almost fixed gradient steps for optimization
def test_one_case(data,
                  Save_path,
                  writer=None,
                  n_epoch=2000,
                  n_sample_line=20000,
                  device='cpu'):
    bounding_box = data['bounding_box']
    vertics1_tensor = data['vertics1_tensor']
    vertics2_tensor = data['vertics2_tensor']
    vertics1_faces_tensor = data['vertics1_faces_tensor']
    vertics2_faces_tensor = data['vertics2_faces_tensor']
    centers = data['centers']
    if os.path.exists(Save_path) is False:
        os.mkdir(Save_path)

    R = (bounding_box[0, :] - bounding_box[-1, :]).norm(p=2).to(device)
    rotation = torch.eye(3).to(device)
    translation = torch.zeros(3).to(device)
    # vertics1_tensor = vertics1_tensor
    for epoch in range(n_epoch):
        lines = Random_uniform_distribution_lines_batch_efficient_resample(
            torch.FloatTensor([R]).reshape(1, 1).to(device),
            centers.reshape(1, -1).to(device), n_sample_line,
            vertics1_tensor.view(1, -1, 3).to(device),
            vertics2_tensor.view(1, -1, 3).to(device),
            device).detach().view(-1, 6)

        # use svd to update the R and T
        Rot, Tra = cal_loss_intersection_batch_whole_median_new_svd(
            1, 1, 5, 5, vertics1_faces_tensor.reshape(1, -1, 9),
            vertics2_faces_tensor.reshape(1, -1, 9), lines.reshape(1, -1, 6),
            device)
        if Rot is None:
            continue
        vertics1_tensor = vertics1_tensor @ Rot.transpose(1, 0) + Tra
        vertics1_faces_tensor = vertics1_faces_tensor.reshape(
            -1, 3) @ Rot.transpose(1, 0) + Tra

        rotation = rotation @ Rot.transpose(1, 0)
        translation = translation @ Rot.transpose(1, 0) + Tra
        # save the results and compare with the chamfer loss!
        loss_cf = chamfer_dist(
            vertics1_tensor.reshape(-1, vertics1_tensor.shape[0], 3),
            vertics2_tensor.reshape(-1, vertics2_tensor.shape[0], 3))
        print("\033[34mthis is the chamfer loss:{:4f}\033[0m".format(
            loss_cf.detach().item()))
        if epoch % 100 == 0:
            F = np.zeros([1, 3], np.int32)
            faces1 = F
            faces2 = F
            igl.write_obj(os.path.join(Save_path,
                                       str(epoch) + ".obj"),
                          vertics1_tensor.detach().cpu().numpy(), faces1)
            igl.write_obj(os.path.join(Save_path, 'target' + ".obj"),
                          vertics2_tensor.detach().cpu().numpy(), faces2)
            # save the results
            transforms = np.ones([3, 4])
            transforms[:3, :3] = rotation.detach().cpu().numpy()
            transforms[:3, 3] = translation.detach().cpu().numpy()

            np.savetxt(os.path.join(Save_path,
                                    str(epoch) + '_transform.txt'), transforms)
        writer.add_scalar('./loss/chamfer_loss',
                          loss_cf.detach().cpu().item(), epoch)


def main(args):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    label1 = "27"
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    vertics1, faces1 = igl.read_triangle_mesh(
        os.path.join(args.data_path, "src_" + label1 + ".obj"))

    vertics2, faces2 = igl.read_triangle_mesh(
        os.path.join(args.data_path, "tar_" + label1 + ".obj"))
    # vertics1, faces1 = igl.read_triangle_mesh(
    #     os.path.join(
    #         "/disk_ssd/dengzhi/registration/model/bunny/reconstruction/registration_intersection/experients1/Rebuttal_cases/rebuttal/human/CD/src_101_24_transformed_0.obj"
    #     ))

    # vertics2, faces2 = igl.read_triangle_mesh(
    #     os.path.join(
    #         "/disk_ssd/dengzhi/registration/model/bunny/reconstruction/registration_intersection/experients1/Rebuttal_cases/rebuttal/human/CD/tar_101_24.obj"
    #     ))

    # use the face_neigh as the neigh pts.
    faces_neighs1 = np.concatenate([
        vertics1[faces1[:, 0], :], vertics1[faces1[:, 1], :],
        vertics1[faces1[:, 2], :]
    ], -1)
    faces_neighs2 = np.concatenate([
        vertics2[faces2[:, 0], :], vertics2[faces2[:, 1], :],
        vertics2[faces2[:, 2], :]
    ], -1)

    vertics1_tensor = torch.from_numpy(vertics1.astype(np.float32)).to(device)
    vertics2_tensor = torch.from_numpy(vertics2.astype(np.float32)).to(device)
    faces1_tensor = torch.from_numpy(faces_neighs1.astype(
        np.float32)).to(device).reshape(1, -1, 3)
    faces2_tensor = torch.from_numpy(faces_neighs2.astype(
        np.float32)).to(device)

    bounding_box = torch.from_numpy(
        igl.bounding_box(vertics2)[0].astype(np.float32)).to(device)
    centers = vertics2_tensor.mean(0).to(device)

    data = {}
    data.update({'bounding_box': bounding_box})
    data.update({'vertics1_tensor': vertics1_tensor})
    data.update({'vertics2_tensor': vertics2_tensor})
    data.update({'vertics1_faces_tensor': faces1_tensor})
    data.update({'vertics2_faces_tensor': faces2_tensor})
    data.update({'centers': centers})
    writer = SummaryWriter(log_dir=os.path.join(args.Save_path, 'log'))
    test_one_case(data, args.Save_path, writer=writer, device=device)


if __name__ == "__main__":
    print("Test our case!")
    label = str(46)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        default=
        "/disk_ssd/dengzhi/registration/model/bunny/reconstruction/registration_intersection/experients1/Rebuttal_cases/dragon_recon/datasets3"
    )
    parser.add_argument(
        '--Save_path',
        type=str,
        default=
        "/disk_ssd/dengzhi/registration/model/bunny/reconstruction/registration_intersection/experients1/2021_7_30/"
        + label)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    main(args)
