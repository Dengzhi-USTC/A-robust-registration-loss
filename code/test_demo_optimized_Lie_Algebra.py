# we use the two mesh, to considerate the double intersection loss!
import torch
import igl
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
from loss import cal_loss_intersection_batch_whole_median_pts_lines
from loss import Reconstruction_point
from loss import Random_uniform_distribution_lines_batch_efficient_resample
from loss import chamfer_dist, Sample_neighs
import sys
import argparse


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
    # if epoch % 500 == 0:
    if epoch % 1000 == 0:
        lr *= (1 * (0.5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# the data:{src_pts, tgt_pts, R, T}
# tgt_pts = src_pts@R + T
# We use almost fixed gradient steps for optimization
def test_one_case(data,
                  Save_path,
                  writer=None,
                  n_epoch=1000,
                  n_sample_line=20000,
                  device='cpu'):
    bounding_box = data['bounding_box']
    vertics1_tensor_input = data['vertics1_tensor']
    vertics2_tensor = data['vertics2_tensor']
    vertics1_faces_tensor_input = data['vertics1_faces_tensor']
    vertics2_faces_tensor = data['vertics2_faces_tensor']
    centers = data['centers']
    if os.path.exists(Save_path) is False:
        os.mkdir(Save_path)
    Reconstruction = Reconstruction_point().to(device)
    optimize = torch.optim.Adam(Reconstruction.parameters(), lr=2e-2)

    # you can define your radius;
    R = (bounding_box[0, :] - bounding_box[-1, :]).norm(p=2).to(device)
    vertics1_tensor = vertics1_tensor_input
    for epoch in range(n_epoch):
        lines = Random_uniform_distribution_lines_batch_efficient_resample(
            torch.FloatTensor([R]).reshape(1, 1).to(device),
            centers.reshape(1, -1).to(device), n_sample_line,
            vertics1_tensor.view(1, -1, 3).to(device),
            vertics2_tensor.view(1, -1, 3).to(device),
            device).detach().view(-1, 6)
        adjust_learning_rate(optimizer=optimize,
                             epoch=epoch,
                             lr=optimize.param_groups[0]['lr'])
        vertics1_tensor, vertics1_faces_tensor = Reconstruction(
            vertics1_tensor_input, vertics1_faces_tensor_input)
        loss_di = cal_loss_intersection_batch_whole_median_pts_lines(
            1, 1, 5, 5, vertics1_faces_tensor.reshape(1, -1, 9),
            vertics2_faces_tensor.reshape(1, -1, 9), lines.reshape(1, -1, 6),
            device)
        if loss_di is not None:
            optimize.zero_grad()
            loss_di.backward()
            optimize.step()
            # save the results and compare with the chamfer loss!
            loss_cf = chamfer_dist(
                vertics1_tensor.reshape(-1, vertics1_tensor.shape[0], 3),
                vertics2_tensor.reshape(-1, vertics2_tensor.shape[0], 3))
            print(
                "\033[34mthis is the chamfer loss:{:4f}, loss_intersection{:4f}\033[0m"
                .format(loss_cf.detach().item(),
                        loss_di.detach().item()))
            if epoch % 10 == 0:
                F = np.zeros([1, 3], np.int32)
                faces1 = F
                faces2 = F
                igl.write_obj(os.path.join(Save_path,
                                           str(epoch) + ".obj"),
                              vertics1_tensor.detach().cpu().numpy(), faces1)
                igl.write_obj(os.path.join(Save_path, 'target' + ".obj"),
                              vertics2_tensor.detach().cpu().numpy(), faces2)
                torch.save(Reconstruction.state_dict(),
                           os.path.join(Save_path, 'model.pkl'))

                # save the results
                transform = Reconstruction.Transform()

                transforms = np.ones([3, 4])
                transforms[:3, :3] = transform[0].detach().cpu().numpy()
                transforms[:3, 3] = transform[1].detach().cpu().numpy()

                np.savetxt(
                    os.path.join(Save_path,
                                 str(epoch) + '_transform.txt'), transforms)
            writer.add_scalar('./loss/chamfer_loss',
                              loss_cf.detach().cpu().item(), epoch)
            writer.add_scalar('./loss/intersection_loss',
                              loss_di.detach().cpu().item(), epoch)


def main(args):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    label1 = args.label1
    device = args.device if torch.cuda.is_available() else 'cpu'
    source_path = os.path.join(args.data_path, label1 + "_src_sample" + ".obj")
    target_path = os.path.join(args.data_path, label1 + "_tar_sample" + ".obj")
    
    vertics1, faces1 = igl.read_triangle_mesh(source_path)
    vertics2, faces2 = igl.read_triangle_mesh(target_path)
    vertics1 = vertics1.astype(np.float32)
    vertics2 = vertics2.astype(np.float32)
    # use the face_neigh as the neigh pts.
    faces_neighs1 = Sample_neighs(vertics1, device=device)
    faces_neighs2 = Sample_neighs(vertics2, device=device)
    center1 = vertics1.mean(0)[np.newaxis, :]
    center2 = vertics2.mean(0)[np.newaxis, :]
    vertics1 = vertics1 - center1
    vertics2 = vertics2 - center2
    faces_neighs1 = faces_neighs1 - center1
    faces_neighs2 = faces_neighs2 - center2
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


# modify the data_path
if __name__ == "__main__":
    print("Test our case!")
    parser = argparse.ArgumentParser()
    path = "./sample_data/challenge_data"
    save_path = "./Results"
    parser.add_argument('--data_path', type=str, default=path)
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--label1', type=str, default=None)
    parser.add_argument('--Save_path', type=str, default=save_path)
    args = parser.parse_args()
    for label1 in ['0', '1', '2', '3', '4']:
        args.label1 = label1
        label = label1 + "challenge_new_1"
        args.Save_path = os.path.join(save_path, label)
        main(args)
