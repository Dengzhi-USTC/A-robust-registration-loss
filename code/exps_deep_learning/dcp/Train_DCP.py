# tide up the code;
# based on the original. we analysis this problem!
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import gc
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import sys

sys.path.append('./../')
sys.path.append('./../../')

from model import DCP
from utils import transform_point_cloud, npmat2euler, dict_all_to_device, makefacevertices, Dict2txt_json
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
import igl
from datetime import datetime
from pre_dataloader import generate_datasets_human

import utils as utils
from loss import cal_loss_intersection_batch_whole_median_pts_lines, Reconstruction_point, Random_uniform_distribution_lines_batch_efficient_resample, chamfer_dist, Sample_neighs
# Part of the code is referred from: https://github.com/floodsung/LearningToCompare_FSL


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' +
              'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' +
              'model.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' +
              'data.py.backup')


def test_one_epoch(args, net, test_loader, save_results=None, epoch=0):
    net.eval()
    mse_ab = 0
    mae_ab = 0
    mse_ba = 0
    mae_ba = 0
    total_loss = 0
    total_cycle_loss = 0
    num_examples = 0
    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []

    rotations_ba = []
    translations_ba = []
    rotations_ba_pred = []
    translations_ba_pred = []

    count_i = 0

    # for data in tqdm(test_loader):
    for k, data in enumerate(test_loader):
        dict_all_to_device(data, 'cuda')
        src = data['points_src_sample']
        target = data['points_tar_sample']
        rotation_ab = data['R']
        translation_ab = data['T']
        rotation_ba = data['R_inv']
        translation_ba = data['T_inv']

        batch_size = src.size(0)
        num_examples += batch_size
        rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = net(
            src, target)

        ## save rotation and translation
        rotations_ab.append(rotation_ab.detach().cpu().numpy())
        translations_ab.append(translation_ab.detach().cpu().numpy())
        rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
        translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
        # eulers_ab.append(euler_ab.numpy())
        ##
        rotations_ba.append(rotation_ba.detach().cpu().numpy())
        translations_ba.append(translation_ba.detach().cpu().numpy())
        rotations_ba_pred.append(rotation_ba_pred.detach().cpu().numpy())
        translations_ba_pred.append(translation_ba_pred.detach().cpu().numpy())
        # eulers_ba.append(euler_ba.numpy())

        transformed_src = transform_point_cloud(src, rotation_ab_pred,
                                                translation_ab_pred)
        gt_transformed_src = transform_point_cloud(src, rotation_ab,
                                                   translation_ab)

        transformed_target = transform_point_cloud(target, rotation_ba_pred,
                                                   translation_ba_pred)
        gt_transformed_target = transform_point_cloud(target, rotation_ba,
                                                      translation_ba)
        ###########################
        identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
        loss_gt = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
               + F.mse_loss(translation_ab_pred, translation_ab)
        loss = loss_gt
        loss_pp_wise, loss_chamfer = cal_test_loss(data, rotation_ab_pred,
                                                   translation_ab_pred)
        if args.cycle:
            rotation_loss = F.mse_loss(
                torch.matmul(rotation_ba_pred, rotation_ab_pred),
                identity.clone())
            translation_loss = torch.mean(
                (torch.matmul(rotation_ba_pred.transpose(2, 1),
                              translation_ab_pred.view(batch_size, 3, 1)).view(
                                  batch_size, 3) + translation_ba_pred)**2,
                dim=[0, 1])
            cycle_loss = rotation_loss + translation_loss

            loss = loss + cycle_loss * 0.1

        if args.cycle:
            print(
                "i{:0d}, loss_gt:{:4f}, loss_pp_wise{:4f}, loss_cycle{:4f}, loss_chamfer{:4f}"
                .format(k,
                        loss_gt.detach().item(),
                        loss_pp_wise.detach().cpu().item(),
                        cycle_loss.detach().cpu().item(),
                        loss_chamfer.detach().cpu().item()))
        else:
            print(
                "i{:0d}, loss_gt:{:4f}, loss_pp_wise{:4f}, loss_chamfer{:4f}".
                format(k,
                       loss_gt.detach().item(),
                       loss_pp_wise.detach().cpu().item(),
                       loss_chamfer.detach().cpu().item()))

        total_loss += loss.item() * batch_size

        if args.cycle:
            total_cycle_loss = total_cycle_loss + cycle_loss.item(
            ) * 0.1 * batch_size

        mse_ab += torch.mean((transformed_src - gt_transformed_src)**2,
                             dim=[0, 1, 2]).item() * batch_size
        mae_ab += torch.mean(torch.abs(transformed_src - gt_transformed_src),
                             dim=[0, 1, 2]).item() * batch_size

        mse_ba += torch.mean((transformed_target - gt_transformed_target)**2,
                             dim=[0, 1, 2]).item() * batch_size
        mae_ba += torch.mean(
            torch.abs(transformed_target - gt_transformed_target),
            dim=[0, 1, 2]).item() * batch_size

        # save results!
        if save_results is not None:
            if epoch % 50 != 0:
                continue
            paths_pred = []
            paths_gt = []
            paths_src = []
            paths_gt_src = []
            src_transform = transform_point_cloud(data['points_src_sample'],
                                                  rotation_ab_pred,
                                                  translation_ab_pred)
            #   considerate the gt_src_transformed
            gt_src_transformed_final_sample = transform_point_cloud(
                data['points_src_sample'], data['R'],
                data['T']).detach().transpose(2, 1).contiguous()
            V_src = data['points_src_sample'].cpu().transpose(2, 1).detach()
            V_pred = src_transform.cpu().transpose(2, 1).detach()
            V_gt = data['points_tar_sample'].cpu().transpose(2, 1).detach()
            gt_V_src = gt_src_transformed_final_sample.cpu().detach()
            for i in range(data['points_src_sample'].shape[0]):

                paths_pred.append(
                    os.path.join(
                        save_results,
                        str(epoch) + "_" + str(count_i) + "pred_src.obj"))
                paths_gt.append(
                    os.path.join(save_results,
                                 str(epoch) + "_" + str(count_i) + "gt.obj"))
                paths_src.append(
                    os.path.join(save_results,
                                 str(epoch) + "_" + str(count_i) + "src.obj"))
                paths_gt_src.append(
                    os.path.join(
                        save_results,
                        str(epoch) + "_" + str(count_i) + "src_gt.obj"))
                count_i += 1

            save_pred_gt_obj(V_src, V_pred, V_gt, gt_V_src, paths_src,
                             paths_pred, paths_gt, paths_gt_src)

    rotations_ab = np.concatenate(rotations_ab, axis=0)
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)

    rotations_ba = np.concatenate(rotations_ba, axis=0)
    translations_ba = np.concatenate(translations_ba, axis=0)
    rotations_ba_pred = np.concatenate(rotations_ba_pred, axis=0)
    translations_ba_pred = np.concatenate(translations_ba_pred, axis=0)

    # eulers_ab = np.concatenate(eulers_ab, axis=0)
    # eulers_ba = np.concatenate(eulers_ba, axis=0)

    return total_loss * 1.0 / num_examples, total_cycle_loss / num_examples, \
           mse_ab * 1.0 / num_examples, mae_ab * 1.0 / num_examples, \
           mse_ba * 1.0 / num_examples, mae_ba * 1.0 / num_examples, rotations_ab, \
           translations_ab, rotations_ab_pred, translations_ab_pred, rotations_ba, \
           translations_ba, rotations_ba_pred, translations_ba_pred


def cal_loss(data, rotation_ab_pred, translation_ab_pred, device):
    R = (torch.norm(
        data['tar_box'][:, 0, :] - data['tar_box'][:, -1, :], dim=-1, p=2) *
         0.5).reshape(-1, 1)
    lines = None

    points_ref = data['points_tar_sample'].transpose(2, 1).contiguous()

    tar_faces_tensor = data['points_based_neighs_tar'].transpose(2, 1).reshape(
        points_ref.shape[0], -1, 9)
    batch_size = points_ref.shape[0]
    pred_src_transformed_final_sample = transform_point_cloud(
        data['points_src_sample'], rotation_ab_pred,
        translation_ab_pred).transpose(2, 1).contiguous()

    loss_chamfer = chamfer_dist(
        pred_src_transformed_final_sample,
        data['points_tar_sample'].transpose(2, 1).contiguous())

    if lines is None:
        lines = Random_uniform_distribution_lines_batch_efficient_resample(
            R, (data['centers']), 15000, pred_src_transformed_final_sample,
            data['points_tar_sample'].transpose(2, 1).contiguous(), device)
    tp_loss_intersection = torch.FloatTensor([0]).cuda()

    pred_src_transformed_final = transform_point_cloud(
        data['points_src_sample'].contiguous(), rotation_ab_pred,
        translation_ab_pred).transpose(2, 1).contiguous()

    pred_src_faces_tensor = transform_point_cloud(
        data['points_based_neighs_src'].contiguous(), rotation_ab_pred,
        translation_ab_pred).transpose(2, 1).reshape(
            pred_src_transformed_final.shape[0], -1, 9)
    for j in range(pred_src_faces_tensor.shape[0]):
        tp_loss_intersection += cal_loss_intersection_batch_whole_median_pts_lines(
            1, 1, 5, 5, pred_src_faces_tensor[j:j + 1, :, :],
            tar_faces_tensor[j:j + 1, :, :], lines[j:j + 1, :, :],
            device) / 5.0

    loss_rotation = F.mse_loss(data['R'], rotation_ab_pred)
    loss_translation = F.mse_loss(data['T'], translation_ab_pred)
    gt_src_transformed_final_sample = transform_point_cloud(
        data['points_src_sample'], data['R'],
        data['T']).detach().transpose(2, 1).contiguous()
    loss_pp_wise = torch.sqrt(
        torch.mean((pred_src_transformed_final_sample -
                    gt_src_transformed_final_sample)**2))

    loss_pp_wise_ori = torch.mean((data['points_src_sample'].transpose(2, 1) -
                                   gt_src_transformed_final_sample)**2)
    loss_pp_wise_mae = torch.mean(
        torch.abs(pred_src_transformed_final_sample -
                  gt_src_transformed_final_sample))

    loss_pp_wise_identity = torch.mean(
        torch.abs(pred_src_transformed_final_sample -
                  data['points_src_sample'].transpose(2, 1).contiguous()))
    np_pred_rotation = rotation_ab_pred.detach().cpu().numpy()
    np_pred_euler = npmat2euler(np_pred_rotation, 'xyz')
    np_gt_rotation = data['R'].detach().cpu().numpy()
    np_gt_euler = npmat2euler(np_gt_rotation, 'xyz')
    loss_rotation_euler_mae = np.mean(np.abs(np_pred_euler - np_gt_euler))
    loss_rotation_euler_rmse = np.sqrt(
        np.mean((np_pred_euler - np_gt_euler)**2))
    return tp_loss_intersection / batch_size, loss_pp_wise.detach(
    ), loss_chamfer, loss_pp_wise_mae, loss_pp_wise_ori, loss_rotation.detach(
    ), loss_translation.detach(
    ), loss_rotation_euler_mae, loss_rotation_euler_rmse, loss_pp_wise_identity
    # return 0


# we can collect the measure of the results!


def cal_test_loss(data, rotation_ab_pred, translation_ab_pred):
    pred_src_transformed_final_sample = transform_point_cloud(
        data['points_src_sample'], rotation_ab_pred,
        translation_ab_pred).transpose(2, 1).contiguous()
    gt_src_transformed_final_sample = transform_point_cloud(
        data['points_src_sample'], data['R'],
        data['T']).detach().transpose(2, 1).contiguous()

    loss_pp_wise = torch.mean(
        torch.abs(pred_src_transformed_final_sample -
                  gt_src_transformed_final_sample))
    loss_chamfer = chamfer_dist(
        pred_src_transformed_final_sample,
        data['points_tar_sample'].transpose(2, 1).contiguous())
    pred_src_transformed_final_sample = pred_src_transformed_final_sample.detach(
    )
    return loss_pp_wise, loss_chamfer


def train_one_epoch(args, net, train_loader, opt, epoch=0, writer=None):
    net.train()

    mse_ab = 0
    mae_ab = 0
    mse_ba = 0
    mae_ba = 0

    total_loss = 0
    total_loss_pp_wise = 0
    total_loss_intersection = 0
    total_loss_gt = 0
    total_cycle_loss = 0
    total_loss_chamfer = 0
    total_loss_rotation = 0
    total_loss_translation = 0
    total_loss_rot_mae = 0
    total_loss_rot_rmse = 0
    num_examples = 0
    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []

    rotations_ba = []
    translations_ba = []
    rotations_ba_pred = []
    translations_ba_pred = []

    batch = 0
    for data in train_loader:
        dict_all_to_device(data, 'cuda')
        src = data['points_src_sample']

        target = data['points_tar_sample']
        rotation_ab = data['R']
        translation_ab = data['T']
        rotation_ba = data['R_inv']
        translation_ba = data['T_inv']

        batch_size = src.size(0)
        opt.zero_grad()
        num_examples += batch_size
        rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = net(
            src, target)

        ## save rotation and translation
        rotations_ab.append(rotation_ab.detach().cpu().numpy())
        translations_ab.append(translation_ab.detach().cpu().numpy())
        rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
        translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
        ##
        rotations_ba.append(rotation_ba.detach().cpu().numpy())
        translations_ba.append(translation_ba.detach().cpu().numpy())
        rotations_ba_pred.append(rotation_ba_pred.detach().cpu().numpy())
        translations_ba_pred.append(translation_ba_pred.detach().cpu().numpy())

        transformed_src = transform_point_cloud(src, rotation_ab_pred,
                                                translation_ab_pred)
        gt_transformed_src = transform_point_cloud(src, rotation_ab,
                                                   translation_ab)

        transformed_target = transform_point_cloud(target, rotation_ba_pred,
                                                   translation_ba_pred)
        gt_transformed_target = transform_point_cloud(target, rotation_ba,
                                                      translation_ba)
        ###########################
        identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
        # return
        # We also use the L1 generate better results than L2!
        loss_gt = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
               + F.mse_loss(translation_ab_pred, translation_ab)

        # We need change it into intersection loss!
        loss_intersection, loss_pp_wise, loss_chamfer, loss_pp_wise_mae, loss_pp_wise_ori, loss_rotation, loss_translation, loss_rot_mae, loss_rot_rmse, loss_pp_wise_identity = cal_loss(
            data, rotation_ab_pred, translation_ab_pred, 'cuda')
        if args.cycle:
            rotation_loss = F.mse_loss(
                torch.matmul(rotation_ba_pred, rotation_ab_pred),
                identity.clone())
            translation_loss = torch.mean(
                (torch.matmul(rotation_ba_pred.transpose(2, 1),
                              translation_ab_pred.view(batch_size, 3, 1)).view(
                                  batch_size, 3) + translation_ba_pred)**2,
                dim=[0, 1])
            cycle_loss = rotation_loss + translation_loss

            loss = loss_intersection + cycle_loss * 0.1
            print(
                "batch{:0d}/epoch{:0d}, l_gt:{:4f}, l_circle:{:4f}. l_sec{:4f}, l_pp_w{:4f}, l_pp_w_o{:4f}, l_rot{:4f}, l_tra{:4f}"
                .format(batch, epoch,
                        loss_gt.detach().item(),
                        cycle_loss.detach().item(),
                        loss_intersection.detach().item(),
                        loss_pp_wise_mae.detach().cpu().item(),
                        loss_pp_wise_ori.detach().ipu().item(),
                        loss_rotation.detach().cpu().item(),
                        loss_translation.detach().cpu().item()))
        else:
            loss = loss_intersection
            print(
                "batch{:0d}/epoch{:0d}, l_gt:{:4f},l_sec{:4f},  l_pp_w{:4f},l_pp_w_o{:4f},l_rot{:4f}, l_trans{:4f}"
                .format(batch, epoch,
                        loss_gt.detach().item(),
                        loss_intersection.detach().item(),
                        loss_pp_wise_mae.detach().cpu().item(),
                        loss_pp_wise_ori.detach().cpu().item(),
                        loss_rotation.detach().cpu().item(),
                        loss_translation.detach().cpu().item()))
        batch += 1
        loss.backward()
        opt.step()
        total_loss += loss.item() * batch_size
        total_loss_intersection += loss_intersection.detach().cpu().item(
        ) * batch_size
        total_loss_pp_wise += loss_pp_wise_mae.detach().cpu().item(
        ) * batch_size
        total_loss_gt += loss_gt.detach().cpu().item() * batch_size
        total_loss_chamfer += loss_chamfer.detach().cpu().item() * batch_size
        total_loss_rotation += loss_rotation.detach().cpu().item() * batch_size
        total_loss_translation += loss_translation.detach().cpu().item(
        ) * batch_size
        total_loss_rot_mae += loss_rot_mae * batch_size
        total_loss_rot_rmse += loss_rot_rmse * batch_size
        if args.cycle:
            total_cycle_loss = total_cycle_loss + cycle_loss.item(
            ) * 0.1 * batch_size

        mse_ab += torch.mean((transformed_src - gt_transformed_src)**2,
                             dim=[0, 1, 2]).item() * batch_size
        mae_ab += torch.mean(torch.abs(transformed_src - gt_transformed_src),
                             dim=[0, 1, 2]).item() * batch_size

        mse_ba += torch.mean((transformed_target - gt_transformed_target)**2,
                             dim=[0, 1, 2]).item() * batch_size
        mae_ba += torch.mean(
            torch.abs(transformed_target - gt_transformed_target),
            dim=[0, 1, 2]).item() * batch_size

    ave_loss_pp_wise = total_loss_pp_wise / num_examples
    ave_loss_cycle = total_cycle_loss / num_examples
    ave_loss_gt = total_loss_gt / num_examples
    ave_loss_intersection = total_loss_intersection / num_examples
    ave_loss_chamfer = total_loss_chamfer / num_examples
    ave_loss_rotation = total_loss_rotation / num_examples
    ave_loss_translation = total_loss_translation / num_examples
    ave_loss_rot_mae = total_loss_rot_mae / num_examples
    ave_loss_rot_rmse = total_loss_rot_rmse / num_examples
    print(
        "\033[36m,ep{:0d}, l_gt:{:4f}, l_cy{:4f}, l_pp_w{:4f}, l_sec{:4f}, l_cf{:4f}, l_rot{:4f}, l_rot_e{:4f} ,l_tra{:4f}!\033[0m"
        .format(epoch, ave_loss_gt, ave_loss_cycle, ave_loss_pp_wise,
                ave_loss_intersection, ave_loss_chamfer, ave_loss_rotation,
                ave_loss_rot_mae, ave_loss_translation))
    if writer is not None:
        writer.add_scalar('./loss/loss_gt', ave_loss_gt, epoch)
        writer.add_scalar('./loss/loss_cycle', ave_loss_cycle, epoch)
        writer.add_scalar('./loss/loss_pp_wise', ave_loss_pp_wise, epoch)
        writer.add_scalar('./loss/loss_intersection', ave_loss_intersection,
                          epoch)
        writer.add_scalar('./loss/loss_rotation', ave_loss_rotation, epoch)
        writer.add_scalar('./loss/loss_translation', ave_loss_translation,
                          epoch)
        writer.add_scalar('./loss/loss_chamfer', ave_loss_chamfer, epoch)
        writer.add_scalar('./loss/loss_rot_euler_mae', ave_loss_rot_mae, epoch)
        writer.add_scalar('./loss/loss_rot_euler_rmse', ave_loss_rot_rmse,
                          epoch)
        writer.add_scalar('./lr', opt.param_groups[0]['lr'], epoch)
    rotations_ab = np.concatenate(rotations_ab, axis=0)
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)

    rotations_ba = np.concatenate(rotations_ba, axis=0)
    translations_ba = np.concatenate(translations_ba, axis=0)
    rotations_ba_pred = np.concatenate(rotations_ba_pred, axis=0)
    translations_ba_pred = np.concatenate(translations_ba_pred, axis=0)

    return total_loss * 1.0 / num_examples, total_cycle_loss / num_examples, \
           mse_ab * 1.0 / num_examples, mae_ab * 1.0 / num_examples, \
           mse_ba * 1.0 / num_examples, mae_ba * 1.0 / num_examples, rotations_ab, \
           translations_ab, rotations_ab_pred, translations_ab_pred, rotations_ba, \
           translations_ba, rotations_ba_pred, translations_ba_pred


def save_pred_gt_obj(V_src, V_pred, V_gt, V_gt_src, path_src, path_pred,
                     path_gt, path_gt_src):
    F = np.zeros(3).reshape(1, 3).astype(np.int32)
    for i in range(V_pred.shape[0]):
        igl.write_triangle_mesh(path_src[i], V_src[i].numpy(), F)
        igl.write_triangle_mesh(path_pred[i], V_pred[i].numpy(), F)
        igl.write_triangle_mesh(path_gt[i], V_gt[i].numpy(), F)
        igl.write_triangle_mesh(path_gt_src[i], V_gt_src[i].numpy(), F)


def test(args, net, test_loader, save_results=None):

    test_loss, test_cycle_loss, \
    test_mse_ab, test_mae_ab, test_mse_ba, test_mae_ba, test_rotations_ab, test_translations_ab, \
    test_rotations_ab_pred, \
    test_translations_ab_pred, test_rotations_ba, test_translations_ba, test_rotations_ba_pred, \
    test_translations_ba_pred = test_one_epoch(args, net, test_loader, save_results = save_results)
    return
    # Igonre from the original dcp code;


#　use the intersection loss replace the surpervised loss！


def train(args,
          net,
          train_loader,
          test_loader,
          save_results=None,
          save_results_models=None,
          trainwriter=None):
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(net.parameters(),
                        lr=args.lr * 100,
                        momentum=args.momentum,
                        weight_decay=1e-4)
    else:
        # 5e-5
        # 1e-5
        # 2e-6
        # Airplane 1000, lr=1e-4!
        # 2e-6 ---> 1e-6
        opt = optim.Adam(net.parameters(), lr=1e-6, weight_decay=0)
    best_test_loss = np.inf
    best_test_cycle_loss = np.inf
    best_test_mse_ab = np.inf
    best_test_rmse_ab = np.inf
    best_test_mae_ab = np.inf

    best_test_r_mse_ab = np.inf
    best_test_r_rmse_ab = np.inf
    best_test_r_mae_ab = np.inf
    best_test_t_mse_ab = np.inf
    best_test_t_rmse_ab = np.inf
    best_test_t_mae_ab = np.inf

    best_test_mse_ba = np.inf
    best_test_rmse_ba = np.inf
    best_test_mae_ba = np.inf

    best_test_r_mse_ba = np.inf
    best_test_r_rmse_ba = np.inf
    best_test_r_mae_ba = np.inf
    best_test_t_mse_ba = np.inf
    best_test_t_rmse_ba = np.inf
    best_test_t_mae_ba = np.inf

    for epoch in range(args.epochs):
        # scheduler.step()
        train_loss, train_cycle_loss, \
        train_mse_ab, train_mae_ab, train_mse_ba, train_mae_ba, train_rotations_ab, train_translations_ab, \
        train_rotations_ab_pred, \
        train_translations_ab_pred, train_rotations_ba, train_translations_ba, train_rotations_ba_pred, \
        train_translations_ba_pred = train_one_epoch(args, net, train_loader, opt, epoch, writer=trainwriter)
        test_loss, test_cycle_loss, \
        test_mse_ab, test_mae_ab, test_mse_ba, test_mae_ba, test_rotations_ab, test_translations_ab, \
        test_rotations_ab_pred, \
        test_translations_ab_pred, test_rotations_ba, test_translations_ba, test_rotations_ba_pred, \
        test_translations_ba_pred = test_one_epoch(args, net, test_loader, save_results=save_results, epoch = epoch)
        train_rmse_ab = np.sqrt(train_mse_ab)
        test_rmse_ab = np.sqrt(test_mse_ab)

        train_rmse_ba = np.sqrt(train_mse_ba)
        test_rmse_ba = np.sqrt(test_mse_ba)

        train_rotations_ab_pred_euler = npmat2euler(train_rotations_ab_pred,
                                                    'xyz')
        train_eulers_ab = npmat2euler(train_rotations_ab, 'xyz')
        train_r_mse_ab = np.mean(
            (train_rotations_ab_pred_euler - train_eulers_ab)**2)
        train_r_rmse_ab = np.sqrt(train_r_mse_ab)
        train_r_mae_ab = np.mean(
            np.abs(train_rotations_ab_pred_euler - train_eulers_ab))
        train_t_mse_ab = np.mean(
            (train_translations_ab - train_translations_ab_pred)**2)
        train_t_rmse_ab = np.sqrt(train_t_mse_ab)
        train_t_mae_ab = np.mean(
            np.abs(train_translations_ab - train_translations_ab_pred))

        train_rotations_ba_pred_euler = npmat2euler(train_rotations_ba_pred,
                                                    'xyz')
        train_eulers_ba = npmat2euler(train_rotations_ba, 'xyz')

        train_r_mse_ba = np.mean(
            (train_rotations_ba_pred_euler - train_eulers_ba)**2)
        train_r_rmse_ba = np.sqrt(train_r_mse_ba)
        train_r_mae_ba = np.mean(
            np.abs(train_rotations_ba_pred_euler - train_eulers_ba))
        train_t_mse_ba = np.mean(
            (train_translations_ba - train_translations_ba_pred)**2)
        train_t_rmse_ba = np.sqrt(train_t_mse_ba)
        train_t_mae_ba = np.mean(
            np.abs(train_translations_ba - train_translations_ba_pred))

        test_rotations_ab_pred_euler = npmat2euler(test_rotations_ab_pred,
                                                   'xyz')
        test_eulers_ab = npmat2euler(test_rotations_ab, 'xyz')

        test_r_mse_ab = np.mean(
            (test_rotations_ab_pred_euler - test_eulers_ab)**2)
        test_r_rmse_ab = np.sqrt(test_r_mse_ab)
        test_r_mae_ab = np.mean(
            np.abs(test_rotations_ab_pred_euler - test_eulers_ab))
        test_t_mse_ab = np.mean(
            (test_translations_ab - test_translations_ab_pred)**2)
        test_t_rmse_ab = np.sqrt(test_t_mse_ab)
        test_t_mae_ab = np.mean(
            np.abs(test_translations_ab - test_translations_ab_pred))

        test_rotations_ba_pred_euler = npmat2euler(test_rotations_ba_pred,
                                                   'xyz')
        test_eulers_ba = npmat2euler(test_rotations_ba, 'xyz')

        test_r_mse_ba = np.mean(
            (test_rotations_ba_pred_euler - test_eulers_ba)**2)
        test_r_rmse_ba = np.sqrt(test_r_mse_ba)
        test_r_mae_ba = np.mean(
            np.abs(test_rotations_ba_pred_euler - test_eulers_ba))
        test_t_mse_ba = np.mean(
            (test_translations_ba - test_translations_ba_pred)**2)
        test_t_rmse_ba = np.sqrt(test_t_mse_ba)
        test_t_mae_ba = np.mean(
            np.abs(test_translations_ba - test_translations_ba_pred))

        if best_test_loss >= test_loss:
            best_test_loss = test_loss
            best_test_cycle_loss = test_cycle_loss

            best_test_mse_ab = test_mse_ab
            best_test_rmse_ab = test_rmse_ab
            best_test_mae_ab = test_mae_ab

            best_test_r_mse_ab = test_r_mse_ab
            best_test_r_rmse_ab = test_r_rmse_ab
            best_test_r_mae_ab = test_r_mae_ab

            best_test_t_mse_ab = test_t_mse_ab
            best_test_t_rmse_ab = test_t_rmse_ab
            best_test_t_mae_ab = test_t_mae_ab

            best_test_mse_ba = test_mse_ba
            best_test_rmse_ba = test_rmse_ba
            best_test_mae_ba = test_mae_ba

            best_test_r_mse_ba = test_r_mse_ba
            best_test_r_rmse_ba = test_r_rmse_ba
            best_test_r_mae_ba = test_r_mae_ba

            best_test_t_mse_ba = test_t_mse_ba
            best_test_t_rmse_ba = test_t_rmse_ba
            best_test_t_mae_ba = test_t_mae_ba

            if torch.cuda.device_count() > 1:
                torch.save(
                    net.module.state_dict(),
                    'checkpoints/%s/models/model.best.t7' % args.exp_name)
                torch.save(net.module.state_dict(),
                           os.path.join(save_results_models, 'best.pkl'))
            else:
                torch.save(
                    net.state_dict(),
                    'checkpoints/%s/models/model.best.t7' % args.exp_name)
                torch.save(net.state_dict(),
                           os.path.join(save_results_models, "best.pkl"))
        if epoch % args.freq_save_results == 0:
            if torch.cuda.device_count() > 1:
                torch.save(
                    net.module.state_dict(),
                    os.path.join(save_results_models,
                                 str(epoch) + '.pkl'))
            else:
                torch.save(
                    net.state_dict(),
                    os.path.join(save_results_models,
                                 str(epoch) + ".pkl"))


def main():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name',
                        type=str,
                        default='exp',
                        metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model',
                        type=str,
                        default='dcp',
                        metavar='N',
                        choices=['dcp'],
                        help='Model to use, [dcp]')
    parser.add_argument('--emb_nn',
                        type=str,
                        default='pointnet',
                        metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Embedding nn to use, [pointnet, dgcnn]')
    parser.add_argument(
        '--pointer',
        type=str,
        default='transformer',
        metavar='N',
        choices=['identity', 'transformer'],
        help='Attention-based pointer generator to use, [identity, transformer]'
    )
    parser.add_argument('--head',
                        type=str,
                        default='svd',
                        metavar='N',
                        choices=[
                            'mlp',
                            'svd',
                        ],
                        help='Head to use, [mlp, svd]')
    parser.add_argument('--emb_dims',
                        type=int,
                        default=512,
                        metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--n_blocks',
                        type=int,
                        default=1,
                        metavar='N',
                        help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads',
                        type=int,
                        default=4,
                        metavar='N',
                        help='Num of heads in multiheadedattention')
    parser.add_argument('--ff_dims',
                        type=int,
                        default=1024,
                        metavar='N',
                        help='Num of dimensions of fc in transformer')
    parser.add_argument('--dropout',
                        type=float,
                        default=0.0,
                        metavar='N',
                        help='Dropout ratio in transformer')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=10,
                        metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs',
                        type=int,
                        default=2500,
                        metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd',
                        action='store_true',
                        default=False,
                        help='Use SGD')
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        metavar='LR',
        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda',
                        action='store_true',
                        default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed',
                        type=int,
                        default=1234,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval',
                        action='store_true',
                        default=False,
                        help='evaluate the model')
    parser.add_argument('--cycle',
                        type=bool,
                        default=False,
                        metavar='N',
                        help='Whether to use cycle consistency')
    parser.add_argument('--gaussian_noise',
                        type=bool,
                        default=False,
                        metavar='N',
                        help='Wheter to add gaussian noise')
    parser.add_argument('--unseen',
                        type=bool,
                        default=False,
                        metavar='N',
                        help='Wheter to test on unseen category')
    parser.add_argument('--num_points',
                        type=int,
                        default=1024,
                        metavar='N',
                        help='Num of points to use')
    parser.add_argument('--dataset',
                        type=str,
                        default='modelnet40',
                        choices=['modelnet40'],
                        metavar='N',
                        help='dataset to use')
    parser.add_argument('--factor',
                        type=float,
                        default=4,
                        metavar='N',
                        help='Divided factor for rotations')
    parser.add_argument('--model_path',
                        type=str,
                        default='',
                        metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--freq_save_results',
                        type=int,
                        default=50,
                        metavar='N',
                        help='frequency of save mdoels')

    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # construct your own datasets, choose your own datasets.
    train_loader, test_loader = generate_datasets_human(True)

    if args.model == 'dcp':
        net = DCP(args).cuda()
        if args.eval:
            if args.model_path is '':
                model_path = 'checkpoints' + '/' + args.exp_name + '/models/model.best.t7'
            else:
                model_path = args.model_path
                print(model_path)
            if not os.path.exists(model_path):
                print("can't find pretrained model")
                return
            net.load_state_dict(torch.load(model_path), strict=False)
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
            print("Let's use", torch.cuda.device_count(), "GPUs!")
    else:
        raise Exception('Not implemented')
    if args.eval:
        if not os.path.exists('results'):
            os.makedirs('results')
        save_results = os.path.join('./results', args.exp_name)
        if not os.path.exists(save_results):
            os.makedirs(save_results)
        test(args, net, test_loader, save_results=save_results)
    else:

        # frame
        net.load_state_dict(torch.load(
            '/data1/dengzhi/dcp-master/M40/2021_03_05_00_31_02/models/200.pkl'
        ),
                            strict=False)
        # generate the exp_name?
        datasets = "Human"

        if not os.path.exists(datasets):
            os.makedirs(datasets)
        save_results = os.path.join(datasets, args.exp_name)
        if not os.path.exists(save_results):
            os.makedirs(save_results)
        timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        save_results = os.path.join(datasets, timestamp)

        save_results_model = os.path.join(save_results, 'models')

        if not os.path.exists(save_results):
            os.makedirs(save_results)
        if not os.path.exists(save_results_model):
            os.makedirs(save_results_model)
        current_log_path = os.path.join(save_results, "log")
        if not os.path.exists(current_log_path):
            os.makedirs(current_log_path)
        train_writer = SummaryWriter(current_log_path)
        train(args,
              net,
              train_loader,
              test_loader,
              save_results=save_results,
              save_results_models=save_results_model,
              trainwriter=train_writer)

    print('FINISH')


if __name__ == '__main__':
    main()
