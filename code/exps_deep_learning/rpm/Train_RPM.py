# We refine the code from (RPM-Net)(https://github.com/tzodge/PCR-CMU/tree/main/RPMNet_Code)
from collections import defaultdict
import os
import random
from typing import Dict, List
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.utils.data
from torch.optim.lr_scheduler import MultiStepLR
from datetime import datetime
import igl

from arguments import rpmnet_train_arguments
from common.colors import BLUE, ORANGE
from common.math.se3 import transform
from common.misc import prepare_logger
from common.torch import dict_all_to_device, CheckPointManager, TorchDebugger
from common.math_torch import se3

from torch.utils.tensorboard import SummaryWriter
import sys

sys.path.append('./model')
sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')
from models.rpmnet import get_model
import utils as utils
from loss import cal_loss_intersection_batch_whole_median_pts_lines, Reconstruction_point, Random_uniform_distribution_lines_batch_efficient_resample, chamfer_dist, Sample_neighs
# from chamfer_distance import ChamferDistance
# from pre_dataloader_hull import Generate_dataloader, Generate_dataloader_test
from pre_dataloader import generate_datasets_human
# Set up arguments and logging
parser = rpmnet_train_arguments()
_args = parser.parse_args()
_logger, _log_path = prepare_logger(_args)


# contained the pretrained.
# contained the training.
# Save the results
class BaseRPMNet(object):
    def train(self):
        loss_intersection = 0
        loss_reg = 0
        loss_chamfer = 0
        loss_gt = 0
        for data in self.train_data_loader:
            self.optimize.zero_grad()
            dict_all_to_device(data, self.device)
            pred_transforms, endpoints = self.model(
                data,
                _args.num_train_reg_iter)  # Use less iter during training

            loss, pred_src_transformed_final = self.cal_loss(
                pred_transforms, endpoints, data)
            self.update(loss, pred_src_transformed_final, data)
            self.counter += 1
            loss_intersection += loss['loss_intersection'].detach().cpu().item(
            )
            loss_reg += loss['loss_reg'].detach().cpu().item()
            loss_chamfer += loss['loss_chamfer'].detach().cpu().item()
            loss_gt += loss['loss_gt'].detach().cpu().item()
        # red
        loss_intersection /= self.counter
        loss_reg /= self.counter
        loss_chamfer /= self.counter
        loss_gt /= self.counter
        print(
            "\033[32m epoch{:0f},cd loss:{:4f}, loss_int{:4f}, loss_reg{:4f}, loss_gt{:4f}\033[0m"
            .format(self.epochs, loss_chamfer, loss_intersection, loss_reg,
                    loss_gt))
        self.scheduler.step()
        self.train_writer.add_scalar('./loss/loss_intersection',
                                     loss_intersection, self.epochs)
        self.train_writer.add_scalar('./loss/loss_reg', loss_reg, self.epochs)
        self.train_writer.add_scalar('./loss/loss_chamfer', loss_chamfer,
                                     self.epochs)
        return

# We test our datasets!

    def Save_eval_results(self, idx, pred_transforms, data):
        num_iter = len(pred_transforms)
        for ni in range(num_iter - 1, num_iter):
            pred_src_transformed_final = se3.transform(
                pred_transforms[ni], data['points_src_sample'][..., :3])
            pred_src_sample_transformed_final = se3.transform(
                pred_transforms[ni], data['points_src_sample'][..., :3])
            src_tar_transform = torch.zeros(1, 4, 4).to(self.device)
            src_tar_transform[0, :3, :3] = data['R'][0].transpose(-1, -2)
            src_tar_transform[0, :3, 3] = data['T'][0]
            V_src_tar = se3.transform(
                src_tar_transform,
                data['points_src_sample'][..., :3]).reshape(-1,
                                                            3).cpu().numpy()
            for i in range(pred_src_transformed_final.shape[0]):
                V_src_pred = pred_src_transformed_final[i].detach().cpu(
                ).numpy().reshape(-1, 3)
                V_src_sample_pred = pred_src_sample_transformed_final[
                    i].detach().cpu().numpy().reshape(-1, 3)
                F_src = np.zeros([1, 3], np.int32)
                src_path = os.path.join(
                    self.current_val_results,
                    str(idx) + "_" + str(ni) + "_" + "_src_transformed_" +
                    str(i) + ".obj")
                igl.write_triangle_mesh(src_path, V_src_pred, F_src)
                src_sample_path = os.path.join(
                    self.current_val_results,
                    str(idx) + "_" + str(ni) + "_" + str(i) +
                    "_src_sample_transformed_" + ".obj")
                igl.write_triangle_mesh(src_sample_path, V_src_sample_pred,
                                        F_src * 0)

                V_tar = data['points_tar_sample'][i].detach().cpu().numpy()
                F_tar = np.zeros([1, 3], np.int32)

                tar_path = os.path.join(
                    self.current_val_results,
                    str(idx) + "_" + str(ni) + "_" + str(i) + "_tar_" + ".obj")
                igl.write_triangle_mesh(tar_path, V_tar, F_tar)

                V_src = data['points_src_sample'][i].detach().cpu().numpy()

                src_raw_path = os.path.join(
                    self.current_val_results,
                    str(idx) + "_" + str(ni) + "_" + str(i) + "_src_" + ".obj")
                igl.write_triangle_mesh(src_raw_path, V_src, F_src)

                src_gt_transformed_path = os.path.join(
                    self.current_val_results,
                    str(idx) + "_" + str(ni) + "_" + str(i) + "_src_gt_" +
                    ".obj")
                igl.write_triangle_mesh(src_gt_transformed_path, V_src_tar,
                                        F_src)

        return None

    def eval(self, loss_ignore_gt=False):
        if self.re_load_model_path is not None and self.model.eval() is True:
            self.re_load()
        loss_chamfer = 0
        loss_gt = 0
        for idx, data in enumerate(self.test_data_loader):
            dict_all_to_device(data, self.device)
            pred_transforms, endpoints = self.model(
                data, 5)  # Use less iter during training
            if loss_ignore_gt is False:
                loss = self.cal_gt_loss(pred_transforms, data)

                tp_loss_gt = loss['loss_gt'].detach().cpu().item()
                tp_loss_cd = loss['loss_chamfer'].detach().cpu().item()

                loss_chamfer += tp_loss_cd
                loss_gt += tp_loss_gt
                self.test_writer.add_scalar('./loss/loss_gt', tp_loss_gt, idx)
                self.test_writer.add_scalar('./loss/loss_chamfer', tp_loss_cd,
                                            idx)

            self.Save_eval_results(idx,
                                   pred_transforms=pred_transforms,
                                   data=data)
            transform = loss['pred_transform'].reshape(3, 4)
            transform1 = transform
            transform1[:3, :3] = transform[:3, :3].transpose(1, 0)

            transform1.detach().cpu().numpy().tofile(
                os.path.join(
                    self.current_val_results,
                    str(self.epochs) + "_pred_src_" + str(idx) + ".bin"))
        print(
            "\033[32m Validate,loss_gt{:4f}, loss_chamfer{:4f}\033[0m".format(
                loss_gt, loss_chamfer))
        # write a txt, for test_datasets.
        loss_dict = {'loss_chamfer': loss_chamfer, 'loss_gt': loss_gt}
        utils.Dict2txt_json(os.path.join(self.current_log_path_eval,
                                         'Val.json'),
                            loss_dict,
                            file_type='json')
        return

    def cal_loss(self,
                 pred_transforms,
                 endpoints,
                 data,
                 reduction: str = 'mean'):
        num_iter = len(pred_transforms)

        points_tar = data['points_tar_sample'].contiguous()

        tar_faces_tensor = data['points_based_neighs_tar'].reshape(
            points_tar.shape[0], -1, 9)
        # tar_faces_hull_vertices = utils.makefacevertices(
        #     data['V_src_hull'], data['F_src_hull'].long())
        losses_intersec = {}
        losses_chamfer = {}

        loss_reg = {}
        pred_src_transformed_final_sample = None
        # cal the intersection loss!
        R = (torch.norm(data['tar_box'][:, 0, :] - data['tar_box'][:, -1, :],
                        dim=-1,
                        p=2)).reshape(-1, 1)
        lines = None
        for ni in range(num_iter):
            pred_src_transformed_final_sample = se3.transform(
                pred_transforms[ni], data['points_src_sample'][..., :3])
            pred_src_faces_tensor = se3.transform(
                pred_transforms[ni], data['points_based_neighs_src']).reshape(
                    pred_src_transformed_final_sample.shape[0], -1, 9)
            losses_intersec['intersec_{}'.format(ni)] = torch.FloatTensor(
                [0]).to(self.device)
            losses_chamfer['chamfer_{}'.format(ni)] = torch.FloatTensor(
                [0]).to(self.device)
            # compute the loss;
            if lines is None:
                lines = Random_uniform_distribution_lines_batch_efficient_resample(
                    R, data['centers'], 10000,
                    pred_src_transformed_final_sample,
                    data['points_tar_sample'], self.device)
            tp_chamfer = chamfer_dist(
                points_tar, pred_src_transformed_final_sample).detach()
            # ignore the batch_loss?
            for j in range(pred_src_faces_tensor.shape[0]):
                losses_intersec['intersec_{}'.format(
                    ni)] += cal_loss_intersection_batch_whole_median_pts_lines(
                        1, 1, 5, 5, pred_src_faces_tensor[j:j + 1, :, :],
                        tar_faces_tensor[j:j + 1, :, :], lines[j:j + 1, :, :],
                        self.device)
            losses_intersec['intersec_{}'.format(ni)] /= num_iter

            losses_chamfer['chamfer_{}'.format(ni)] = tp_chamfer
        # Penalize outliers
        for ni1 in range(num_iter):
            ref_outliers_strength = (1.0 - torch.sum(
                endpoints['perm_matrices'][ni1], dim=1)) * _args.wt_inliers
            src_outliers_strength = (1.0 - torch.sum(
                endpoints['perm_matrices'][ni1], dim=2)) * _args.wt_inliers
            if reduction.lower() == 'mean':
                loss_reg['outlier_{}'.format(ni1)] = torch.mean(
                    ref_outliers_strength) + torch.mean(src_outliers_strength)
            elif reduction.lower() == 'none':
                loss_reg['outlier_{}'.format(ni1)] = torch.mean(ref_outliers_strength, dim=1) + \
                                                 torch.mean(src_outliers_strength, dim=1)

        discount_factor = 0.5  # Early iterations will be discounted, discount**(n-i).
        total_losses_inter = []
        total_losses_cd = []
        losses = {}
        for k1, k2 in zip(losses_intersec, losses_chamfer):
            discount = discount_factor**(num_iter -
                                         int(k1[k1.rfind('_') + 1:]) - 1)
            total_losses_inter.append(losses_intersec[k1] * discount)
            total_losses_cd.append(losses_chamfer[k2] * discount)
        losses['loss_intersection'] = torch.sum(
            torch.stack(total_losses_inter), dim=0)
        losses['loss_chamfer'] = torch.sum(torch.stack(total_losses_cd), dim=0)
        total_losses_reg = []

        for k in loss_reg:
            discount = discount_factor**(num_iter - int(k[k.rfind('_') + 1:]) -
                                         1)
            total_losses_reg.append(loss_reg[k] * discount)
        losses['loss_reg'] = torch.sum(torch.stack(total_losses_reg), dim=0)

        # cal the gt
        src_tar_transform = torch.zeros(
            pred_src_transformed_final_sample.shape[0], 4, 4).to(self.device)
        src_tar_transform[:, :3, :3] = data['R'].transpose(-1, -2)
        src_tar_transform[:, :3, 3] = data['T']
        gt_src_transformed_final_sample = se3.transform(
            src_tar_transform, data['points_src_sample'][..., :3])
        losses['loss_gt'] = torch.mean(
            (gt_src_transformed_final_sample -
             pred_src_transformed_final_sample).abs()).detach()

        # We clip the gradients!
        return losses, pred_src_transformed_final_sample

    def cal_gt_loss(self, pred_transforms, data):

        losses = {}
        num_iter = len(pred_transforms)

        points_tar = data['points_tar_sample']
        pred_src_transformed_final_sample = se3.transform(
            pred_transforms[num_iter - 1], data['points_src_sample'][..., :3])
        src_tar_transform = torch.zeros(
            pred_src_transformed_final_sample.shape[0], 4, 4).to(self.device)
        src_tar_transform[:, :3, :3] = data['R'].transpose(-1, -2)
        src_tar_transform[:, :3, 3] = data['T']
        gt_src_transformed_final_sample = se3.transform(
            src_tar_transform, data['points_src_sample'][..., :3])
        tp_loss_gt = torch.mean(
            (gt_src_transformed_final_sample -
             pred_src_transformed_final_sample).abs()).detach()
        # we just considera the ni, avoid the 0--> ni;
        # cal the intersection loss!
        tp_loss_chamfer = chamfer_dist(
            points_tar, pred_src_transformed_final_sample).detach()

        losses['loss_gt'] = tp_loss_gt
        losses['loss_chamfer'] = tp_loss_chamfer
        losses['pred_transform'] = pred_transforms[num_iter - 1]
        return losses


# Update the
# for unpervise learning reg:10.
# when training, we can change the weights of the loss term.

    def update(self, loss, pred_src_transformed_final, data):
        # loss_total = 0.01 * loss['loss_reg'] + 1.0 * loss['loss_gt']
        loss_total = 10 * loss['loss_reg'] + 1.0 * loss['loss_intersection']
        loss_total.backward()
        self.optimize.step()
        if self.epochs % self.frequency_save_results == 0 and self.counter % 100 == 0:
            # if self.epochs % self.frequency_save_results == 0:
            idx = np.random.choice(np.arange(self.batch_size))
            src_tar_transform = torch.zeros(1, 4, 4).to(self.device)
            src_tar_transform[0, :3, :3] = data['R'][idx].transpose(-1, -2)
            src_tar_transform[0, :3, 3] = data['T'][idx]
            V_src_tar = se3.transform(
                src_tar_transform,
                data['points_src_sample'][idx:idx + 1][..., :3]).reshape(
                    -1, 3)
            self.save_results(V_src_tar, pred_src_transformed_final[idx],
                              data['points_src_sample'][idx],
                              data['points_tar_sample'][idx])

        self.train_writer.add_scalar(
            './loss/intersection_batch',
            loss['loss_intersection'].detach().cpu().item(),
            self.epochs * self.train_ds + self.counter)
        self.train_writer.add_scalar(
            './loss/chamfer_batch', loss['loss_chamfer'].detach().cpu().item(),
            self.epochs * self.train_ds + self.counter)
        print(
            "\033[34m eh{:0d}/{:0d}, deh{:0d}/{:0d}, cd loss:{:4f}, loss_inter{:4f}, loss_reg{:4f}, loss_gt{:4f}\033[0m"
            .format(self.epochs, self.whole_epoch, self.counter, self.train_ds,
                    loss['loss_chamfer'].detach().cpu().item(),
                    loss['loss_intersection'].detach().cpu().item(),
                    loss['loss_reg'].detach().cpu().item(),
                    loss['loss_gt'].detach().cpu().item()))
        return

    def pretrained_params(self):
        for epoch in range(self.num_pretrained_epochs):
            for data in self.train_data_loader:
                self.optimize.zero_grad()
                dict_all_to_device(data, self.device)
                # pretrained our network!
                pred_transforms, endpoints = self.model(
                    data, 1)  # Use less iter during training

                R, T = pred_transforms[0][..., :3, :3], pred_transforms[0][
                    ..., :3, 3]
                R0 = torch.eye(3).reshape(1, 3, 3).repeat(R.shape[0], 1,
                                                          1).to(self.device)
                T0 = torch.zeros_like(T).to(self.device)
                loss_R = torch.mean((R - R0)**2)
                loss_translation = torch.mean((T - T0)**2)

                print(
                    "pretrained loss: rotation{:4f}, translation{:4f}".format(
                        loss_R.detach().cpu().item(),
                        loss_translation.detach().cpu().item()))
                loss = loss_R + loss_translation
                loss.backward()
                self.optimize.step()
                print("\033[34m epoch{:0d}/{:0d},loss:{:6f}!\033[0m".format(
                    epoch, self.num_pretrained_epochs,
                    loss.detach().cpu().item()))
        # Save the pretrained model!
        torch.save(self.model.state_dict(),
                   os.path.join(self.checkpoints_path, 'pretrained_model.pkl'))

    def run(self, num):
        self.whole_epoch = num
        self.re_load()
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimize,
            max_lr=2e-5,
            steps_per_epoch=1,
            epochs=100000,
            div_factor=1,
            final_div_factor=1,
            pct_start=0.001)
        for i in range(num):
            self.counter = 0
            self.model.train()
            self.train()
            self.eval()
            self.epochs += 1

    def __init__(self,
                 exps_path,
                 train_data_loader=None,
                 test_data_loader=None):
        super().__init__()
        utils.mkdir_ifnotexists(exps_path)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        self.root_path = os.path.join(exps_path, self.timestamp)
        utils.mkdir_ifnotexists(self.root_path)
        self.current_log_path = os.path.join(self.root_path, "log")
        utils.mkdir_ifnotexists(self.current_log_path)
        self.train_writer = SummaryWriter(self.current_log_path)

        self.checkpoints_path = os.path.join(self.root_path, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)

        self.results_path = os.path.join(self.root_path, "3d_models")

        utils.mkdir_ifnotexists(self.results_path)

        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        if self.test_data_loader is not None:
            self.current_log_path_eval = os.path.join(self.root_path,
                                                      "eval/log")
            utils.mkdir_ifnotexists(os.path.join(self.root_path, "eval"))
            utils.mkdir_ifnotexists(self.current_log_path)
            self.current_val_results = os.path.join(self.root_path,
                                                    "eval/results")
            utils.mkdir_ifnotexists(
                os.path.join(self.root_path, "eval/results"))
            self.test_writer = SummaryWriter(self.current_log_path_eval)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = get_model(_args)
        self.model.to(self.device)
        self.start_lr = 2e-3
        self.optimize = torch.optim.Adam(self.model.parameters(),
                                         lr=self.start_lr)
        self.is_continue = True
        self.epochs = 0
        self.chamfer_dist = chamfer_dist
        # self.chamfer_dist = ChamferDistance()
        self.frequency_save_results = _args.frequency_save_results
        self.frequency_update_lr = _args.frequency_update_lr
        if self.train_data_loader is not None:
            self.batch_size = self.train_data_loader.batch_size
            self.train_ds = len(self.train_data_loader) // self.batch_size
        self.counter = _args.counter
        self.whole_epoch = _args.whole_epoch
        self.re_load_model_path = _args.re_load_model_path
        self.num_pretrained_epochs = 10000
        self.is_pretrained = _args.is_pretrained
        if self.is_pretrained:
            self.scheduler = None

    def re_load(self):
        if self.re_load_model_path is None:
            self.re_load_model_path = os.path.join(self.root_path,
                                                   os.listdir[-1],
                                                   "checkpoints", "latest.pkl")
        if os.path.exists(self.re_load_model_path):
            print(self.re_load_model_path)
            self.model.load_state_dict(torch.load(self.re_load_model_path))
            print("\033[34m Reload the pretrained model!\033[0m")
        else:
            print("\033[34m Igonre reloading the pretrained model!\033[0m")
        return

    def save_results(self, V_src_tar, V_src, V_src_sample, V_tar):
        # save the model and results
        F = np.zeros([1, 3], np.int32)
        F_src = F
        F_tar = F
        if torch.is_tensor(V_src):
            V_src = V_src.detach().cpu().numpy()
        if torch.is_tensor(V_src_tar):
            V_src_tar = V_src_tar.detach().cpu().numpy()
        if torch.is_tensor(F_src):
            F_src = F_src.detach().cpu().numpy()

        if torch.is_tensor(V_tar):
            V_tar = V_tar.detach().cpu().numpy()
        if torch.is_tensor(F_tar):
            F_tar = F_tar.detach().cpu().numpy()
        if torch.is_tensor(V_src_sample):
            V_src_sample = V_src_sample.cpu().detach().numpy()
        igl.write_triangle_mesh(
            os.path.join(
                self.results_path,
                str(self.epochs) + "_pred_src_" + str(self.counter) + ".obj"),
            V_src, F_src)

        igl.write_triangle_mesh(
            os.path.join(
                self.results_path,
                str(self.epochs) + "_src_tar_" + str(self.counter) + ".obj"),
            V_src_tar, F_src)

        igl.write_triangle_mesh(
            os.path.join(
                self.results_path,
                str(self.epochs) + "_tar_" + str(self.counter) + ".obj"),
            V_tar, F_tar)
        igl.write_triangle_mesh(
            os.path.join(
                self.results_path,
                str(self.epochs) + "_src_" + str(self.counter) + ".obj"),
            V_src_sample, F_tar)
        # Save the check_points
        torch.save(
            self.model.state_dict(),
            os.path.join(self.checkpoints_path,
                         str(self.epochs) + 'model.pkl'))
        # Rewrite the latest model!
        torch.save(self.model.state_dict(),
                   os.path.join(self.checkpoints_path, 'latest.pkl'))
        return


def main():

    train_data_loader, test_data_loader = generate_datasets_human()
    exps_path = _args.exps_path

    Trainner = BaseRPMNet(exps_path=exps_path,
                          train_data_loader=train_data_loader,
                          test_data_loader=test_data_loader)
    Trainner.run(10000)


if __name__ == '__main__':
    main()
