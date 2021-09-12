"""
create model
Creator: Xiaoshui Huang
Date: 2020-06-19
"""
from se_math.so3 import inverse, transform
import torch
import numpy as np
from random import sample

import se_math.se3 as se3
import se_math.invmat as invmat

import igl
import os
import sys

sys.path.append('./../')
sys.path.append('./../../')
from loss import cal_loss_intersection_batch_whole_median_pts_lines, Reconstruction_point, Random_uniform_distribution_lines_batch_efficient_resample, chamfer_dist, Sample_neighs

from utils import npmat2euler

# we also make chamfer_loss for data!


def dict_all_to_device(tensor_dict, device):
    """Sends everything into a certain device """
    for k in tensor_dict:
        if isinstance(tensor_dict[k], torch.Tensor):
            tensor_dict[k] = tensor_dict[k].to(device)


def save_pred_gt_obj(V_src, V_pred, V_gt, V_tgt_trans, paths_src, paths_pred,
                     paths_gt, paths_gt_pred):
    Face = np.zeros(3).reshape(1, 3).astype(np.int32)
    for i in range(V_pred.shape[0]):
        igl.write_triangle_mesh(paths_src[i], V_src[i].numpy(), Face)
        igl.write_triangle_mesh(paths_pred[i], V_pred[i].numpy(), Face)
        igl.write_triangle_mesh(paths_gt[i], V_gt[i].numpy(), Face)
        igl.write_triangle_mesh(paths_gt_pred[i], V_tgt_trans[i].numpy(), Face)


# a global function to flatten a feature
def flatten(x):
    return x.view(x.size(0), -1)


# a global function to calculate max-pooling
def symfn_max(x):
    # [B, K, N] -> [B, K, 1]
    a = torch.nn.functional.max_pool1d(x, x.size(-1))
    return a


# a global function to generate mlp layers
def _mlp_layers(nch_input,
                nch_layers,
                b_shared=True,
                bn_momentum=0.1,
                dropout=0.0):
    """ [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    layers = []
    last = nch_input
    for i, outp in enumerate(nch_layers):
        if b_shared:
            weights = torch.nn.Conv1d(last, outp, 1)
        else:
            weights = torch.nn.Linear(last, outp)
        layers.append(weights)
        # layers.append(torch.nn.BatchNorm1d(outp, momentum=bn_momentum))
        layers.append(torch.nn.GroupNorm(8, outp))
        layers.append(torch.nn.ReLU())
        if b_shared == False and dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        last = outp
    return layers


# a class to generate MLP network
class MLPNet(torch.nn.Module):
    """ Multi-layer perception.
        [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    def __init__(self,
                 nch_input,
                 nch_layers,
                 b_shared=True,
                 bn_momentum=0.1,
                 dropout=0.0):
        super().__init__()
        list_layers = _mlp_layers(nch_input, nch_layers, b_shared, bn_momentum,
                                  dropout)
        self.layers = torch.nn.Sequential(*list_layers)

    def forward(self, inp):
        out = self.layers(inp)
        return out


# encoder network
class PointNet(torch.nn.Module):
    def __init__(self, dim_k=1024):
        super().__init__()
        scale = 1
        mlp_h1 = [int(64 / scale), int(64 / scale)]
        mlp_h2 = [int(64 / scale), int(128 / scale), int(dim_k / scale)]

        self.h1 = MLPNet(3, mlp_h1, b_shared=True).layers
        self.h2 = MLPNet(mlp_h1[-1], mlp_h2, b_shared=True).layers
        self.sy = symfn_max

    def forward(self, points):
        """ points -> features
            [B, N, 3] -> [B, K]
        """
        # for pointnet feature extraction
        x = points.transpose(1, 2)  # [B, 3, N]
        x = self.h1(x)
        x = self.h2(x)  # [B, K, N]
        x = flatten(self.sy(x))

        return x


# decoder network
class Decoder(torch.nn.Module):
    def __init__(self, num_points=2048, bottleneck_size=1024):
        super(Decoder, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        # self.bn1 = torch.nn.BatchNorm1d(bottleneck_size)
        # self.bn2 = torch.nn.BatchNorm1d(bottleneck_size // 2)
        # self.bn3 = torch.nn.BatchNorm1d(bottleneck_size // 4)
        self.bn1 = torch.nn.GroupNorm(8, bottleneck_size)
        self.bn2 = torch.nn.GroupNorm(8, bottleneck_size // 2)
        self.bn3 = torch.nn.GroupNorm(8, bottleneck_size // 4)
        self.fc1 = torch.nn.Linear(self.bottleneck_size, bottleneck_size)
        self.fc2 = torch.nn.Linear(self.bottleneck_size, bottleneck_size // 2)
        self.fc3 = torch.nn.Linear(bottleneck_size // 2, bottleneck_size // 4)
        self.fc4 = torch.nn.Linear(bottleneck_size // 4, self.num_points * 3)
        self.th = torch.nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        x = torch.nn.functional.relu(self.bn1(self.fc1(x)))
        x = torch.nn.functional.relu(self.bn2(self.fc2(x)))
        x = torch.nn.functional.relu(self.bn3(self.fc3(x)))
        x = self.th(self.fc4(x)) * 10
        x = x.view(batchsize, 3, self.num_points).transpose(1, 2).contiguous()
        return x


# the neural network of feature-metric registration
class SolveRegistration(torch.nn.Module):
    def __init__(self, ptnet, decoder=None):
        super().__init__()
        # network
        self.encoder = ptnet
        self.decoder = decoder
        # functions
        self.inverse = invmat.InvMatrix.apply
        self.exp = se3.Exp  # [B, 6] -> [B, 4, 4]
        self.transform = se3.transform  # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]

        # initialization for dt: [w1, w2, w3, v1, v2, v3], 3 rotation angles and 3 translation
        delta = 1.0e-2  # step size for approx. Jacobian (default: 1.0e-2)
        dt_initial = torch.autograd.Variable(
            torch.Tensor([delta, delta, delta, delta, delta, delta]))
        self.dt = torch.nn.Parameter(dt_initial.view(1, 6), requires_grad=True)

        # results
        self.last_err = None
        self.g_series = None  # for debug purpose
        self.prev_r = None
        self.g = None  # estimated transformation T
        self.device = None
        self.g_series_gpu = None

    # estimate T
    # noly return the encoder loss, but also return intersection loss

    def estimate_t(self,
                   data,
                   maxiter=5,
                   xtol=1.0e-7,
                   p0_zero_mean=True,
                   p1_zero_mean=True,
                   mode='train'):
        """
        give two point clouds, estimate the T by using IC algorithm
        :param p0: point cloud
        :param p1: point cloud
        :param maxiter: maximum iteration
        :param xtol: a threshold for early stop of transformation estimation
        :param p0_zero_mean: True: normanize p0 before IC algorithm
        :param p1_zero_mean: True: normanize p1 before IC algorithm
        :return: feature-metric projection error (r), encoder-decoder loss (loss_ende) and intersection loss!
        """
        p1 = data['points_src_sample']
        p0 = data['points_tar_sample']
        a0 = torch.eye(4).view(1, 4, 4).expand(p0.size(0), 4,
                                               4).to(p0)  # [B, 4, 4]
        a1 = torch.eye(4).view(1, 4, 4).expand(p1.size(0), 4,
                                               4).to(p1)  # [B, 4, 4]
        self.device = p1.device
        batch_size = p1.shape[0]
        # normalization
        if p0_zero_mean:
            p0_m = p0.mean(dim=1)  # [B, N, 3] -> [B, 3]
            a0 = a0.clone()
            a0[:, 0:3, 3] = p0_m
            q0 = p0 - p0_m.unsqueeze(1)
        else:
            q0 = p0
        if p1_zero_mean:
            p1_m = p1.mean(dim=1)  # [B, N, 3] -> [B, 3]
            a1 = a1.clone()
            a1[:, 0:3, 3] = -p1_m
            q1 = p1 - p1_m.unsqueeze(1)
        else:
            q1 = p1

        # use IC algorithm to estimate the transformation
        # generate the transform!
        g0 = torch.eye(4).to(q0).view(1, 4, 4).expand(q0.size(0), 4,
                                                      4).contiguous()
        r, g, loss_ende = self.ic_algo(g0, q0, q1, maxiter, xtol)
        # the g don't backgrade the gradinent？
        self.g = g
        # re-normalization
        if p0_zero_mean or p1_zero_mean:
            est_g = self.g
            if p0_zero_mean:
                est_g = a0.to(est_g).bmm(est_g)
            if p1_zero_mean:
                est_g = est_g.bmm(a1.to(est_g))
            self.g = est_g

            est_gs = self.g_series  # [M, B, 4, 4]
            if p0_zero_mean:
                est_gs = a0.unsqueeze(0).contiguous().to(est_gs).matmul(est_gs)
            if p1_zero_mean:
                est_gs = est_gs.matmul(a1.unsqueeze(0).contiguous().to(est_gs))
            self.g_series = est_gs

            est_gs_gpu = self.g_series_gpu  # [M, B, 4, 4]
            if p0_zero_mean:
                est_gs_gpu = a0.unsqueeze(0).contiguous().to(
                    est_gs_gpu).matmul(est_gs_gpu)
            if p1_zero_mean:
                est_gs_gpu = est_gs_gpu.matmul(
                    a1.unsqueeze(0).contiguous().to(est_gs_gpu))
            self.g_series_gpu = est_gs_gpu

        loss_pp_wise = (torch.mean(
            torch.abs(
                self.transform(self.g.unsqueeze(1), data['points_src_sample'])
                - self.transform(
                    torch.inverse(data['igt']).unsqueeze(1),
                    data['points_src_sample']))))
        if mode is 'train':
            R = (torch.norm(
                data['tar_box'][:, 0, :] - data['tar_box'][:, -1, :],
                dim=-1,
                p=2) * 0.5).reshape(-1, 1)
            lines = None
            points_ref = data['points_tar_sample'].contiguous()

            tar_faces_tensor = data['points_based_neighs_tar'].reshape(
                points_ref.shape[0], -1, 9)
            # if we used the transformed, we may generate better results!

            temp_g = self.g_series_gpu[-1]

            pred_src_transformed_final_sample = self.transform(
                temp_g.unsqueeze(1),
                data['points_src_sample'].contiguous()).detach()

            # pred_src_transformed_final_sample = data['points_src_sample']
            if lines is None:
                lines = Random_uniform_distribution_lines_batch_efficient_resample(
                    R, data['centers'], 15000,
                    pred_src_transformed_final_sample.contiguous(),
                    data['points_tar_sample'].contiguous(), self.device)

            # set our loss;
            loss_intersection = torch.FloatTensor([0]).to(self.device)
            for i in range(maxiter - 3, maxiter):
                temp_g = self.g_series_gpu[i]
                pred_src_transformed_final_sample = self.transform(
                    temp_g.unsqueeze(1), data['points_src_sample'])

                pred_src_faces_tensor = self.transform(
                    temp_g.unsqueeze(1),
                    data['points_based_neighs_src']).reshape(
                        pred_src_transformed_final_sample.shape[0], -1, 9)
                tp_loss_intersection = torch.FloatTensor([0]).to(self.device)
                for j in range(pred_src_faces_tensor.shape[0]):
                    tp_loss_intersection += cal_loss_intersection_batch_whole_median_pts_lines(
                        1, 1, 5, 5, pred_src_faces_tensor[j:j + 1, :, :],
                        tar_faces_tensor[j:j + 1, :, :], lines[j:j + 1, :, :],
                        self.device) / 5.0
                loss_intersection = loss_intersection + \
                    tp_loss_intersection*0.5**(maxiter-i-1)

            loss_chamfer = chamfer_dist(pred_src_transformed_final_sample,
                                        data['points_tar_sample'])

            return r, loss_ende, loss_intersection / batch_size, loss_pp_wise, loss_chamfer
        return r, loss_ende, loss_pp_wise,

    # IC algorithm
    # encoder, we just use the chamfer!
    def ic_algo(self, g0, p0, p1, maxiter, xtol):
        """
        use IC algorithm to estimate the increment of transformation parameters
        :param g0: initial transformation
        :param p0: point cloud
        :param p1: point cloud
        :param maxiter: maxmimum iteration
        :param xtol: a threashold to check increment of transformation  for early stop
        :return: feature-metric projection error (r), updated transformation (g), encoder-decoder loss
        """
        training = self.encoder.training
        # training = self.decoder.training
        batch_size = p0.size(0)

        self.last_err = None
        g = g0
        self.g_series = torch.zeros(maxiter + 1, *g0.size(), dtype=g0.dtype)
        self.g_series[0] = g0.clone()
        self.g_series_gpu = torch.zeros(maxiter, *g0.size(),
                                        dtype=g0.dtype).to(self.device)

        # generate the features
        f0 = self.encoder(p0)
        f1 = self.encoder(p1)

        # task 1
        loss_enco_deco = 0.0
        if self.decoder is not None:
            # we generate the decoder f0?
            # make an encoder decoder!
            decoder_out_f0 = self.decoder(f0)
            decoder_out_f1 = self.decoder(f1)

            # the decoder meets AE!
            p0_dist1, p0_dist2 = self.chamfer_loss(
                p0.contiguous(), decoder_out_f0)  # loss function
            loss_net0 = (torch.mean(p0_dist1)) + (torch.mean(p0_dist2))
            p1_dist1, p1_dist2 = self.chamfer_loss(
                p1.contiguous(), decoder_out_f1)  # loss function
            loss_net1 = (torch.mean(p1_dist1)) + (torch.mean(p1_dist2))
            loss_enco_deco = loss_net0 + loss_net1

        # self.encoder.eval()  # and fix them BN.
        # if fix, ho to backward gradients？
        # task 2
        f0 = self.encoder(p0)  # [B, N, 3] -> [B, K]
        # approx. J by finite difference
        dt = self.dt.to(p0).expand(batch_size,
                                   6)  # convert to the type of p0. [B, 6]
        J = self.approx_Jac(p0, f0, dt)
        # compute pinv(J) to solve J*x = -r
        try:
            Jt = J.transpose(1, 2)  # [B, 6, K]
            H = Jt.bmm(J)  # [B, 6, 6]
            # H = H + u_lamda * iDentity
            B = self.inverse(H)
            pinv = B.bmm(Jt)  # [B, 6, K]
        except RuntimeError as err:
            self.last_err = err
            f1 = self.encoder(p1)  # [B, N, 3] -> [B, K]
            r = f1 - f0
            self.ptnet.train(training)
            return r, g, -1

        itr = 0
        r = None
        # we
        for itr in range(maxiter):
            p = self.transform(g.unsqueeze(1),
                               p1)  # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]
            f1 = self.encoder(p)  # [B, N, 3] -> [B, K]
            r = f1 - f0  # [B,K]
            # generate the r！
            dx = -pinv.bmm(r.unsqueeze(-1)).view(batch_size, 6)

            check = dx.norm(p=2, dim=1, keepdim=True).max()
            if float(check) < xtol:
                if itr == 0:
                    self.last_err = 0  # no update.
                break

            g = self.update(g, dx)
            self.g_series_gpu[itr] = g
            self.g_series[itr + 1] = g.clone()
            self.prev_r = r

        self.encoder.train(training)
        return r, g, loss_enco_deco

    # estimate Jacobian matrix
    def approx_Jac(self, p0, f0, dt):
        # p0: [B, N, 3], Variable
        # f0: [B, K], corresponding feature vector
        # dt: [B, 6], Variable
        # Jk = (ptnet(p(-delta[k], p0)) - f0) / delta[k]
        batch_size = p0.size(0)
        num_points = p0.size(1)

        # compute transforms
        transf = torch.zeros(batch_size, 6, 4, 4).to(p0)
        for b in range(p0.size(0)):
            d = torch.diag(dt[b, :])  # [6, 6]
            D = self.exp(-d)  # [6, 4, 4]
            transf[b, :, :, :] = D[:, :, :]
        transf = transf.unsqueeze(2).contiguous()  # [B, 6, 1, 4, 4]
        p = self.transform(transf,
                           p0.unsqueeze(1))  # x [B, 1, N, 3] -> [B, 6, N, 3]

        f0 = f0.unsqueeze(-1)  # [B, K, 1]
        f1 = self.encoder(p.view(-1, num_points, 3))
        f = f1.view(batch_size, 6, -1).transpose(1, 2)  # [B, K, 6]

        df = f0 - f  # [B, K, 6]
        J = df / dt.unsqueeze(1)  # [B, K, 6]

        return J

    # update the transformation
    def update(self, g, dx):
        # [B, 4, 4] x [B, 6] -> [B, 4, 4]
        dg = self.exp(dx)
        return dg.matmul(g)

    # calculate the chamfer loss
    def chamfer_loss(self, a, b):
        x, y = a, b
        bs, num_points, points_dim = x.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        # diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)
        diag_ind = torch.arange(0, num_points)
        rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
        ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return torch.min(P, 1)[0], torch.min(P, 2)[0]

    @staticmethod
    def rsq(r):
        # |r| should be 0
        z = torch.zeros_like(r)
        return torch.nn.functional.mse_loss(r, z, reduction='sum')

    @staticmethod
    def comp(g, igt):
        """ |g*igt - I| (should be 0) """
        assert g.size(0) == igt.size(0)
        assert g.size(1) == igt.size(1) and g.size(1) == 4
        assert g.size(2) == igt.size(2) and g.size(2) == 4
        A = g.matmul(igt)
        I = torch.eye(4).to(A).view(1, 4, 4).expand(A.size(0), 4, 4)
        return torch.nn.functional.mse_loss(A, I, reduction='mean') * 16

    @staticmethod
    def comp_inv(g, igt):
        """ |g*igt - I| (should be 0) """
        assert g.size(0) == igt.size(0)
        assert g.size(1) == igt.size(1) and g.size(1) == 4
        assert g.size(2) == igt.size(2) and g.size(2) == 4
        # A = g.matmul(igt)
        gt = torch.inverse(igt)
        # I = torch.eye(4).to(A).view(1, 4, 4).expand(A.size(0), 4, 4)

        return torch.nn.functional.mse_loss(g, gt, reduction='mean')


# main algorithm class
class FMRTrain:
    def __init__(self, dim_k, num_points, train_type):
        self.dim_k = dim_k
        self.num_points = num_points
        self.max_iter = 5  # max iteration time for IC algorithm
        # 0: unsupervised, 1: semi-supervised see. self.compute_loss()
        self._loss_type = train_type

        self.transform = se3.transform  # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]

    def create_model(self):
        # Encoder network: extract feature for every point. Nx1024
        ptnet = PointNet(dim_k=self.dim_k)
        # Decoder network: decode the feature into points
        decoder = Decoder(num_points=self.num_points)
        # feature-metric ergistration (fmr) algorithm: estimate the transformation T
        fmr_solver = SolveRegistration(ptnet, decoder)
        return fmr_solver

    def compute_loss(self, solver, data, device, mode='train', maxiter=5):
        # p0, p1, igt = data
        # p0 = p0.to(device)  # template
        # p1 = p1.to(device)  # source
        # igt = igt.to(device)  # igt: p0 -> p1
        dict_all_to_device(data, device)
        p1 = data['points_src_sample']
        p0 = data['points_tar_sample']
        igt = data['igt']

        if mode is 'train':
            r, loss_ende, loss_intersection, loss_pp_wise, loss_chamfer = solver.estimate_t(
                data, self.max_iter, mode=mode)
        else:
            # test model!
            r, loss_ende, loss_pp_wise = solver.estimate_t(data,
                                                           maxiter,
                                                           mode=mode)
        loss_r = solver.rsq(r)
        est_g = solver.g
        # generate the difference between the pred and gt!
        loss_g = solver.comp_inv(est_g, igt)

        # unsupervised learning, set max_iter=0
        if self.max_iter == 0:
            return loss_ende

        # semi-supervised learning, set max_iter>0
        if self._loss_type == 0:
            loss = loss_ende
        elif self._loss_type == 1:
            loss = loss_ende + loss_g
        elif self._loss_type == 2:
            loss = loss_r + loss_g
        else:
            loss = loss_g

        # we need use the multiple indicators to measure the quality!
        np_pred_rotation = est_g[:, :3, :3].transpose(
            2, 1).detach().cpu().numpy()
        np_pred_euler = npmat2euler(np_pred_rotation, 'xyz')
        np_gt_rotation = data['R'].detach().cpu().numpy()
        np_gt_euler = npmat2euler(np_gt_rotation, 'xyz')
        loss_rotation_euler_mae = np.mean(np.abs(np_pred_euler - np_gt_euler))
        loss_rotation_euler_rmse = np.sqrt(
            np.mean((np_pred_euler - np_gt_euler)**2))

        np_loss = {
            'loss_rot_euler_mae': loss_rotation_euler_mae,
            'loss_rot_euler_rmse': loss_rotation_euler_rmse
        }

        # set the weights
        if mode is 'train':
            return 0.01 * loss_ende + 1.0 * loss_intersection + .0 * loss_g + 0.0 * loss_chamfer, loss_g.detach(
            ), loss_intersection.detach(), loss_pp_wise.detach(
            ), loss_ende.detach(), np_loss
        return loss_g, loss_g.detach(), loss_pp_wise.detach(
        ), loss_ende.detach(), np_loss

    def train(self,
              model,
              trainloader,
              optimizer,
              device,
              epoch,
              train_writer=None):
        model.train()

        Debug = True
        total_loss = 0
        total_loss_gt = 0
        total_loss_intersection = 0
        total_loss_pp_wise = 0
        total_loss_encoder = 0
        total_loss_rot_euler_mae = 0
        total_loss_rot_euler_rmse = 0
        if Debug:
            epe = 0
            count = 0
            count_mid = 9
        for i, data in enumerate(trainloader):
            loss, loss_gt, loss_intersection, loss_pp_wise, loss_ende, np_loss = self.compute_loss(
                model, data, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_item = loss.item()
            total_loss += loss_item
            total_loss_gt += loss_gt.item()
            total_loss_pp_wise += loss_pp_wise.item()
            total_loss_intersection += loss_intersection.item()
            total_loss_encoder += loss_ende.item()
            total_loss_rot_euler_mae += np_loss['loss_rot_euler_mae']
            total_loss_rot_euler_rmse += np_loss['loss_rot_euler_rmse']
            if Debug:
                epe += loss_item
                if count % 10 == 0:
                    print('i=%d, fmr_loss=%f ' % (i, float(epe) /
                                                  (count_mid + 1)))
                    epe = 0.0
            count += 1
            print(
                "ba/ep{:0d}/{:0d},l_insec:{:4f}, l_gt{:4f},l_pp_w{:4f}, l_en{:4f}, l_rot_eu_mae{:4f}, l_rot_eu_rmse{:4f}"
                .format(i, epoch, loss_intersection.item(), loss_gt.item(),
                        loss_pp_wise.item(), loss_ende.item(),
                        np_loss['loss_rot_euler_mae'],
                        np_loss['loss_rot_euler_rmse']))
        ave_loss = float(total_loss) / count
        ave_loss_gt = float(total_loss_gt) / count
        ave_loss_intersection = float(total_loss_intersection) / count
        ave_loss_wise = float(total_loss_pp_wise) / count
        ave_loss_encoder = float(total_loss_encoder) / count
        ave_loss_rot_euler_mae = (float)(total_loss_rot_euler_mae) / count
        ave_loss_rot_euler_rmse = (float)(total_loss_rot_euler_rmse) / count
        if train_writer is not None:
            train_writer.add_scalar('./loss/loss_sum', ave_loss, epoch)
            train_writer.add_scalar('./loss/loss_gt', ave_loss_gt, epoch)
            train_writer.add_scalar('./loss/loss_intersec',
                                    ave_loss_intersection, epoch)
            train_writer.add_scalar('./loss/loss_wise_mse', ave_loss_wise,
                                    epoch)
            train_writer.add_scalar('./loss/loss_ende', ave_loss_encoder,
                                    epoch)
            train_writer.add_scalar('./lr', optimizer.param_groups[0]['lr'],
                                    epoch)
            train_writer.add_scalar('./loss/loss_rot_euler_mae',
                                    ave_loss_rot_euler_mae, epoch)
            train_writer.add_scalar('./loss/loss_rot_euler_rmse',
                                    ave_loss_rot_euler_rmse, epoch)


# \033[36m,test gt:{:4f}, pp_wise:{:4f}, rot_mae{:4f}, rot_rmse{:4f}\033[0m
        print(
            " \033[36m,train:l_gt:{:4f}, l_intersec:{:4f}, l_pp_wise{:4f}, l_encoder{:4f}, l_rot_eu_mae{:4f}, l_rot_eu_rmse{:4f} \033[0m, "
            .format(ave_loss_gt, ave_loss_intersection, ave_loss_wise,
                    ave_loss_encoder, ave_loss_rot_euler_mae,
                    ave_loss_rot_euler_rmse))
        return ave_loss

    def validate(self, model, testloader, device, epoch, save_results=None):
        # model.eval()
        vloss = 0.0
        vloss_gt = 0.0
        vloss_pp_wise = 0.0
        vloss_rot_euler_mae = 0.0
        vloss_rot_euler_rmse = 0.0
        count = 0
        count_i = 0
        with torch.no_grad():
            for i, data in enumerate(testloader):
                loss_net, loss_gt, loss_pp_wise, loss_ende, np_loss = self.compute_loss(
                    model, data, device, mode='test')
                vloss += loss_net.item()
                vloss_gt += loss_gt.item()
                vloss_pp_wise += loss_pp_wise.item()
                vloss_rot_euler_mae += np_loss['loss_rot_euler_mae']
                vloss_rot_euler_rmse += np_loss['loss_rot_euler_rmse']
                count += 1
                print("Test:sample{:0d},loss_pp_wise:{:4f}".format(
                    i, loss_pp_wise.item()))
                if epoch % 10 == 0:
                    est_g = model.g  # (1, 4, 4)

                    igt = data['igt']
                    ig_gt = igt.cpu().contiguous().view(-1, 4,
                                                        4)  # --> [1, 4, 4]
                    g_hat = est_g.cpu().contiguous().view(-1, 4,
                                                          4)  # --> [1, 4, 4]
                    p1 = data['points_src_sample']
                    p0 = data['points_tar_sample']
                    if save_results is not None:

                        paths_pred = []
                        paths_gt = []
                        paths_src = []
                        paths_gt_pred = []
                        src_transform = self.transform(est_g.unsqueeze(1), p1)
                        src_transform_sample = self.transform(
                            est_g.unsqueeze(1), data['points_src_sample'])
                        tgt_transform = self.transform(igt.unsqueeze(1), p0)
                        V_src = p0.cpu().detach()
                        V_pred = src_transform.cpu().detach()
                        V_gt = p1.cpu().detach()
                        V_tgt_trans = tgt_transform.cpu().detach()
                        for j in range(p0.shape[0]):

                            paths_pred.append(
                                os.path.join(
                                    save_results,
                                    str(epoch) + "pred_src" + str(count_i) +
                                    ".obj"))
                            paths_gt.append(
                                os.path.join(
                                    save_results,
                                    str(epoch) + "gt" + str(count_i) + ".obj"))
                            paths_src.append(
                                os.path.join(
                                    save_results,
                                    str(epoch) + "src" + str(count_i) +
                                    ".obj"))
                            paths_gt_pred.append(
                                os.path.join(
                                    save_results,
                                    str(epoch) + "pred_gt" + str(count_i) +
                                    ".obj"))

                            F = np.zeros([1, 3]).astype(np.int32)
                            igl.write_obj(
                                paths_gt_pred[j].replace(
                                    'pred_gt', 'transformed_sample', 1),
                                src_transform_sample.cpu().detach().numpy().
                                reshape(-1, 3), F)
                            igl.write_obj(
                                paths_gt_pred[j].replace(
                                    'pred_gt', 'src_sample', 1),
                                data['points_src_sample'].cpu().detach().numpy(
                                ).reshape(-1, 3), F)
                            igl.write_obj(
                                paths_gt_pred[j].replace(
                                    'pred_gt', 'tar_sample', 1),
                                data['points_tar_sample'].cpu().detach().numpy(
                                ).reshape(-1, 3), F)

                            count_i += 1
                        save_pred_gt_obj(V_src, V_pred, V_gt, V_tgt_trans,
                                         paths_src, paths_pred, paths_gt,
                                         paths_gt_pred)

        ave_vloss = float(vloss) / count
        ave_vloss_gt = float(vloss_gt) / count
        ave_vloss_pp_wise = float(vloss_pp_wise) / count
        ave_vloss_rot_euler_mae = float(vloss_rot_euler_mae) / count
        ave_vloss_rot_euler_rmse = float(vloss_rot_euler_rmse) / count
        print(
            "\033[36m,test gt:{:4f}, pp_wise:{:4f}, rot_mae{:4f}, rot_rmse{:4f}\033[0m, "
            .format(ave_vloss_gt, ave_vloss_pp_wise, ave_vloss_rot_euler_mae,
                    ave_vloss_rot_euler_rmse))
        return ave_vloss


class FMRTest:
    def __init__(self, args):
        self.filename = args.outfile
        self.dim_k = args.dim_k
        self.max_iter = 10  # max iteration time for IC algorithm
        self._loss_type = 3  # see. self.compute_loss()
        self.transform = se3.transform  # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]

    def create_model(self):
        # Encoder network: extract feature for every point. Nx1024
        ptnet = PointNet(dim_k=self.dim_k)
        # feature-metric ergistration (fmr) algorithm: estimate the transformation T
        fmr_solver = SolveRegistration(ptnet)
        return fmr_solver

    # we save the results!
    # pay attention to final results!
    def evaluate(self,
                 solver,
                 testloader,
                 device,
                 save_results=None,
                 writer=None):
        solver.eval()
        with open(self.filename, 'w') as fout:
            self.eval_1__header(fout)
            count_i = 0
            total_loss_pp_wise = 0
            total_loss_gt = 0
            with torch.no_grad():
                for i, data in enumerate(testloader):
                    # p0, p1, igt = data  # igt: p0->p1
                    dict_all_to_device(data, device)
                    p1 = data['points_src_sample']
                    p0 = data['points_tar_sample']
                    igt = data['igt']
                    # igt =
                    # # compute trans from p1->p0
                    # g = se3.log(igt)  # --> [-1, 6]
                    # igt = se3.exp(-g)  # [-1, 4, 4]
                    # p0, p1 = self.ablation_study(p0, p1)

                    p0 = p0.to(device)  # template (1, N, 3)
                    p1 = p1.to(device)  # source (1, M, 3)
                    # When we evaluate, we ignore the chafer, ignore any loss function!
                    r, loss_ende, loss_pp_wise = solver.estimate_t(
                        data, self.max_iter, mode='test')
                    total_loss_pp_wise += loss_pp_wise
                    est_g = solver.g  # (1, 4, 4)

                    ig_gt = igt.cpu().contiguous().view(-1, 4,
                                                        4)  # --> [1, 4, 4]
                    g_hat = est_g.cpu().contiguous().view(-1, 4,
                                                          4)  # --> [1, 4, 4]

                    dg = g_hat.bmm(ig_gt)  # if correct, dg == identity matrix.
                    dx = se3.log(
                        dg)  # --> [1, 6] (if corerct, dx == zero vector)
                    dn = dx.norm(p=2, dim=1)  # --> [1]
                    dm = dn.mean()

                    self.eval_1__write(fout, ig_gt, g_hat)
                    print('test, %d/%d, %f, %f' %
                          (i, len(testloader), dm, loss_pp_wise))
                    if writer is not None:
                        writer.add_scalar('./loss/test', dm, i)
                    # p = self.transform(g.unsqueeze(1),
                    #                    p1)  # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]
                    # est_g:p1--->p0
                    # igt: p0-->p1
                    if save_results is not None:
                        paths_pred = []
                        paths_gt = []
                        paths_src = []
                        paths_gt_pred = []
                        src_transform = self.transform(est_g.unsqueeze(1), p1)

                        tgt_transform = self.transform(igt.unsqueeze(1), p0)
                        V_src = p0.cpu().detach()
                        V_pred = src_transform.cpu().detach()
                        V_gt = p1.cpu().detach()
                        V_tgt_trans = tgt_transform.cpu().detach()
                        for i in range(p0.shape[0]):

                            paths_pred.append(
                                os.path.join(save_results,
                                             str(count_i) + "pred_src.obj"))
                            paths_gt.append(
                                os.path.join(save_results,
                                             str(count_i) + "gt.obj"))
                            paths_src.append(
                                os.path.join(save_results,
                                             str(count_i) + "src.obj"))
                            paths_gt_pred.append(
                                os.path.join(save_results,
                                             str(count_i) + "pred_gt.obj"))

                            count_i += 1
                        save_pred_gt_obj(V_src, V_pred, V_gt, V_tgt_trans,
                                         paths_src, paths_pred, paths_gt,
                                         paths_gt_pred)

    def ablation_study(self, p0, p1, add_noise=False, add_density=False):
        # ablation study
        # mesh = self.plyread("./box1Kinect1.ply")
        # p0 = torch.tensor(mesh).to(device).unsqueeze(0)
        # mesh = self.plyread("./box11.ply")
        # p1 = torch.tensor(mesh).to(device).unsqueeze(0)

        # add noise
        if add_noise:
            p1 = torch.tensor(np.float32(np.random.normal(p1, 0.01)))

        # add outliers
        if add_density:
            density_ratio = 0.5
            pts_num = p1.shape[0]
            sampleNum = int(pts_num *
                            density_ratio)  # the number of remaining points
            if pts_num > sampleNum:
                num = sample(range(1, pts_num), sampleNum)
            elif pts_num > 0:
                num = range(0, pts_num)
            else:
                print("No points in this point cloud!")
                return
            p1 = p1[num, :]
        return p0, p1

    def eval_1__header(self, fout):
        cols = [
            'h_w1', 'h_w2', 'h_w3', 'h_v1', 'h_v2', 'h_v3', 'g_w1', 'g_w2',
            'g_w3', 'g_v1', 'g_v2', 'g_v3'
        ]  # h: estimated, g: ground-truth twist vectors
        print(','.join(map(str, cols)), file=fout)
        fout.flush()

    def eval_1__write(self, fout, ig_gt, g_hat):
        x_hat = se3.log(g_hat)  # --> [-1, 6]
        mx_gt = se3.log(ig_gt)  # --> [-1, 6]
        for i in range(x_hat.size(0)):
            x_hat1 = x_hat[i]  # [6]
            mx_gt1 = mx_gt[i]  # [6]
            vals = torch.cat((x_hat1, -mx_gt1))  # [12]
            valn = vals.cpu().numpy().tolist()
            print(','.join(map(str, valn)), file=fout)
        fout.flush()
