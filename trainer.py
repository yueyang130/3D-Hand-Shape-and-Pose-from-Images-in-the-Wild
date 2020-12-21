import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from model import resnet34_Mano
from utils import get_scheduler, \
    weights_init, get_criterion, get_model_list
import os
from scripts.make_dataset import show_pts_on_img, show_line_on_img, show_3dmesh
from scripts.segment import show_mask_on_img
import scripts.prepare_dataset.mm2px as mm2px

def __ee__(x) :
    if type(x) is torch.Tensor :
        return x.item()
    else :
        return x


class EncoderTrainer(nn.Module):
    """
    We pre-train the encoder to ensure that
    the camera and hand parameters converge towards
    acceptable values.
    """

    def __init__(self, params, ispretrain):
        super(EncoderTrainer, self).__init__()
        self.ispretrain = ispretrain
        self.input_option = params['input_option']
        self.weight = params

        # initiate the network modules
        #self.model = resnet34_Mano(ispretrain=ispretrain, input_option=params['input_option'])
        self.model = torch.nn.DataParallel(resnet34_Mano(input_option=params['input_option']))
        self.model = self.model.module
        self.mean_3d = torch.zeros(3)

        # setup the optimizer
        lr = params.lr
        beta1 = params.beta1
        beta2 = params.beta2
        #p_view = self.model.state_dict()
        self.encoder_opt = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad],
                                    lr = lr, betas=(beta1, beta2), weight_decay=params.weight_decay)
        self.encoder_opt = nn.DataParallel(self.encoder_opt).module

        self.encoder_scheduler = get_scheduler(self.encoder_opt, params)

        # set loss fn
        if self.ispretrain:
            self.param_recon_criterion = get_criterion(params['pretrain_loss_fn'])

        # Network weight initialization
        self.model.apply(weights_init(params.init))

        self.transformer = mm2px.JointTransfomer('BB')

    @staticmethod
    def convert_vec_to_2d(vec):
        # vec : (bs, 42)
        u = vec[:, 0::2].unsqueeze(2)  # (bs, 21, 1)
        v = vec[:, 1::2].unsqueeze(2)
        uv = torch.cat([u,v], dim=2)
        return uv


    def comupte_2d_joint_loss(self, joint_rec, joint_gt):
        joint_rec = self.convert_vec_to_2d(joint_rec)

        valid_idx = (joint_gt[:, :, 2] == 1)  # (bs, 21)
        valid_idx = valid_idx.unsqueeze(dim=2).repeat((1,1,2))
        valid_joint_rec = joint_rec[valid_idx]
        valid_joint_gt  = joint_gt[:,:,:2][valid_idx]
        loss1 = torch.mean(torch.abs(valid_joint_rec - valid_joint_gt))
        # extra loss2, which helps to train translation
        pc_valid_idx = (joint_gt[:, 0, 2] == 1)
        valid_pc_joint_rec = joint_rec[:, 0, :2][pc_valid_idx]
        valid_pc_joint_gt  = joint_gt[:, 0, :2][pc_valid_idx]
        loss2 = torch.mean(torch.abs(valid_pc_joint_rec - valid_pc_joint_gt))
        # extra loss3, which helps to train rotation
        finger_end_idx = [1, 5, 9, 13, 17]
        fend_valid_idx = (joint_gt[:, finger_end_idx, 2] == 1)
        fend_valid_joint_rec = (joint_rec[:, finger_end_idx, :2] - joint_rec[:, [0], :2])[fend_valid_idx]
        fend_valid_joint_gt = (joint_gt[:, finger_end_idx, :2] - joint_gt[:,[0], :2])[fend_valid_idx]
        loss3 = torch.mean(torch.abs(fend_valid_joint_gt - fend_valid_joint_rec))

        #ret = loss1 + 0 * loss2 + 0 * loss3
        ret = loss1

        return ret

    def compute_3d_joint_loss_norm(self, joint_rec, joint_gt, valid, detach):
        bs = valid.shape[0]
        valid_index = torch.arange(bs)[valid == 1]
        if valid_index.shape[0] == 0 :
            return torch.tensor(0.).cuda()

        joint_gt_valid = joint_gt[valid_index]
        joint_rec_valid = joint_rec[valid_index]
        # convert
        if detach:
            joint_gt_norm = (joint_gt_valid - torch.mean(joint_gt_valid, dim=1, keepdim=True).detach()) / torch.std(
                joint_gt_valid, dim=1, keepdim=True).detach()
            joint_rec_norm = (joint_rec_valid - torch.mean(joint_rec_valid, dim=1, keepdim=True).detach()) / torch.std(
                joint_rec_valid, dim=1, keepdim=True).detach()
        else:
            joint_gt_norm = (joint_gt_valid - torch.mean(joint_gt_valid, dim=1, keepdim=True)) / torch.std(
                joint_gt_valid, dim=1, keepdim=True)
            joint_rec_norm = (joint_rec_valid - torch.mean(joint_rec_valid, dim=1, keepdim=True)) / torch.std(
                joint_rec_valid, dim=1, keepdim=True)

        tmp = joint_rec_norm - joint_gt_norm
        loss1 = torch.mean(tmp ** 2)

        return loss1

    def compute_3d_joint_loss(self, joint_rec, joint_gt, valid, scale):
        """
            weak perspective model
        """
        bs = valid.shape[0]
        valid_index = torch.arange(bs)[valid == 1]
        if valid_index.shape[0] == 0:
            return torch.tensor(0.).cuda()

        finger_end_idx = [1,5,9,13,17]
        f  =  float(self.transformer.fx)
        z = torch.mean(joint_gt[valid_index][:,:,2], dim=1)

        # convert
        joint_gt_scale  = joint_gt[valid_index] * torch.tensor(f) / (z * scale[valid_index]).unsqueeze(1).unsqueeze(2)
        joint_rec_scale = joint_rec[valid_index]

        joint_gt_norm = joint_gt_scale - joint_gt_scale.mean(dim=1, keepdim=True)
        joint_rec_norm = joint_rec_scale - joint_rec_scale.mean(dim=1, keepdim=True)

        tmp = joint_rec_norm - joint_gt_norm
        loss1 = torch.mean(tmp**2)


        # # extra loss 2: set the palm centre as start point and scale
        # fend_gt_norm = joint_gt[:,finger_end_idx,:] - joint_gt[:,[0],:] + torch.tensor(10**-8).cuda()
        # fend_rec_norm = joint_rec[:,finger_end_idx,:] - joint_rec[:,[0],:]
        # scale = torch.sqrt(torch.pow(fend_rec_norm, 2).sum(dim=2) / torch.pow(fend_gt_norm, 2).sum(dim=2)).detach().unsqueeze(dim=2)
        # fend_gt_norm_scale = fend_gt_norm * scale
        #
        #
        # tmp = fend_gt_norm_scale  - fend_rec_norm
        # loss2 = torch.mean(tmp**2)
        #
        #
        # # extra loss 3: set five fingers' end as start point
        # loss_list = []
        # for idx in finger_end_idx:
        #     point_idx = [idx+i for i in xrange(1,4)]
        #     fpoint_gt_norm = joint_gt[:, point_idx, :] - joint_gt[:, [idx], :] + torch.tensor(10**-8).cuda()
        #     fpoint_rec_norm = joint_rec[:, point_idx, :] - joint_rec[:, [idx], :]
        #
        #     scale = torch.sqrt(torch.pow(fpoint_rec_norm, 2).sum(dim=2) / torch.pow(fpoint_gt_norm, 2).sum(dim=2)).detach().unsqueeze(dim=2)
        #     fpoint_gt_norm = fpoint_gt_norm * scale
        #
        #     tmp = fpoint_gt_norm - fpoint_rec_norm
        #     loss_list.append(torch.mean(tmp**2))
        #
        # loss3 = sum(loss_list) / len(loss_list)

        #ret = 1. * loss1 + 1. * loss2 + 5. *  loss2
        ret = loss1

        return ret


    def compute_mask_loss(self, mesh2d, mask, valid):
        # for debug
        # mesh2d = mesh2d.cpu()
        # mask = mask.cpu()
        # valid = valid.cpu()
        bs = mesh2d.shape[0]

        mesh2d = self.convert_vec_to_2d(mesh2d)  # [bs, 778, 2]
        new_mesh2d = mesh2d.unsqueeze(1)  # [bs, 1, 778, 2]

        # For mesh, mesh[0] means width; mesh[1] means height
        # For grid_sample's grid parameter , the first coordinate means width; the second means height
        # so we don't reverse mesh's x and y

        # mesh may out of image
        size = mask.shape[1]
        new_mesh2d[new_mesh2d < 0] = 0
        new_mesh2d[new_mesh2d >= size] = size - 1
        # normalized to [-1,1]
        new_mesh2d = new_mesh2d / (size - 1) * 2 - 1  # [bs, 1, 778, 2]

        mask = mask.unsqueeze(1).float()   # [bs, 1, H, W]

        ret = torch.nn.functional.grid_sample(mask, new_mesh2d)  # [bs, 1, 1, 778]
        ret = ret.squeeze(1).squeeze(1)

        valid_index = torch.arange(bs)[valid == 1]
        if valid_index.shape[0] == 0:
            return torch.tensor(0.).cuda()
        ret = ret[valid_index]
        #ret = torch.tensor(1.) - torch.mean(ret)
        ret = torch.tensor(1.).cuda() - torch.mean(ret)
        return ret

    def compute_param_reg_loss(self, vec):
        assert vec.shape[1] == 22
        beta_weight = 10**4
        beta = vec[:, -10:]
        theta = vec[:, -16:-10]
        ret = torch.mean(theta**2) + beta_weight * torch.mean(beta**2)
        return ret

    def encoder_update(self, x, gt_2d, gt_3d, mask, valid):
        assert not self.ispretrain, "the method can be only used in train mode"

        self.encoder_opt.zero_grad()
        x, valid = torch.detach(x), torch.detach(valid)
        gt_2d, gt_3d, mask = torch.detach(gt_2d), torch.detach(gt_3d), torch.detach(mask)
        # forward
        x2d, x3d, param = self.model(x)
        joint_2d, mesh_2d = x2d[:, :42], x2d[:, 42:]
        joint_3d, mesh_3d = x3d[:, :21, :], x3d[:, 21:, :]

        scale = param[:,0]
        trans = param[:, 1:3]
        #test
        #img = np.transpose(x[0].cpu().numpy()*255, axes=(1,2,0))
        # show_line_on_img(img, gt_2d[0].cpu().numpy(), '/home/lyf2/dataset/3dhand/dataset/pts_on_img4.png')
        # show_pts_on_img(img, self.convert_vec_to_2d(x2d)[0].detach().cpu().numpy(), '/home/lyf2/dataset/3dhand/dataset/mesh_on_img5.png')
        # show_line_on_img(img, self.convert_vec_to_2d(joint_2d)[0].detach().cpu().numpy(), '/home/lyf2/dataset/3dhand/dataset/pts_on_img5.png')
        # show_mask_on_img(img, mask[0].detach().cpu().numpy())
        #show_3dmesh(x3d[0])

        self.loss_2d = self.comupte_2d_joint_loss(joint_2d, gt_2d)
        if '3d_norm' in self.weight.values() or '3d_norm_no_detach' in self.weight.values():
            self.loss_3d = self.compute_3d_joint_loss_norm(joint_3d, gt_3d, valid[:, 0],
                                                           detach='3d_norm_no_detach' not in self.weight.values())
        else:
            self.loss_3d = self.compute_3d_joint_loss(joint_3d, gt_3d, valid[:, 0], scale)
        self.loss_mask    = self.compute_mask_loss(mesh_2d, mask, valid[:, 1])
        self.loss_reg     = self.compute_param_reg_loss(param)

        w = self.weight
        self.train_loss = w.w_2d * self.loss_2d + \
                        w.w_3d * self.loss_3d + \
                        w.w_mask * self.loss_mask + \
                        w.w_reg * self.loss_reg

        self.train_loss.backward()
        self.encoder_opt.step()

    @torch.no_grad()
    def sample_train(self, x, gt_2d, gt_3d, mask, valid):
        assert not self.ispretrain, "the method can be only used in train mode"

        x, valid = torch.detach(x), torch.detach(valid)
        gt_2d, gt_3d, mask = torch.detach(gt_2d), torch.detach(gt_3d), torch.detach(mask)
        # forward
        x2d, x3d, param = self.model(x)
        scale = param[:, 0]

        joint_2d, mesh_2d = x2d[:, :42], x2d[:, 42 :]
        joint_3d, mesh_3d = x3d[:, :21, :], x3d[:, 21 :, :]
        self.loss_2d = self.comupte_2d_joint_loss(joint_2d, gt_2d)
        if '3d_norm' in self.weight.values() or '3d_norm_no_detach' in self.weight.values() :
            self.loss_3d = self.compute_3d_joint_loss_norm(joint_3d, gt_3d, valid[:, 0],
                                                           detach='3d_norm_no_detach' not in self.weight.values())
        else:
            self.loss_3d = self.compute_3d_joint_loss(joint_3d, gt_3d, valid[:, 0], scale)
        self.loss_mask = self.compute_mask_loss(mesh_2d, mask, valid[:, 1])
        self.loss_reg = self.compute_param_reg_loss(param)

        w = self.weight
        self.train_loss = w.w_2d * self.loss_2d + \
                          w.w_3d * self.loss_3d + \
                          w.w_mask * self.loss_mask + \
                          w.w_reg * self.loss_reg

        losses = np.array([self.train_loss.cpu().numpy(),
                           self.loss_2d.cpu().numpy(),
                           self.loss_3d.cpu().numpy(),
                           self.loss_mask.cpu().numpy(),
                           self.loss_reg.cpu().numpy()
                           ])
        return [joint_2d, mesh_2d, joint_3d, mesh_3d], losses

    def encoder_pretrain_update(self, x, gt_vec):
        assert self.ispretrain, "the method can be only used in pretrain mode"

        self.encoder_opt.zero_grad()
        # encode
        _, _, param_encoded = self.model(x)
        # get groundtruth
        assert gt_vec.shape == param_encoded.shape   # (bs,22)
        # loss
        param_gt = torch.detach(gt_vec).float()

        self.loss_s    = self.param_recon_criterion(param_encoded[:,0], param_gt[:,0])
        self.loss_t    = self.param_recon_criterion(param_encoded[:,1:3], param_gt[:,1:3])
        self.loss_r    = self.param_recon_criterion(param_encoded[:,3:6], param_gt[:,3:6])
        self.loss_pose = self.param_recon_criterion(param_encoded[:,6:12], param_gt[:, 6:12])
        self.loss_beta = self.param_recon_criterion(param_encoded[:,12:], param_gt[:, 12:])

        w = self.weight
        self.vec_rec_loss = w.w_L1_pose * self.loss_pose + \
                            w.w_L1_beta * self.loss_beta + \
                            w.w_L1_r * self.loss_r + \
                            w.w_L1_t * self.loss_t + \
                            w.w_L1_s * self.loss_s

        self.vec_rec_loss.backward()
        self.encoder_opt.step()


    @torch.no_grad()
    def sample_pretrain(self, x, gt_vec):
        assert self.ispretrain, "the method can be only used in pretrain mode"
        # encode
        _, _, param_encoded = self.model(x)
        # get groundtruth
        assert gt_vec.shape == param_encoded.shape   # (bs,22)
        # loss
        param_gt = torch.detach(gt_vec)
        self.loss_s    = self.param_recon_criterion(param_encoded[:,0], param_gt[:,0])
        self.loss_t    = self.param_recon_criterion(param_encoded[:,1:3], param_gt[:,1:3])
        self.loss_r    = self.param_recon_criterion(param_encoded[:,3:6], param_gt[:,3:6])
        self.loss_pose = self.param_recon_criterion(param_encoded[:,6:12], param_gt[:, 6:12])
        self.loss_beta = self.param_recon_criterion(param_encoded[:,12:], param_gt[:, 12:])


        w = self.weight
        self.vec_rec_loss = w.w_L1_pose * self.loss_pose + \
                            w.w_L1_beta * self.loss_beta + \
                            w.w_L1_r * self.loss_r + \
                            w.w_L1_t * self.loss_t + \
                            w.w_L1_s * self.loss_s

        losses = np.array([self.vec_rec_loss.cpu().numpy(),
                           self.loss_pose.cpu().numpy(),
                           self.loss_beta.cpu().numpy(),
                           self.loss_r.cpu().numpy(),
                           self.loss_t.cpu().numpy(),
                           self.loss_s.cpu().numpy()
                           ])

        return losses

    def set_lr(self, lr):
        for param_group in self.encoder_opt.param_groups :
            param_group['lr'] = lr


    def update_lr(self):
        if self.encoder_scheduler is not None:
            self.encoder_scheduler.step()

    def print_losses(self):
        print("lr is %.7f" % self.encoder_opt.state_dict()['param_groups'][0]['lr'])
        if self.ispretrain:
            print(("total loss: %.4f | " + "pose loss: %.4f | " + "beta loss: %.4f | " +
                  "rotation loss: %.4f | translation loss: %.4f | scale loss : %.4f")
                    %(self.vec_rec_loss, self.loss_pose, self.loss_beta, self.loss_r, self.loss_t, self.loss_s))
        else:
            print(("total loss: %.4f | " + "2d loss: %.4f | " + "3d loss: %.6f | " +
                  "mask loss: %.4f | reg loss: %.4f")
                  %(self.train_loss, self.loss_2d, self.loss_3d, self.loss_mask, self.loss_reg)
    )

    def save(self, snapshot_dir, iterations):
        if self.ispretrain:
            model_pth = os.path.join(snapshot_dir, 'pretrained-model-%d_%08d.pth'
                                 %(self.input_option, iterations+1))
            opt_pth = os.path.join(snapshot_dir, 'optimizer.pth')
        else:
            model_pth = os.path.join(snapshot_dir, 'model-%d_%08d.pth'
                                     % (self.input_option, iterations + 1))
            opt_pth = os.path.join(snapshot_dir, 'optimizer.pth')

        torch.save(self.model.state_dict(), model_pth)
        torch.save(self.encoder_opt.state_dict(), opt_pth)

    def resume(self, checkpoint_dir, params, version = None):

        model_name = get_model_list(checkpoint_dir, 'model', version)

        state_dict = torch.load(model_name)
        self.model.load_state_dict(state_dict)
        iterations = int(model_name[-12:-4])
        if version is not None: assert version == iterations

        try:
            state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pth'))
            self.encoder_opt.load_state_dict(state_dict)
        except IOError:
            print 'use a new optimizer'

        # reinitialize scheduler
        self.encoder_scheduler = get_scheduler(self.encoder_opt, params, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def load_model(self, model_pth):
        state_dict = torch.load(model_pth)
        self.model.load_state_dict(state_dict)
        return 0