import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from model import resnet34_Mano
from utils import get_scheduler, \
    weights_init, get_criterion, get_model_list
import os


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
        self.model = resnet34_Mano(ispretrain=ispretrain, input_option=params['input_option'])

        # setup the optimizer
        lr = params.lr
        beta1 = params.beta1
        beta2 = params.beta2
        p_view = self.model.state_dict()
        self.encoder_opt = torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad],
                                    lr = lr, betas=(beta1, beta2), weight_decay=params.weight_decay)
        self.encoder_scheduler = get_scheduler(self.encoder_opt, params)

        # set loss fn
        if self.ispretrain:
            self.get_param_recon_criterion = get_criterion(params['pretrain_loss_fn'])
        else :
            self.joint_2d_criterion = get_criterion('L1')
            self.joint_3d_criterion = get_criterion('L2')


        # Network weight initialization
        self.model.apply(weights_init(params.init))

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
        ret = torch.mean(torch.abs(valid_joint_rec - valid_joint_gt))
        return ret

    def compute_3d_joint_loss(self, joint_rec, joint_gt, valid):
        # joint_gt: [bs, 21, 3]
        #ret = torch.sqrt(torch.sum((joint_rec - joint_gt)**2))
        ret = (joint_rec - joint_gt)**2
        bs = valid.shape[0]
        valid_index = torch.arange(bs)[valid == 1]
        ret = ret[valid_index]   # valid_num * 21 * 3
        ret = torch.mean(ret)
        return ret


    def compute_mask_loss(self, mesh2d, mask, valid):
        # for debug
        # mesh2d = mesh2d.cpu()
        # mask = mask.cpu()
        # valid = valid.cpu()

        mesh2d = self.convert_vec_to_2d(mesh2d)  # [bs, 778, 2]
        mesh2d = mesh2d.type(torch.int64)
        #mesh2d = mesh2d.round()


        # mesh may out of image
        size = mask.shape[1]
        mesh2d[mesh2d < 0] = 0
        mesh2d[mesh2d >= size] = size - 1

        batch_size = mesh2d.shape[0]
        index0 = torch.arange(batch_size).reshape((batch_size, 1)).repeat(1, 778).type(torch.int64) # [bs, 778]
        # u means width; v means height
        #index1 = mesh2d[:,:,0]
        #index2 = mesh2d[:,:,1]
        index1 = mesh2d[:,:,1]
        index2 = mesh2d[:,:,0]
        ret = mask[index0, index1, index2]  # [bs, 778]

        bs = valid.shape[0]
        valid_index = torch.arange(bs)[valid == 1]
        ret = ret[valid_index]
        ret = torch.tensor(1.) - torch.mean(ret.type(torch.float32))
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
        self.loss_2d = self.comupte_2d_joint_loss(joint_2d, gt_2d)
        self.loss_3d = self.compute_3d_joint_loss(joint_3d, gt_3d, valid[:, 0])
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
        joint_2d, mesh_2d = x2d[:, :42], x2d[:, 42 :]
        joint_3d, mesh_3d = x3d[:, :21, :], x3d[:, 21 :, :]
        self.loss_2d = self.comupte_2d_joint_loss(joint_2d, gt_2d)
        self.loss_3d = self.compute_3d_joint_loss(joint_3d, gt_3d, valid[:, 0])
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
        param_encoded = self.model(x)
        # get groundtruth
        assert gt_vec.shape == param_encoded.shape
        # loss
        param_gt = torch.detach(gt_vec)
        self.vec_rec_loss = self.get_param_recon_criterion(param_encoded, param_gt)
        self.vec_rec_loss.backward()
        self.encoder_opt.step()


    @torch.no_grad()
    def sample_pretrain(self, x, gt_vec):
        assert self.ispretrain, "the method can be only used in pretrain mode"
        # encode
        param_encoded = self.model(x)
        # get groundtruth
        assert gt_vec.shape == param_encoded.shape
        # loss
        param_gt = torch.detach(gt_vec)
        self.vec_rec_loss = self.get_param_recon_criterion(param_encoded, param_gt)
        return self.vec_rec_loss


    def update_lr(self):
        if self.encoder_scheduler is not None:
            self.encoder_scheduler.step()

    def print_losses(self):
        if self.ispretrain:
            print("pretrain rec_param loss: %.4f" % (__ee__(self.vec_rec_loss)))
        else:
            print(("total loss: %.4f | " + "2d loss: %.4f | " + "3d loss: %.4f | " +
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

        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pth'))
        self.encoder_opt.load_state_dict(state_dict)

        # reinitialize scheduler
        self.encoder_scheduler = get_scheduler(self.encoder_opt, params, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations