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

    def comupte_2d_joint_loss(self, joint_rec, joint_gt):
        pass
        #TODO

    def compute_3d_joint_loss(selfself, joint_rec, joint_gt):
        pass
        #TODO

    def compute_mask_loss(self, mask):
        ret = torch.tensor(1.) - torch.mean(mask)
        return ret

    def compute_param_reg_loss(self, vec):
        assert vec.dim[0] == 22
        beta_weight = 10**4
        beta = vec[-10:]
        theta = vec[-16:-10]
        ret = torch.mean(theta**2) + beta_weight * torch.mean(beta**2)
        return ret

    def encoder_update(self, x, gt_2d, gt_3d, mask):
        assert not self.ispretrain, "the method can be only used in train mode"

        self.encoder_opt.zero_grad()
        #TODO

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
            #TODO
            pass

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

    def resume(self, checkpoint_dir, params):
        last_model_name = get_model_list(checkpoint_dir, 'model')
        state_dict = torch.load(last_model_name)
        self.model.load_state_dict(state_dict)
        iterations = int(last_model_name[-12:-4])

        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pth'))
        self.encoder_opt.load_state_dict(state_dict)

        # reinitialize scheduler
        self.encoder_scheduler = get_scheduler(self.encoder_opt, params, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations