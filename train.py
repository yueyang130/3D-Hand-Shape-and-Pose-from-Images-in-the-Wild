# coding=utf-8
from trainer import EncoderTrainer
import torch
import argparse
import torch.backends.cudnn as cudnn
import os
import tensorboardX
import shutil
import utils
from torch.utils import data
from test_pretrain_model import sample
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
                    help='Path to the config file')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--version', type=int, default=None, help='The iteraiton of the model that you want to resume from')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu_id')
opts = parser.parse_args()

cudnn.benchmark = True  # the code can accelerate the training usually
torch.cuda.set_device(opts.gpu_id)

# Load experiment setting
config = utils.get_config(opts.config)
max_iter = config['max_iter']

# setup model and data loader
trainer = EncoderTrainer(config, ispretrain=False)
trainer.cuda()
trainloader, testloader = utils.get_data_loader(config, isPretrain=False)

# setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
model_dir = os.path.join(config['output_pth'], model_name)
log_dir, checkpoint_dir, image_dir, test_dir = utils.prepare_folder_strcutre(model_dir)
train_writer = tensorboardX.SummaryWriter(log_dir)


# start train
if opts.resume:
    iterations = trainer.resume(checkpoint_dir, config, version=opts.version)
    loss_log = utils.resume_loss_log(test_dir, iterations)
else:
    loss_log = [[], [], []]
    iterations = 0

while True:
    for images, gt_2d, gt_3d, mask, valid_3d  in trainloader:
        # detach returns a new Variable whose req_grd is False
        images = images.cuda().detach()
        gt_2d  = gt_2d.cuda().detach()
        gt_3d  = gt_3d.cuda().detach()
        mask   = mask.cuda().detach()
        valid_3d = valid_3d.cuda().detach()

        #with utils.Timer("Elapsed time in update: %f"):
        trainer.encoder_update(images, gt_2d, gt_3d, mask, valid_3d)
        trainer.update_lr()
        torch.cuda.synchronize()  # the code synchronize gpu and cpu process , ensuring the accuracy of time measure


        # Dump training stats in log file
        if (iterations + 1) % config['print_loss_iter'] == 0 :
            print "Iteration: %08d/%08d, " % (iterations + 1, max_iter),
            trainer.print_losses()
            utils.write_loss(iterations, trainer, train_writer)

        # test
        if (iterations + 1) % config['test_iter'] == 0:
            new_trainloader, new_testLoader = utils.get_data_loader(config, isPretrain=False)
            train_num, train_loss = sample(trainer, new_trainloader, config['batch_size'], config['test_num'])
            test_num, test_loss = sample(trainer, new_testLoader, config['batch_size'], config['test_num'])
            #print 'test on %d iamges in trainset, %d iamges in testset'%(train_num, test_num)
            loss_log[0].append(iterations+1)
            loss_log[1].append(train_loss)
            loss_log[2].append(test_loss)
            with open(os.path.join(test_dir, 'iter_%d_loss.pickle'%(iterations + 1)), 'w') as fo:
                pickle.dump(loss_log, fo)


        # save the test result
        if (iterations + 1) % config['show_iter'] == 0 :
            losses = ['total_loss' ,'2d_loss', '3d_loss', 'mask_loss', 'reg_loss']
            for i, loss in enumerate(losses):
                train_loss = np.array(loss_log[1])
                test_loss = np.array(loss_log[2])
                if i == 0:
                   plt.subplot(311)
                   plt.plot(loss_log[0], train_loss[:,i].tolist(), '.-', label='train_loss')
                   plt.plot(loss_log[0], test_loss[:,i].tolist(), '.-', label='test_loss')
                else:
                    plt.subplot(322+i)
                    plt.plot(loss_log[0], train_loss[:,i].tolist(), '.-', label='train_loss')
                    plt.plot(loss_log[0], test_loss[:,i].tolist(), '.-', label='test_loss')
                plt.xlabel('iterations')
                plt.ylabel(loss)
                plt.legend()  # 加了这一句才显示label
            plt.savefig(os.path.join(test_dir, 'iter_%d_loss.png'%(iterations + 1)))
            plt.close()

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_dir, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')