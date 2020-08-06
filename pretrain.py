# coding=utf-8
from trainer import EncoderTrainer
import torch
import argparse
import torch.backends.cudnn as cudnn
import os
import tensorboardX
import utils
import sys
from test_pretrain_model import pre_sample
import matplotlib.pyplot as plt
import pickle
import numpy as np
import tester

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
                    help='Path to the config file')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--version', type=int, default=None, help='The iteraiton of the model that you want to resume from')
parser.add_argument('--gpu_id', type=str, default='0', help='gpu_id')
parser.add_argument("--lr", type=float,default=None)
opts = parser.parse_args()

cudnn.benchmark = True  # the code can accelerate the training usually
os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
#torch.cuda.set_device(opts.gpu_id)

# Load experiment setting
config = utils.get_config(opts.config)
max_iter = config['max_iter']

# setup model and data loader
pretrainer = EncoderTrainer(config, ispretrain=True)
pretrainer.cuda()
pretrainer.train()
trainloader, testloader = utils.get_data_loader(config, isPretrain=True)

# setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
model_dir = os.path.join(config['output_pth'], model_name)
log_dir, checkpoint_dir, image_dir, test_dir = utils.prepare_folder_strcutre(model_dir)
train_writer = tensorboardX.SummaryWriter(log_dir)
loss_log = [[], [], []]

# start train
if opts.resume:
    iterations = pretrainer.resume(checkpoint_dir, config, version=opts.version)
    loss_log = utils.resume_loss_log(test_dir, iterations)
else:
    iterations = 0

# set lr
if opts.lr is not None:
    pretrainer.set_lr(opts.lr)

while True:
    for images, gt_vecs in trainloader:
        # detach returns a new Variable whose req_grd is False
        images = images.cuda().detach()
        gt_vecs = gt_vecs.cuda().detach()

        #with utils.Timer("Elapsed time in update: %f"):
        pretrainer.encoder_pretrain_update(images, gt_vecs)
        pretrainer.update_lr()
        torch.cuda.synchronize()  # the code synchronize gpu and cpu process , ensuring the accuracy of time measure

        # Dump training stats in log file
        if (iterations + 1) % config['print_loss_iter'] == 0:
            print "Iteration: %08d/%08d, " % (iterations + 1, max_iter),
            pretrainer.print_losses()
            utils.write_loss(iterations, pretrainer, train_writer)

        # test
        if iterations == 0 or (iterations + 1) % config['test_iter'] == 0:
            with torch.no_grad():

                pretrainer.eval()
                img_iter_dir = os.path.join(image_dir, '%08d' % (iterations + 1))
                if not os.path.exists(img_iter_dir) :
                    os.makedirs(img_iter_dir)
                tester.test(config['input_option'], pretrainer.model, img_iter_dir)
                pretrainer.train()

                new_trainloader, new_testLoader = utils.get_data_loader(config, isPretrain=True)
                train_num, train_loss = pre_sample(pretrainer, new_trainloader, config['batch_size'], config['test_num'])
                test_num, test_loss = pre_sample(pretrainer, new_testLoader, config['batch_size'], config['test_num'])
                #print 'test on %d iamges in trainset, %d iamges in testset'%(train_num, test_num)
                loss_log[0].append(iterations+1)
                loss_log[1].append(train_loss)
                loss_log[2].append(test_loss)
                with open(os.path.join(test_dir, 'iter_%d_loss.pickle'%(iterations + 1)), 'w') as fo:
                    pickle.dump(loss_log, fo)


        # save the test result
        if iterations == 0 or (iterations + 1) % config['show_iter'] == 0 :

            losses = ['total_loss', 'pose_loss', 'beta_loss', 'r_loss', 't_loss','s_loss']
            # do not show iteration = 1
            iters = loss_log[0][1 :]
            train_loss = np.array(loss_log[1])[1 :]
            test_loss = np.array(loss_log[2])[1 :]
            for i, loss in enumerate(losses) :
                plt.subplot(611 + i)
                plt.plot(iters, train_loss[:, i].tolist(), ',-', label='train_loss')
                plt.plot(iters, test_loss[:, i].tolist(), ',-', label='test_loss')
                plt.xlabel('iterations')
                plt.ylabel(loss)
                plt.legend()  # 加了这一句才显示label
            plt.savefig(os.path.join(test_dir, 'iter_%d_loss.png' % (iterations + 1)))
            plt.close()

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            pretrainer.save(checkpoint_dir, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')