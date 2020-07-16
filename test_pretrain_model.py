# The script is used to compare the loss value
# between trainset (4000 synthetic images) and testset (1000 synthetic images)

from trainer import EncoderTrainer
import torch
import numpy as np
import argparse
import torch.backends.cudnn as cudnn
import os
import tensorboardX
import utils
import sys

def pre_sample(pretrainer, dataloader, batch_size, test_num):
    cnt_batch = 0
    total_losses = np.zeros(6)
    msg = []
    with torch.no_grad() :
        for imgs, gt_vecs in dataloader :
            imgs = imgs.cuda().detach()
            gt_vecs = gt_vecs.cuda().detach()
            losses = pretrainer.sample_pretrain(imgs, gt_vecs)
            total_losses += losses
            cnt_batch += 1
            num_data = cnt_batch * batch_size
            if num_data >= test_num : break
    return num_data, total_losses/cnt_batch

def sample(trainer, dataloader, batch_size, test_num):
    cnt_batch = 0
    total_losses = np.zeros(5)
    with torch.no_grad() :
        for imgs, gt_2d, gt_3d, mask, valid_3d in dataloader :
            imgs = imgs.cuda().detach()
            gt_2d = gt_2d.cuda().detach()
            gt_3d = gt_3d.cuda().detach()
            mask = mask.cuda().detach()
            valid_3d = valid_3d.cuda().detach()
            results, losses = trainer.sample_train(imgs, gt_2d, gt_3d, mask, valid_3d)
            total_losses += losses
            cnt_batch += 1
            num_data = cnt_batch * batch_size
            if num_data >= test_num : break
    return num_data, total_losses/cnt_batch

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='Path to the config file')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu_id')
    opts = parser.parse_args()

    cudnn.benchmark = True  # the code can accelerate the training usually
    torch.cuda.set_device(opts.gpu_id)

    # Load experiment setting
    config = utils.get_config(opts.config)
    max_iter = config['max_iter']

    # setup model and data loader
    pretrainer = EncoderTrainer(config, ispretrain=True)
    pretrainer.cuda()
    pretrainer.eval()   # close BN and droplet
    trainloader, testloader = utils.get_data_loader(config, isPretrain=True)

    # setup logger and output folders
    model_name = os.path.splitext(os.path.basename(opts.config))[0]
    model_dir = os.path.join(config['output_pth'], model_name)
    log_dir, checkpoint_dir, image_dir, test_dir = utils.prepare_folder_strcutre(model_dir)

    # start test
    iterations = pretrainer.resume(checkpoint_dir, config)
    test_log_pth = os.path.join(test_dir, '%d.txt'%iterations)

    num_data, avg_loss = pre_sample(pretrainer, trainloader, config['batch_size'])
    msg = ['The train dataset has %d input data, '%num_data  +
        'For pretrain-model%d-%d,the avarage vec_rec_loss is %f\n'%
               (config['input_option'], iterations, avg_loss)]

    num_data, avg_loss = pre_sample(pretrainer, testloader, config['batch_size'])
    msg.append('The test dataset has %d input data, '%num_data  +
        'For pretrain-model%d-%d,the avarage vec_rec_loss is %f\n'%
               (config['input_option'], iterations, avg_loss))

    for m in msg:
        print m

    # write log
    with open(test_log_pth, 'w') as f:
        f.writelines(msg)

if __name__ == '__main__':
    main()