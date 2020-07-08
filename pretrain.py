from trainer import EncoderTrainer
import torch
import argparse
import torch.backends.cudnn as cudnn
import os
import tensorboardX
import shutil
import utils
from torch.utils import data

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
trainloader, testloader = utils.get_data_loader(config, isPretrain=True)

# setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
model_dir = os.path.join(config['output_pth'], model_name)
log_dir, checkpoint_dir, image_dir = utils.prepare_folder_strcutre(model_dir)
train_writer = tensorboardX.SummaryWritter(log_dir)

# start train
if opts.resume:
    iterations = pretrainer.resume(checkpoint_dir, config)
else:
    iterations = 0

while True:
    for images, gt_vecs in trainloader:
        # detach returns a new Variable whose req_grd is False
        images = images.cuda().detach()
        gt_vecs = gt_vecs.cuda().detach()

        with utils.Timer("Elapsed time in update: %f"):
            pretrainer.update_lr()
            pretrainer.encoder_pretrain_update(images, gt_vecs)
            torch.cuda.synchronize()  # the code synchronize gpu and cpu process , ensuring the accuracy of time measure

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            pretrainer.print_losses()
            utils.write_loss(iterations, pretrainer, train_writer)

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            pretrainer.save(checkpoint_dir, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')