from torch.optim import lr_scheduler
from torch.nn import init
import torch.nn as nn
import math
import os
import easydict
import time
import yaml
from torch.utils.data import DataLoader
from datasets import HandPretrainSet, HandTrainSet
import json
import numpy as np
import pickle

def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun

def get_criterion(loss_type):
    if loss_type == 'L1':
        return nn.L1Loss('mean')
    elif loss_type == 'L2':
        return nn.MSELoss('mean')
    else:
        assert 0, "Unsupported loss fn"

# Get model list for resume
def get_model_list(dirname, key, version = None):
    if os.path.exists(dirname) is False:
        return None

    if version is None:
        gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                      os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
        if gen_models is None :
            return None
        gen_models.sort()
        last_model_name = gen_models[-1]
        return last_model_name
    else:
        for f in os.listdir(dirname):
            if os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f and int(f[-12:-4]) == version:
                return  os.path.join(dirname, f)
        return None



def get_config(config):
    with open(config, 'r') as stream:
        return easydict.EasyDict(yaml.load(stream))


def get_data_loader(config, isPretrain):
    batch_size = config['batch_size']
    new_size = config['new_size']
    num_workers = config['num_workers']

    if isPretrain:
        trainDataset = HandPretrainSet(os.path.join(config['data_root'], 'train'))
        testDataset  = HandPretrainSet(os.path.join(config['data_root'], 'test'))
    else:
        trainDataset = HandTrainSet(os.path.join(config['data_root'], 'train'))
        testDataset  = HandTrainSet(os.path.join(config['data_root'], 'test'))
    trainloader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testLoader  = DataLoader(testDataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return trainloader, testLoader

# outpur dir structure
# output_pth
#         |- pretrain
#         |          |- model0
#         |          |- model1
#         |- train
#                 |-model0
#                 |-model1
#                        |-logs
#                        |-checkpoints
#                        |-images
def prepare_folder_strcutre(model_dir):
    log_directory = os.path.join(model_dir, 'logs')
    if not os.path.exists(log_directory) :
        print("Creating directory: {}".format(log_directory))
        os.makedirs(log_directory)

    image_directory = os.path.join(model_dir, 'images')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)

    checkpoint_directory = os.path.join(model_dir, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)

    test_directory = os.path.join(model_dir, 'test')
    if not os.path.exists(test_directory):
        print("Creating directory: {}".format(test_directory))
        os.makedirs(test_directory)



    return log_directory, checkpoint_directory,\
           image_directory, test_directory

class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))

def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer) \
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and ('loss' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)



def parse_labelfile(label_mode, label_ext, label_pth, joint_keyname=None, n_image=None) :
    """
    label_model: 1 means every image has a label file; 0 means all iamges has a common file.
    """
    joints = []
    if label_mode :
        if label_ext == '.json' :
            for ii in range(n_image) :
                with open(os.path.join(label_pth, str(ii) + '.json'), 'r') as fd :
                    dat = json.load(fd)
                    joint = np.array(dat[joint_keyname])
                    joints.append(joint)
        elif label_ext == '.mat' :
            pass
        else :
            assert 0, "Unsupported file type of label"
    else :
        if label_ext == '.pickle':
            with open(label_pth, 'r') as fd :
                joints = np.array(pickle.load(fd))
        elif label_ext == '.json':
            with open(label_pth, 'r') as fd :
                dat = json.load(fd)
                joints = dat if joint_keyname is None else np.array(dat[joint_keyname])
        else :
            assert 0, "Unsupported file type of label"


    return joints

def resume_loss_log(test_dir, version):
    if version == 0: return [[], [], []]
    with open(os.path.join(test_dir, 'iter_%d_loss.pickle'%version), 'r') as fo:
        return pickle.load(fo)

