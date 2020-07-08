import os
from PIL import Image
import cv2
import torch
from torch.utils import data
from scripts.prepare_dataset.prepare_background import get_img_path_list
from torchvision.transforms import ToTensor, Compose
from transform import Scale
import pickle

def getItem(data_dir, index, img_transform):
    imgs = [img_transform(Image.open(os.path.join(data_dir, '%d.png' % index)).convert('RGB'))]
    # TODO: comment the heatmap temporarily due to the uninstallation of pyOpenPose
    # for i in xrange(7) :
    #     imgs.append(
    #         img_transform(Image.open(os.path.join(data_dir, '%d_%d.png' % (index, i))).convert('RGB')))
    imgs = torch.cat(imgs, dim=0)
    return imgs

class HandTestSet(data.Dataset):
    def __init__(self, root, img_transform=None):
        self.data_dir = root
        self.img_transform = img_transform
                                
    def __len__(self):
        return 3

    def __getitem__(self, index):
        return getItem(self.data_dir, index, self.img_transform)


class HandPretrainSet(data.Dataset) :
    def __init__(self, root) :
        self.data_dir = root
        gt_pth = os.path.join(root, 'gt.pickle')
        assert os.path.isfile(gt_pth)
        with open(gt_pth, 'r') as fo:
            self.vectors = pickle.load(fo)

        self.imgs = get_img_path_list(root)
        if len(self.imgs) == 0 :
            assert 0, "dir %s includes no image!"%(root)
        if len(self.imgs) % 8 != 0 :
            assert 0, "the image number in dir %s is not the multiple of 8 (1 image and 7 heat maps)"

    def __len__(self) :
        return len(self.imgs)/8

    def __getitem__(self, index) :
        self.img_transform = Compose([
            Scale((256, 256), Image.BILINEAR),
            ToTensor()])
        input_img = getItem(self.data_dir, index, self.img_transform)
        # get gt params
        vec = self.vectors[index]
        vec = vec[1:]
        return input_img, vec


class HandTrainSet(data.Dataset):
    def __init__(self, root):
        self.data_dir = root
        self.imgs = get_img_path_list(root)
        if len(self.imgs) == 0 :
            assert 0, "dir %s includes no image!"%(root)
        if len(self.imgs) % 8 != 0 :
            assert 0, "the image number in dir %s is not the multiple of 8 (1 image and 7 heat maps)"

    def __getitem__(self, index):
        #TODO: images of left hand are flipped horizontally
        self.img_transform = Compose([
            Scale((256, 256), Image.BILINEAR),
            ToTensor()])
        input_img = getItem(self.data_dir, index, self.img_transform)
        #TODO: get 2d & 3d joint annotations, masks










