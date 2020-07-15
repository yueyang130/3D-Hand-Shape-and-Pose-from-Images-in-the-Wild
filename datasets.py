import os
from PIL import Image
import torch
from torch.utils import data
from scripts.prepare_background import get_img_path_list
from torchvision.transforms import ToTensor, Compose
from utils.transform import Scale
import pickle
import json
import imgaug as ia
import imgaug.augmenters as iaa
from scripts.make_dataset import show_pts_on_img
from scripts.segment import show_mask_on_img
import random
import numpy as np

def getItem(data_dir, index, img_transform):
    img = Image.open(os.path.join(data_dir, '%d.png' % index)).convert('RGB')
    #img_view = ToTensor(img)
    imgs = [img_transform(img)]
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
        # TODO: uncomment the below code after installing PyOpenPose
        # if len(self.imgs) % 8 != 0 :
        #     assert 0, "the image number in dir %s is not the multiple of 8 (1 image and 7 heat maps)"

    def __len__(self) :
        #TODO
        #return len(self.imgs)/8
        return len(self.imgs)

    def __getitem__(self, index) :
        self.img_transform = Compose([
            Scale((256, 256), Image.BILINEAR),
            ToTensor()])
        input_img = getItem(self.data_dir, index, self.img_transform)
        # get gt params
        vec = self.vectors[index]
        vec = vec[1:]  # delete the first element 1 (valid bit)
        return input_img, vec




class HandTrainSet(data.Dataset):
    def __init__(self, root):
        self.img_dir = os.path.join(root, 'image')
        self.mask_dir = os.path.join(root, 'mask')
        label_pth = os.path.join(root, 'joints.json')
        with open(label_pth, 'r') as fo:
            self.anno = json.load(fo)['image']


        self.imgs = get_img_path_list(self.img_dir)
        if len(self.imgs) == 0 :
            assert 0, "dir %s includes no image!"%(root)

        self.count = [0] * self.__len__()
        # if len(self.imgs) % 8 != 0 :
        #     assert 0, "the image number in dir %s is not the multiple of 8 (1 image and 7 heat maps)"

    def __len__(self) :
        #TODO
        #return len(self.imgs)/8
        return len(self.imgs)

    def data_augmentation(self, img, mask, pts) :
        """
            It's no problem to scale the pretrain set images because the images are originally 320*320.
            However, when scale the trainset images, the corresponding labels should be changed simultaniously
            Additionally, flip is forbidden because we are using right-hand model.
            Is rotate permitted?
        """
        start_v = random.randint(0, 10)
        start_u = random.randint(0, 10)
        crop_len = 15
        end_v = img.shape[0] + start_v - crop_len
        end_u = img.shape[1] + start_u - crop_len
        cropped_img = img[start_v: end_v+1, start_u: end_u+1, : ]
        cropped_mask = mask[start_v: end_v+1, start_u: end_u+1]
        cropped_pts = np.concatenate([pts[:,0:1] - start_u, pts[:,1:2] - start_v, pts[:, 2:]], axis=1)

        resz = iaa.Resize({'height' : 256, 'width' : 256}, interpolation='linear')

        resized_img, resized_pts = resz.augment(image=cropped_img, keypoints=[cropped_pts[:, :2]])
        resized_mask = resz.augment(image=cropped_mask)

        resized_pts = resized_pts[0]
        resized_pts = np.concatenate([resized_pts, cropped_pts[:, 2 :]], axis=1)

        # guassian noise
        resized_img = resized_img + np.random.randn(256, 256, 3)
        resized_img[resized_img > 255] = 255
        resized_img[resized_img < 0] = 0

        # test
        # show_pts_on_img(resized_img, resized_pts)
        # show_mask_on_img(resized_img, resized_mask * 255)

        return resized_img, resized_mask, resized_pts

    # def __getitem__(self, index):
    #     #input_img = getItem(self.data_dir, index, self.img_transform)
    #
    #     try:
    #         img = Image.open(os.path.join(self.img_dir, '%08d.png' % index)).convert('RGB')
    #         mask = Image.open(os.path.join(self.mask_dir, '%08d.png' % index))
    #         joint_2d = np.array(self.anno[index]['2d_joint'])
    #         if '3d_joint' in self.anno[index].keys() :
    #             joint_3d = np.array(self.anno[index]['3d_joint'])
    #         else :
    #             joint_3d = np.zeros((21, 3))
    #
    #     except IOError:
    #         print('mask %08d not found'%index)
    #
    #
    #
    #     # data augmentaion and resize
    #     img, mask, joint_2d = self.data_augmentation(img, mask, joint_2d)
    #     # transform
    #     img = ToTensor()(img)
    #     return img, joint_2d, joint_3d, mask

    def __getitem__(self, index):

        self.count[index] += 1
        #input_img = getItem(self.data_dir, index, self.img_transform)
        valid = np.array([1, 1]) # the validility of 3d joint and mask
        img = Image.open(os.path.join(self.img_dir, '%08d.png' % index)).convert('RGB')
        img = np.asarray(img)
        try:
            mask = Image.open(os.path.join(self.mask_dir, '%08d.png' % index))
            mask = np.asarray(mask)
        except IOError:
            mask = np.zeros((256, 256), dtype=np.uint8)
            valid[1] = 0
            print('mask %08d not found'%index)

        joint_2d = np.array(self.anno[index]['2d_joint'])
        if '3d_joint' in self.anno[index].keys():
            joint_3d = np.array(self.anno[index]['3d_joint'])
        else:
            joint_3d = np.zeros((21,3))
            valid[0] = 0

        # data augmentaion and resize
        img, mask, joint_2d = self.data_augmentation(img, mask, joint_2d)
        # transform
        img = np.transpose(img/255., axes=(2,0,1)).astype(np.float32)
        joint_2d = joint_2d.astype(np.float32)
        joint_3d = joint_3d.astype(np.float32)

        return img, joint_2d, joint_3d, mask, valid














