# The script make the following setps:
# 1. crop images into hand box  (note: label also changes)
# 2. flip the left hand into right (note: label also change)
# 3. resize to 320 * 320
# 4. merge all labels into one
# 5. split into trainset and testset

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.kps import KeypointsOnImage, Keypoint
import scipy.misc as misc
import os
import json
import pickle
import shutil
from prepare_background import get_img_path_list
import numpy as np
from crop import get_crop_pos


def show_pts_on_img(image, pts,
    img_pth = '/home/lyf2/dataset/3dhand/dataset/pts_on_img.png'):

    kps = [Keypoint(x, y) for x,y,_ in pts]
    kpsoi = KeypointsOnImage(kps, shape=image.shape)
    misc.imsave(img_pth, kpsoi.draw_on_image(image, size = 7))

def crop(image, pts):
    #show_pts_on_img(image, pts)

    vs, ve, us, ue = get_crop_pos(image, pts)
    cropped_image = image[vs : ve + 1, us : ue + 1, :]

    #img_pth = '/home/lyf2/dataset/3dhand/dataset/cropped_img.png'
    #misc.imsave(img_pth, cropped_image)

    # In fact, pt[0] means width axis, and pt[1] means height axis
    cropped_pts = np.concatenate([pts[:,0:1] - us, pts[:,1:2] - vs, pts[:, 2:]], axis=1)

    #show_pts_on_img(cropped_image, cropped_pts)

    resz = iaa.Resize({'height' : 320, 'width' : 320}, interpolation='linear')

    resized_image, resized_pts = resz.augment(image=cropped_image, keypoints=[cropped_pts[:,:2]])
    resized_pts = resized_pts[0]
    resized_pts = np.concatenate([resized_pts, cropped_pts[:,2:]], axis=1)

    #show_pts_on_img(resized_image, resized_pts)
    resized_pts = resized_pts.round(3)
    return resized_image, resized_pts

# def flip(image,label):
#     return image, label
#     #return flipped_img, flipped_label

def get_image_dict(cnt, pts):
    image_dict = {
                    'index': cnt,
                    'img_paths' : 'image/08%d.png' % cnt,
                    'joint_self' : pts.tolist()
                  }
    return image_dict


def process_PANOPTIC(outpath, infor_dict, cnt, num = float('inf')):
    """
    The panoptic dataset only contains labels of right hand.
    And it has a intergrated label with the extension name of 'json'.
    """
    root = "/home/lyf2/dataset/3dhand/panoptic/hand143_panopticdb/"
    label_pth = "/home/lyf2/dataset/3dhand/panoptic/hand143_panopticdb/hands_v143_14817.json"

    with open(label_pth, 'r') as fo:
        dat_all = json.load(fo)['root']


    # dat_all is a list, dat is a dict
    for ii, dat in enumerate(dat_all):
        if ii >= num: break
        pts = np.array(dat['joint_self'])
        #assert pts.shape[0] == 21
        img = misc.imread(root + dat['img_paths'])
        #img = np.array(img)
        # crop
        new_img, new_pts = crop(img, pts)

        img_path = outpath + '%08d.png'%cnt
        misc.imsave(img_path, new_img)
        pt_on_img_path = outpath + 'pt_on_img_%08d.png'%cnt
        show_pts_on_img(new_img, new_pts, pt_on_img_path)
        infor_dict['image'].append(get_image_dict(cnt, new_pts))

        cnt += 1
    return cnt






def process_MPII(outpath, infor_dict, num = float('inf')):
    pass

def process_stereo(outpath, infor_dict, num = float('inf')):
    pass


if __name__ == '__main__':
    outpath = "/home/lyf2/dataset/3dhand/dataset/image/"
    newlabel_path = "/home/lyf2/dataset/3dhand/dataset/joints.json"

    infor_dict = {'image': []}
    num_list = []
    cnt = 0
    # test the process use only 3 images
    cnt = process_PANOPTIC(outpath, infor_dict, cnt, 3)

    infor_dict['img_num'] = cnt
    fjson = open(newlabel_path, 'w')
    json.dump(infor_dict, fjson)
    fjson.close()