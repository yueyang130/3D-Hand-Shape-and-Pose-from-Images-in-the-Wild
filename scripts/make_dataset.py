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
import scipy.io as scio
import shutil
from prepare_background import get_img_path_list
import numpy as np
from crop import get_crop_pos
from prepare_dataset.mm2px import JointTransfomer

def show_pts_on_img(image, pts,
    img_pth = '/home/lyf2/dataset/3dhand/dataset/pts_on_img.png'):

    kps = [Keypoint(x, y) for x,y,_ in pts]
    kpsoi = KeypointsOnImage(kps, shape=image.shape)
    misc.imsave(img_pth, kpsoi.draw_on_image(image, size = 7))

def crop(image, pts, pts_3d = None):
    #TODO: add 3d
    #show_pts_on_img(image, pts)

    vs, ve, us, ue = get_crop_pos(image, pts)
    cropped_image = image[vs : ve + 1, us : ue + 1, :]

    #img_pth = '/home/lyf2/dataset/3dhand/dataset/cropped_img.png'
    #misc.imsave(img_pth, cropped_image)

    # In fact, pt[0] means width axis, and pt[1] means height axis
    cropped_pts = np.concatenate([pts[:,0:1] - us, pts[:,1:2] - vs, pts[:, 2:]], axis=1)

    #show_pts_on_img(cropped_image, cropped_pts)

    resz = iaa.Resize({'height' : 320, 'width' : 320}, interpolation='linear')

    resized_image, resized_pts = resz.augment(image=cropped_image, keypoints=[cropped_pts[:, :2]])
    resized_pts = resized_pts[0]
    resized_pts = np.concatenate([resized_pts, cropped_pts[:,2:]], axis=1)
    #show_pts_on_img(resized_image, resized_pts)

    resized_pts = resized_pts.round(3)
    return resized_image, resized_pts

def flip(image, pts, pts_3d = None):
    # TODO: test whether augment can flip 3d, too.
    myFilp = iaa.Fliplr(p = 1.0)
    flipped_img, flipped_pts = myFilp.augment(image=image, keypoints=[pts[:, :2]])
    flipped_pts = flipped_pts[0]

    if pts_3d is not None:
        flipped_pts_3d = myFilp.augment(keypoints=pts_3d)
        return flipped_img, flipped_pts, flipped_pts_3d
    else:
        return flipped_img, flipped_pts


def get_image_dict(cnt, pts_2d, pts_3d):
    image_dict = {
                    'index': cnt,
                    'img_paths' : 'image/08%d.png' % cnt,
                  }
    if pts_2d is not None:
        image_dict['2d_joint'] = pts_2d.tolist()
    if pts_3d is not None:
        image_dict['3d_joint'] = pts_3d.tolist()
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
        # crop and resize
        new_img, new_pts = crop(img, pts)

        img_path = outpath + '%08d.png'%cnt
        misc.imsave(img_path, new_img)
        #pt_on_img_path = outpath + 'pt_on_img_%08d.png'%cnt
        #show_pts_on_img(new_img, new_pts, pt_on_img_path)
        infor_dict['image'].append(get_image_dict(cnt, new_pts, pts_3d=None))

        cnt += 1
    return cnt

def process_MPII(outpath, infor_dict, cnt, num = float('inf'), trainset = True):
    """
        The MPII dataset also contains left hands, which should be flipped to right one.
        And it has (per image) a label with the extension name of 'json'.
    """
    root = '/home/lyf2/dataset/3dhand/MPII/' + ('manual_train/' if trainset else 'manual_test/')
    json_list = get_img_path_list(root, key='.json')
    for ii, json_pth in enumerate(json_list):
        if ii >= num : break
        dat = json.load(json_pth)
        pts = dat['hand_pts']
        is_left = dat['is_left']
        img_name = os.path.splitext(os.path.split(json_pth)[1])[0]
        img = misc.imread(root + img_name + '.jpg')


        # crop and resize
        new_img, new_pts = crop(img, pts)
        # flip
        # TODO: test
        if is_left:
            new_img, new_pts = flip(new_img, new_pts)

        img_pth = outpath + '%08d.png'%cnt
        misc.imsave(img_pth, new_img)
        infor_dict['image'].append(get_image_dict(cnt, new_pts, pts_3d=None))

        cnt += 1

    return cnt



def process_stereo(outpath, infor_dict, cnt, num = float('inf')):
    """
        The stereo dataset only contains left hands, which should be flipped to right one.
        And it has (per dir) a label with the extension name of 'mat'.
    """
    sequences = [
        'B2Counting', 'B2Random',
        'B3Counting', 'B3Random',
        'B4Counting', 'B4Random',
        'B5Counting', 'B5Random',
        'B6Counting', 'B6Random',
        ]

    root = ''
    myJTransfomer = JointTransfomer('BB')

    for seq in sequences:
        anno_pth = root + 'labels' + '/%s_BB.mat' % seq
        anno = scio.loadmat(anno_pth) # (3, 21, 1500)


        for im_id in range(1500):
            if im_id >= num: break
            img_pth_l = root + seq + '/BB_left_%d.png'%im_id
            img_pth_r = root + seq + '/BB_right_%d.png'%im_id

            anno_xyz = anno[:, :, im_id]

            # TODO: transpose (3,21) to (21,3)

            # TODO: add the code that changes 3d joints

            # get 2d joints
            anno_uv_l, anno_uv_r = myJTransfomer.transfrom3d_to_2d(anno_xyz)

            if os.path.exists(img_pth_l):
                img_l = misc.imread(img_pth_l)
                img_l, anno_uv_l = crop(img_l, anno_uv_l)
                new_img_l, new_anno_uv_l = flip(img_l, anno_uv_l)
                img_pth = outpath + '%08d.png' % cnt
                misc.imsave(img_pth, new_img_l)
                infor_dict['image'].append(get_image_dict(cnt, new_anno_uv_l, pts_3d=anno_xyz))
                cnt += 1

            if os.path.exists(img_pth_r):
                img_r = misc.imread(img_pth_r)
                # crop and resize
                img_r, anno_uv_r = crop(img_r, anno_uv_r)
                # flip
                new_img_r, new_anno_uv_r = flip(img_r, anno_uv_r)
                img_pth = outpath + '%08d.png' % cnt
                misc.imsave(img_pth, new_img_r)
                infor_dict['image'].append(get_image_dict(cnt, new_anno_uv_r, pts_3d=anno_xyz))
                cnt += 1



    return cnt

def main():
    train_outpath = "/home/lyf2/dataset/3dhand/dataset/train/image/"
    test_outpath = "/home/lyf2/dataset/3dhand/dataset/test/image/"
    train_newlabel_path = "/home/lyf2/dataset/3dhand/dataset/train/joints.json"
    test_newlabel_path = "/home/lyf2/dataset/3dhand/dataset/test/joints.json"

    # trainset
    infor_dict = {'image' : []}
    cnt = 0
    # test the process use only 3 images
    #cnt = process_PANOPTIC(train_outpath, infor_dict, cnt, 3)
    cnt = process_MPII(train_outpath, infor_dict, cnt, 1, trainset=True)
    cnt = process_stereo(train_outpath, infor_dict, cnt, 3)

    infor_dict['img_num'] = cnt
    fjson = open(train_newlabel_path, 'w')
    json.dump(infor_dict, fjson)
    fjson.close()


    # testset
    infor_dict = {'image' : []}
    cnt = 0
    # test the process use only 3 images
    #cnt = process_MPII(test_outpath, infor_dict, cnt, 3, trainset=False)

    infor_dict['img_num'] = cnt
    fjson = open(test_newlabel_path, 'w')
    json.dump(infor_dict, fjson)
    fjson.close()

if __name__ == '__main__':
    main()