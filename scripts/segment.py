import numpy as np
import json
import matplotlib.pyplot as plt
import scipy.misc as misc
import cv2 as cv
import pickle
import matplotlib.pyplot as plt
import utils
import os

def show_mask_on_img(img, mask,
    img_pth = '/home/lyf2/dataset/3dhand/dataset/mask_on_img.png'):
    mask = np.expand_dims(mask, 2)
    blue_mask = np.concatenate([np.zeros_like(mask),
                                        np.zeros_like(mask), mask], axis=2)
    # mask_on_img = img + blue_mask
    # mask_on_img[mask_on_img > 255] = 255
    mask_on_img = np.copy(img)
    mask_on_img[blue_mask > 0] = 255
    misc.imsave(img_pth, mask_on_img)


def inside_polygon(x, y, points):
    n = len(points)

    inside = False
    p1x, p1y = points[0]
    for i in range(1, n + 1):
        p2x, p2y = points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

edges = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]

def generate_mask(img, anno, mask_dir, ii):
    """
    anno is 42 long vector.
    """
    anno = np.array(anno)
    if anno.shape == (21, 3):
        anno = anno[:, :2]
        anno = np.resize(anno, 42)
    assert anno.shape == (42,)

    mask = np.zeros((320, 320), np.uint8)

    # Draw skeleton

    for e in edges:
        p1u = int(anno[2*e[0]])
        p1v = int(anno[2*e[0]+1])
        p2u = int(anno[2*e[1]])
        p2v = int(anno[2*e[1]+1])

        cv.line(mask, (p1u,p1v), (p2u,p2v), 3, 70)

    for e in edges:
        p1u = int(anno[2*e[0]])
        p1v = int(anno[2*e[0]+1])
        p2u = int(anno[2*e[1]])
        p2v = int(anno[2*e[1]+1])

        cv.line(mask, (p1u,p1v), (p2u,p2v), 1, 1)

    poly_list = [[0,17,18],[0,17,1],[0,1,5],[0,5,13],[0,13,9]]
    polys = []

    # Draw triangles

    for i in poly_list:
        poly = []

        for ind in i:
            pu = int(anno[2*ind])
            pv = int(anno[2*ind+1])
            poly.append((pv,pu))
            polys.append(poly)

    for u in xrange(0,320):
        for v in xrange(0,320):
            for j in polys:
                if inside_polygon(u, v, j):
                    mask[u,v] = 1

    # Segment

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    # use GrabCut get image segmentation(mask)
    try:
        cv.grabCut(img,mask,None,bgdModel,fgdModel,5,cv.GC_INIT_WITH_MASK)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        misc.imsave(mask_dir + 'mask_%08d.png' % ii, mask2 * 255)
        misc.imsave(mask_dir + '%08d.png' % ii, mask2)
        #show_mask_on_img(img, mask2 * 255)
    except Exception:
        print "ignore the %dth mask"%(ii+1)



def main2():
    label_pth = '../../data/cropped/labels.pickle'  # 2D joint annotation location
    img_dir = 'data/cropped/'  # input image dir, please end with /
    mask_dir = ''  # output mask dir, please end with /


def main(resume_train=True, resume_index=0):
    root = '/home/lyf2/dataset/3dhand/dataset1/'
    sel = ['train','test']
    label_pths = [root + x + '/joints.json' for x in sel]  # 2D joint annotation location
    img_dirs = [root + x +'/' for x in sel]   # input image dir, please end with /
    mask_dirs = [root + x + '/mask/' for x in sel]   # output mask dir, please end with /
    # fi = open(label_pth, 'rb')
    # anno = pickle.load(fi)
    # fi.close()
    for dir in mask_dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

    for j in range(2):

        datas = utils.parse_labelfile(0, '.json', label_pths[j])
        num = datas['img_num']
        ls = datas['image']

        if resume_train:
            start = resume_index if j == 0 else 0
        else:
            start = num if j == 0 else resume_index

        for ii in xrange(start,num) :
            dat = ls[ii]
            # img = misc.imread(img_dirs[j]+str(ii) +'.png')
            img = misc.imread(img_dirs[j] + dat['img_paths'])
            anno = dat['2d_joint']
            generate_mask(img, anno, mask_dirs[j], ii)
            print '%s mask: %d/%d'%(sel[j], ii+1, num)


if __name__ == '__main__':
    main(True, resume_index=0)


