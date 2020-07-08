# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as misc
import cv2 as cv
from prepare_background import get_img_path_list
import os
from utils import parse_labelfile
import shutil
import pickle
#############################################################
### set path and mode

img_dir = '/home/lyf2/dataset/3dhand/syn/raw'
img_ext = '.png'
crop_dir = '/home/lyf2/dataset/3dhand/syn/crop'
label_mode = 0 # In mode 1, every image has a single label file
               # In mode 0, all iamges have a common label file
label_ext = '.pickle'  # label file extension [.json/.mat/.pickle...]
label_pth = '/home/lyf2/dataset/3dhand/syn/raw/labels.pickle'  # In mode 1, label_pth is a directory; In mode 0, it is a file.
joint_keyname = 'hand_pts'   # the key's name of joint in label file



n_image = len(get_img_path_list(img_dir))
print('total number of iamges is %d'%n_image)


#############################################################


if os.path.exists(crop_dir):
    shutil.rmtree(crop_dir)
os.makedirs(crop_dir)

joints = parse_labelfile(label_mode, label_ext, label_pth, joint_keyname, n_image)


for ii in xrange(n_image):
    if ii % 100 == 0:
        print('crop %d / %d'%(ii, n_image))
    #with open('../../data/original/'+str(ii)+'.json', 'r') as fid:
    #    dat = json.load(fid)
    #pts = np.array(dat[])
    # pts is keypoint, whose shape is 21*3
    pts = joints[ii]
    if pts.ndim == 1:
        new_pts = np.ones((pts.shape[0]/2, 3))
        for idx in range(0, pts.shape[0], 2):
            new_pts[idx/2, 0] = pts[idx]
            new_pts[idx/2, 1] = pts[idx + 1]
        pts = new_pts

    image = misc.imread(os.path.join(img_dir,str(ii) +img_ext))

    vsz, usz = image.shape[:2]
    minsz = min(usz,vsz)
    maxsz = max(usz,vsz)

    # 判断有效性，得到的应该是一个21*1的bool矩阵
    kp_visible = (pts[:, 2] == 1)

    # 取出有效的坐标，维度小于21
    uvis = pts[kp_visible,0]
    vvis = pts[kp_visible,1]    
    
    umin = min(uvis)
    vmin = min(vvis)
    umax = max(uvis)
    vmax = max(vvis) 

    B = round(2.2 * max([umax-umin, vmax-vmin]))    

    us = 0
    ue = usz-1 

    vs = 0
    ve = vsz-1 

    umid = umin + (umax-umin)/2 
    vmid = vmin + (vmax-vmin)/2 
     
    if (B < minsz-1): 
                        
        us = round(max(0, umid - B/2))
        ue = us + B

        if (ue>usz-1):
            d = ue - (usz-1)
            ue = ue - d
            us = us - d

        vs = round(max(0, vmid - B/2))
        ve = vs + B

        if (ve>vsz-1):
            d = ve - (vsz-1)
            ve = ve - d
            vs = vs - d    
        
    if (B>=minsz-1):    
        
        B = minsz-1
        if usz == minsz:           
            vs = round(max(0, vmid - B/2))
            ve = vs + B

            if (ve>vsz-1):
                d = ve - (vsz-1)
                ve = ve - d
                vs = vs - d    

        if vsz == minsz:
            us = round(max(0, umid - B/2))
            ue = us + B

            if (ue>usz-1):
                d = ue - (usz-1)
                ue = ue - d
                us = us - d        

    us = int(us)
    vs = int(vs)
    ue = int(ue)
    ve = int(ve)        
    
    uvis  = (uvis - us) * (319.0/(ue-us)) 	
    vvis  = (vvis - vs) * (319.0/(ve-vs))     

    img = misc.imresize(image[vs:ve+1,us:ue+1,:], (320, 320), interp='bilinear')
    misc.imsave(os.path.join(crop_dir,str(ii)+'.png'),img)



