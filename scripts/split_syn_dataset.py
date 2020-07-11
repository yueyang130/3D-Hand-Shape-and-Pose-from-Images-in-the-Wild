import os
import shutil
import pickle
from prepare_background import get_img_path_list

img_dir = '/home/lyf2/dataset/3dhand/syn/raw'
gt_pth = '/home/lyf2/dataset/3dhand/syn/raw/gt.pickle'

n_image = len(get_img_path_list(img_dir))
print('total number of iamges is %d'%n_image)

TESTSET_SZIE = 1000
TRAINSET_SZIE = n_image - TESTSET_SZIE

out_dir = img_dir + '_split'
trainset_pth = os.path.join(out_dir, 'train')
testset_pth = os.path.join(out_dir, 'test')
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(trainset_pth)
os.makedirs(testset_pth)

with open(gt_pth, 'r') as fd:
    gts = pickle.load(fd)

train_gt = []
test_gt = []

for ii in range(n_image):
    src = os.path.join(img_dir, '%d.png' % ii)
    if ii < TRAINSET_SZIE:
        dst = os.path.join(trainset_pth, '%d.png'%ii)
        train_gt.append(gts[ii])
    else:
        dst = os.path.join(testset_pth, '%d.png'%(ii - TRAINSET_SZIE))
        test_gt.append(gts[ii])
    shutil.copy(src, dst)

with open(os.path.join(trainset_pth, 'gt.pickle'), 'w') as fo:
    pickle.dump(train_gt, fo, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(testset_pth, 'gt.pickle'), 'w') as fo:
    pickle.dump(test_gt, fo, protocol=pickle.HIGHEST_PROTOCOL)