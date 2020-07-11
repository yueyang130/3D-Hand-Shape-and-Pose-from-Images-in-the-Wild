from __future__ import print_function

"""
The script can cumpute 2d joints(pixel) from 3d joints(mm)
"""

import numpy as np

# sequences = [
#     'B2Counting', 'B2Random',
#     'B3Counting', 'B3Random',
#     'B4Counting', 'B4Random',
#     'B5Counting', 'B5Random',
#     'B6Counting', 'B6Random',
#     ]
# cam = 'BB'  # [BB/SK]

class JointTransfomer:
    def __init__(self, cam):
        self.finger_ind = list(range(21))
        self.cam = cam
        if cam == 'BB':
            fx = 822.79041
            fy = 822.79041
            tx = 318.47345
            ty = 250.31296
            base = 120.054

            self.R_l = np.zeros((3,4))
            self.R_l[0,0] = 1
            self.R_l[1,1] = 1
            self.R_l[2,2] = 1
            self.R_r = self.R_l.copy()
            self.R_r[0, 3] = -base


            self.K = np.diag([fx, fy, 1.0])
            self.K[0, 2] = tx
            self.K[1, 2] = ty
        elif cam == 'SK':
            fx = 607.92271
            fy = 607.88192
            tx = 314.78337
            ty = 236.42484
            self.K = np.diag([fx, fy, 1])
            self.K[0, 2] = tx
            self.K[1, 2] = ty
            self.K = np.concatenate([self.K, np.zeros(3,1)], axis=1)
            assert self.K.shape == (3, 4)
        else:
            assert 0, 'Unsupported camera type!'


    def transfrom3d_to_2d(self, anno_xyz):
        anno_xyz_l = np.array(anno_xyz)
        assert anno_xyz_l.ndim == 2 and anno_xyz_l.shape[0] == 3
        # left
        anno_uv_l = self.K.dot(self.R_l).dot(np.concatenate([anno_xyz_l, np.ones([1, 21])], axis=0))
        for k in self.finger_ind:
            anno_uv_l[:, k] = anno_uv_l[:, k] / anno_uv_l[2, k]
        # right
        anno_xyz_r = self.R_r.dot(np.concatenate([anno_xyz_l, np.ones([1,21])], axis=0))
        anno_uv_r = self.K.dot(anno_xyz_r)
        for k in self.finger_ind:
            anno_uv_r[:, k] = anno_uv_r[:, k] / anno_uv_r[2, k]
        # Note: the third elem is valid bit
        return anno_uv_l, anno_uv_r

if __name__ == '__main__':
    #test
    myJTansfromer = JointTransfomer('BB')
    a = np.random.randn(3, 21).round(2)
    print('3d joints is\n ')
    print(a)
    print('2d joints is\n')
    print(myJTansfromer.transfrom3d_to_2d(a))
