import os
import cv2
import shutil

rawdirpath = "/home/lyf/yy_ws/code/3dhand/data/raw_backgrounds/iccv09Data/images/"
dstImgPth = '/home/lyf/yy_ws/code/3dhand/data/backgrounds/'

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(str.endswith(filename, ext) for ext in IMG_EXTENSIONS)


def get_img_path_list(dirpath, key = None):
    assert os.path.isdir(dirpath)
    img_pth_list = []
    for dirpath, _, files in os.walk(dirpath):
        for fname in files:
            if is_image_file(fname):
                if key is None or key in fname:
                    img_pth = os.path.join(dirpath, fname)
                    img_pth_list.append(img_pth)



    return img_pth_list

def preprocess_data(img):
    w, h = (320, 320)
    return cv2.resize(img, (w,h))

def make_bg_dataset():
    raw_img_list = get_img_path_list(rawdirpath)
    shutil.rmtree(dstImgPth)
    os.mkdir(dstImgPth)
    idx = 0
    for pth in raw_img_list:
        img = cv2.imread(pth)
        img = preprocess_data(img)
        cv2.imwrite(os.path.join(dstImgPth, '%d.png'%(idx)), img)
        idx += 1

def add_bg_dataset():
    raw_img_list = get_img_path_list(rawdirpath)
    dst_img_list = get_img_path_list(dstImgPth)
    idx = len(dst_img_list)
    for pth in raw_img_list:
        img = cv2.imread(pth)
        img = preprocess_data(img)
        cv2.imwrite(os.path.join(dstImgPth, '%d.png'%(idx)), img)
        idx += 1

if __name__ == '__main__':
    make_bg_dataset()








