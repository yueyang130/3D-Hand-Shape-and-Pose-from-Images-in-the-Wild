from __future__ import division
import torch.utils.data
from torch.autograd import Variable
from model import resnet34_Mano
from utils.transform import Scale
from torchvision.transforms import ToTensor, Compose
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import datasets
from scripts.make_dataset import show_pts_on_img, show_line_on_img
import os

# dataloader will load images as well as 2d joint heat maps
def test(input_option, model, out_path, data_pth = 'data/cropped'):
    img_transform = Compose([
        Scale((256, 256), Image.BILINEAR),
        ToTensor()])

    model.eval()

    template = open('data/template.obj')
    content = template.readlines()
    template.close()


    testloader = torch.utils.data.DataLoader(datasets.HandTestSet(data_pth, img_transform=img_transform),
                                num_workers=0,batch_size=1, shuffle=False, pin_memory=False)

    for i, data in enumerate(testloader, 0):
        print("%d/%d"%(i, len(testloader)))
        images = data
        images = Variable(images.cuda())
        with torch.no_grad():
            out1, out2, _ = model(images)

        imgs = images[0].data * 255
        imgs = imgs[:3, :, :].permute(1, 2, 0)
        imgs = imgs.cpu().numpy()

        # Display 2D joints
        u = np.zeros(21)
        v = np.zeros(21)
        for ii in xrange(21):
            u[ii] = out1[0,2*ii]
            v[ii] = out1[0,2*ii+1]
        #plt.plot(u, v, 'ro', markersize=5)
        #fig = plt.figure(1)
        #plt.imshow(imgs)
        #plt.show()
        #show_pts_on_img(imgs, np.array([u,v, np.ones_like(u)]).transpose(), img_pth=os.path.join(out_path, 'pts_%d.png'%i))
        show_line_on_img(imgs, np.array([u,v, np.ones_like(u)]).transpose(), img_pth=os.path.join(out_path, 'pts_%d.png'%i))

        joints = out2[:, :21].cpu().numpy()
        # Save 3D mesh
        file1 = open(os.path.join(out_path,str(i)+'.obj'),'w')
        for j in xrange(778):
            file1.write("v %f %f %f\n"%(out2[0,21+j,0],-out2[0,21+j,1],-out2[0,21+j,2]))
        for j,x in enumerate(content):
            a = x[:len(x)-1].split(" ")
            if (a[0] == 'f'):
                file1.write(x)
        file1.close()

def main():
    # 1 use image and joint heat maps as input
    # 0 use image only as input
    input_option = 0
    model = resnet34_Mano(input_option=input_option)
    model.cuda()
    mdict = torch.load('data/model-' + str(input_option) + '-module.pth')
    model.load_state_dict(mdict)

    test(input_option, model, out_path="/home/lyf2/dataset/3dhand/visual_test/test2/out_author/", data_pth="/home/lyf2/dataset/3dhand/visual_test/test2/image/")


def main2():
    ls = [91000]
    #ls = [10000,13000,15000,17000,19000,21000,30000,40000,50000,60000,70000]
    for iter in ls:
        input_option = 0
        model_pth = "/home/lyf2/checkpoints/3dhand/train/train_model0_3d_norm_no_detach/checkpoints/model-0_%08d.pth"%iter
        assert os.path.isfile(model_pth)
        model_name = os.path.splitext(os.path.split(model_pth)[1])[0]
        out_pth =  "/home/lyf2/dataset/3dhand/visual_test/test2/" + model_name
        if not os.path.exists(out_pth):
            os.makedirs(out_pth)

        #model = torch.nn.DataParallel(resnet34_Mano(input_option=input_option), device_ids=[0])
        model = resnet34_Mano(input_option=input_option)
        model.load_state_dict(torch.load(model_pth))
        model.cuda()
        test(input_option, model, out_pth, data_pth="/home/lyf2/dataset/3dhand/visual_test/test2/image/")

if __name__ == '__main__':
    main2()