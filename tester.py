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
from scripts.make_dataset import show_pts_on_img
import os

# dataloader will load images as well as 2d joint heat maps
def test(input_option, model, out_path):
    img_transform = Compose([
        Scale((256, 256), Image.BILINEAR),
        ToTensor()])

    template = open('data/template.obj')
    content = template.readlines()
    template.close()


    testloader = torch.utils.data.DataLoader(datasets.HandTestSet('data/cropped', img_transform=img_transform),
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
        show_pts_on_img(imgs, np.array([u,v, np.ones_like(u)]).transpose(), img_pth=os.path.join(out_path, 'pts_%d.png'%i))


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
    model = torch.nn.DataParallel(resnet34_Mano(input_option=input_option), device_ids=[0])
    model.load_state_dict(torch.load('data/model-' + str(input_option) + '.pth'))
    model.eval()
    test(input_option, model, 'data/out')



