# 			     3d hand复现工作说明文档

[TOC]





## 0. 总体工作说明

根据论文3D Hand Shape and Pose from Images in the Wild和开放源码https://github.com/boukhayma/3dhand（开放源码仅有network部分），复现了论文的训练过程和结果。本人工作可以分为以下部分：

- 预训练
- 训练



其中各部分的位置：

- code:           /home/workspace/yy_ws/code/3dhand/
- dataset:      /home/workspace2/dataset/3dhand/
- 训练结果： /home/workspace2/checkpoints/3dhand/
- MANO模型：/home/workspace/yy_ws/code/manopth/



环境配置：

- **python 2.7**



数据说明：

- 涉及到的所有2d坐标，我们都把它转化为x代表width, y代表height

---

## 1.预训练

### 1.1 生成数据集

create a synthetic dataset of paired hand images with their groundtruth camera and hand parameters。

脚本位于  /home/workspace/yy_ws/code/3dhand/scripts

- python2 create_colored_meshes.py
- python2 create_synthetic_data.py

生成好的数据集位于/home/workspace2/dataset/3dhand/syn/raw_split/, trainset有18000张带有groundtruth camera and hand parameters的图片， testset则有2000张。



### 1.2 训练过程

```
python2 --config configs/pretrain_model0.yaml 
```

需要注意的点：

1. 原论文并没有说明预训练采用L1 / L2损失函数，经本人尝试，**此处使用L2 Loss**更好。至于训练时的各种超参数，见configs/pretrain_model0.yaml。
2. 对于输出的23维向量，要将其分解为5个参数，0维是scale参数，1-2维是translation参数，3-5维是旋转向量，6-11维是pose参数，12-21维是shape参数。**不同的参数有相对应的权重**，具体取值见configs/pretrain_model0.yaml。
3. 进行数据增强时，注意对groundtruth parameter的处理要与image的处理相对应。



### 1.3 训练结果

训练结果位于/home/workspace2/checkpoints/3dhand/pretrain/pretrain_model0/



![pretrain_iter_1w4](\image\pretrain_iter_1w4.jpg)

------



##  2.正式训练

### 2.1 生成数据集

运行以下脚本，会将panoptic， MPII+NZSL， stereo三个数据集整合为我们要用的trainset和testset。数据集的具体介绍见论文的数据集部分。

生成的数据集位于/home/workspace2/dataset/3dhand/dataset1/。

```
python2 scripts/make_dataset.py
```

这个脚本干了以下几件事。

1. 从图片crop出hand box。
2. 因为我们用的是MANO模型（位于/home/workspace/yy_ws/code/manopth/）的right hand模型，所以还要把左手flip到右手。
3. resize到320x320
4. 因为不同的数据库有不同格式的标签，最后我们还把所有的标签都整合为我们定义的形式，并保存到一个.json文件中。

**需要注意的是，以上的crop，flip，resize操作，对应的annotation也需要变化。 ** 还有就是2**d keypoints的第一个坐标代表的是width, 第二个坐标代表的是height，这个与image先height后width相反。**



另外比较坑的是**存在三种keypoints的定义顺序**。见下图。我们统一采用MANO顺序。

但由于生成数据集时本人还没有发现存在几种定义顺序，是后来训练结果不佳才发现的。所以将其他两种定义的顺序转化为MANO顺序的工作放在了训练的数据预处理阶段，见dataset.py/HandTrainSet/\__getitem__和sort

![annotation_order](\image\annotation_order.jpg)



### 2.2 生成mask

计算mask Loss时需要用到hand的mask，我们在训练之前就生成dataset中所有图片的mask。

```
python2 scripts/segment.py
```

mask分别位于/home/workspace2/dataset/3dhand/dataset1/的train和test目录下的mask目录中。



### 2.3 训练

需要注意的几点是：

1. stereo的3d annotation的单位是mm， 而MANO生成的3d mesh没有明确的单位，因此为了计算3d Loss还需要统一二者的单位。原论文本没有说明采用什么方法，这里笔者分别使用了两种方法**，归一化和弱透视模型**。而对于归一化，又有对均值方差求导和不求导。所以一共有三种计算3d Loss的方法。分别对应config文件3d_loss的三个选择。对于归一化方法，3d Loss可以采用论文的推荐值100； 对弱透视模型，经过测试，权重取100000左右效果比较好。

2. mask loss在PyTorch下的实现需要用到 torch.nn.functional.grid_sample函数，函数有两个输入，第一个是images，要求维度为bs x C x H x W. 第二个输入是要归一化后的sample的点的坐标，但是**这个坐标却要求x代表Width， y代表height**，也就是说与image的H x W是反过来的。而与我们的2d mesh一样是先W后H，所以不需要transpose 2d mesh.

3. 在pretrain中，推荐batch size取32或64. 而在train中**，比较容易陷入鞍点，所以推荐取8或16、**

   ###                       *********************特别强调第四点***************** ###

4. 论文中提出的mask loss在训练前期有加速训练的效果，但是**如果在训练后期还取100，会导致生成的2d keypoints明显小于实际大小（见下图），也就是导致训练不收敛。**

   ![pts_0_mask100](\image\pts_0_mask100.png)

   在**训练后期建议权重下调为10或1**， 下图是mask下调为10后的结果。

   

![pts_0_mask10](\image\pts_0_mask10.png)

运行脚本进行训练：

```
# 弱透视模型计算3d Loss
python2 train.py --config configs/train_model0_3d.yaml 
# 归一化计算3d Loss， 不对均值方差求导
python2 train.py --config configs/train_model0_3d_norm.yaml 
# 归一化计算3d Loss， 对均值方差求导
python2 train.py --config configs/train_model0_3d_norm_no_detach.yaml
```





### 2.4 训练结果

笔者对三种计算3d Loss的方法分别进行了模型训练，结果见/home/workspace2/checkpoints/3dhand/train/。

train_model0_3d代表的是弱透视模型方法； Norm字样代表归一化方法。

mask1和mask10分别表示训练后期权重下调为1 和100。

author字样表示的是对论文作者训练好的模型继续训练。



![checkpoint](image\checkpoint.jpg)

从视觉效果来看，以上三种方法训练出的模型以及论文作者训练好的模型视觉效果都相差无几。以下是结果展示图：

3d mesh

![sample_3d](image\sample_3d.jpg)

弱透视模型方法得到的2d keypoints（其余效果类似就不再展示）:

![sample](F:\OneDrive\research\hand-repeat\交接文档\image\sample.jpg)



## 3. 其他脚本

将stereo中的3d annotation(单位：mm)转化为2d annotation（单位：Pixel）

```
python2 scripts/prepare_dataset/mm2px.py
```

识别hand box并crop

```
python2 scripts/crop.py
```

 测试预训练模型

```
python2 test_pretrain_model.py
```

测试模型的视觉效果(将model_pth,  img_pth, out_pth改为自己的)

```
python2 tester.py
```

生成图片的heatmap

```
python2 scripts/heat_map.py
```

注：heat_map.py依赖于PyOpenpose库，但本人实在是能力有限，前前后后折腾了三四天也没有安装好opencv3和PyOpenpose。未成功安装的opencv3.2和Pyopenpose位于/home/workspace/tools/

---


另外还有一点，今天我训练的时候，发现比以前训练慢了一些。我猜测可能是之前调试的时候将某些中间结果保存为图片查看，调试完成后没有注释掉，大量的写文件操作导致程序变慢。但今天找了半天没找到类似的代码。如果找到的话可以将其注释掉以加速训练。



笔者能力有限，所做工作肯定有很多疏漏错误之处，如果有任何疑问，请联系我。

Author:乐洋

contact me：

​    Email: yuey23@buaa.edu.cn

