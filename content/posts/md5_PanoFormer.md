---
title: "attention in CV 学习笔记：PanoFormer"
date: 2023-2-26T01:50:11+08:00
math: true
---


---

# attention in CV 学习笔记：PanoFormer


回顾一下ViT和Swin，可以发现图像中应用attention机制的一个很重要的部分就是patch，或者说token。ViT关注的是patch与patch之间的注意力，而Swin则利用了卷积的结构，将注意力机制放在了局部的patch的内部。因此，patch作为attention的基本单位，无疑具有很大的研究潜力。反过来思考一下，虽然swin及其变体通过patch内部的attention取得了很大的成功，但是patch的形态反而限制了attention的性能，我们可以通过改造patch来引入归纳偏置来帮助模型学习。

# PanoFormer: Panorama Transformer for Indoor 360° Depth Estimation

- 论文地址：<https://arxiv.org/abs/2203.09283>
- 参考代码：<https://github.com/zhijieshen-bjtu/PanoFormer>

### 背景介绍
基于CNN的深度学习的方式在普通深度估计上取得了很好的效果。然而，如果要对全景照片进行深度估计，CNN效果就不是很好。这是由于全景照片是基于EPR进行投影过的，在图像的上下两端会造成非常巨大的形变。CNN在这种任务上有一种巨大的劣势：卷积的感受野是固定的，也就导致CNN只能螺蛳壳里做道场，被动地想办法去移除形变造成的影响，例如SphereNet。(虽然当时，我觉得SphereNet是一篇不错的工作)

PanoFormer的作者则反其道而行之，为什么不能让算子来适配图片的形变呢，例如Deformable CNN。这就要求最好能够找到一种没有归纳偏置的算子，而Transformer就是一个很好的选择。因此，PanoFormer的核心思想就是，让Transformer去寻找一个非规则的patch然后计算attention。

### 如何选取patch?

本文的核心在于三点：
- pixel-level patch division strategy
- relative position embedding method
- panoramic self-attention mechanism

ViT这种模型会直接把图片分成$16\times16$的patch，这样在全景图上会损失很多细节，不利于密集预测。所以文章采用的分割算法是与patch的中心位置相关的。在这篇文章中，作者采用了一个$3\times3$的patch，也就是8个pixels环绕中心pixel。所以整个patch可以定义为中心点的位置和中心点到其他八个点的相对位置。

那么问题变成了，以知一个中心点位置和一张panorama，如何找到剩余8个点。由于panorama是从球面上投影到平面上的，所以我们在球面上的取得一个patch是近似没有变形的。因此，流程变为：
1. 将中心点投影回球面 
2. 在球面上找以该点为中心的patch
3. 在patch上找8个对应点
4. 找到这8个点在panorama上的对应点

球面为$S^2$，对于任一点$S_(x, y)=(\theta, \phi)$对其周围的8个点的定义为：
$$
S(\pm 1, 0)=(\theta\pm \Delta\theta, \phi)\\\
S(0, \pm1) = (\theta, \phi\pm\Delta\phi)\\\
S(\pm1, \pm1)=(\theta\pm\Delta\theta, \phi\pm\Delta\phi)
$$
上面是论文中给出的公式，但是我实际上觉得这里存在一点问题，水平方向的临近点不能简单的用$\theta$的加减表示，实际寻找应该发生在切平面上，按照下图所示。

![](/src/panoformer_projection.png)

按照这个方式，我们就可以得到许多非规则的patch的基础形状，之所以说是基础是因为网络在后面学习了一个$\Delta\theta$和$\Delta\phi$用于更精细的修正。有点DeformableCNN的感觉。我们可以构造出一个每个patch采样的基础位置函数$s$。

### PST Block

网络模型如下所示，是一个很简单的U-Net结构。
![](/src/panoformer_model.png)

可以观察到，最关键的结构就是PST Block这个东西。
![](/src/panoformer_pstblock.png)
作者首先用LeFF替换了FFN。然后改造了multi-head attention结构。当一个$H\times W\times C$的特征图输入的时候，首先会展开$HW$成$N\times C(N=H\times W)$的形状，然后分别由全连接层生成$M(M=C/d)$个头的$Q\in\mathbb{R}^{N\times d}$和$V\in\mathbb{R}^{N\times d}$, $Q$经过一个全连接层得到$W\in\mathbb{R}^{N\times 9}$然后经由Softmax得到attention score $A$, 同理得到一个$\Delta s\in\mathbb{R}^{N\times 18}$。$\Delta s$也就是用于矫正patch选点位置的学习值。

在得到了$s$和$\Delta s$之后，我们就可以根据$s+\Delta s$来在$V$上面采样了。对于每个头采样出来$H\times W$个patch每个patch包含9个点，可以理解为原本每个位置的特征深度变为原来的9倍。然后通过把patch和权重$A$矩阵相乘得到结果，和计算attention的原理一致，计算方法如下：
$$
\text{PSA}(f, \hat{s})=\sum^M_{m=1} W_m\times \left[\sum^{H\times W}_{q=1}\sum^9_{k=1}A_{mqk}\times W^{'}_mf(\hat{s}_{mqk}+\Delta s_{mqk}) \right]
$$
解读一下上式，m代表某个head，q代表某个patch，k代表patch中的某个点。对于每个头之间都有权重$W_m$，在一个头的计算之中，首先需要遍历所有的$H\times W$和点，也就是遍历所有的patch来计算，然后patch之内还要遍历所有的9个点。$A_{mqk}$就是每个头的每个patch的每个点的权重，$f$是用于采样patch的函数。