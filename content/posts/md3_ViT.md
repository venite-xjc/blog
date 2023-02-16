---
title: "attention in CV 学习笔记：ViT"
date: 2022-11-26T01:50:11+08:00
math: true
---


---

# AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE

- 论文地址：<https://arxiv.org/abs/2010.11929>

## Transformer in CV

在这篇文章之前，已经有很多工作是关于如何把attention应用到CV任务上，有的是把attention添加到模型里面，要么是替换掉一些CNN的组件。但是很少有任务是把纯attention网络应用到CV上的。

而在NLP领域，Transformer已经成为了主流方法，由于Transformer的long-range和缺乏归纳偏置的特性，Transformer通常是需要在超大规模数据集上进行预训练然后在小规模数据集上fine-tune。同时，Transformer至今没有出现性能饱和的现象，这就让我们在超大规模数据集上训练一个超大模型成为可能。

鉴于CV领域并没有有效地推广Transformer，作者希望能够在尽量少的改变标准的Transformer的情况下把Transformer应用到CV领域。

这篇文章在Transformer in CV领域地位非常高，可以说直接导致了Transformer在CV的大火，在这篇文章出来的几个月之内，就有很多篇文章follow他们的工作到其他方面。它的主要贡献我概括为以下几点：
1. 证明了nlp里面的Transformer架构是可以直接迁移到CV的。
2. 提出了如何适配Transformer处理图像的问题。
3. 分析了Transformer和CNN的区别，并且提出了如何训练ViT（a lot of money, a lot of GPUs）。
4. 再次证明了google的算力不是开玩笑的。

## An image is worth 16×16 words

这篇文章的心心念念地就是把标准Transformer迁移到CV领域，所以本篇文章的Transformer和nlp里面的Transformer保持了一致。

一个问题是，在attention is all you need里面，我们知道了Transformer的输入是一个序列，如何把图像处理成一个序列呢？一个非常简单的方法就是reshape一把梭哈，直接把[C,H,W]变成[C, H×W]就是一个典型的序列了，正如Non-local Net的做法。

然而，一般图像的大小是224×224或者256×256，如果按照pixel展开的话得有50176的长度或者65536的长度，让他们去做self-attention计算量空前的大。non-local对此给出的解决办法是尽量放在模型的后面，以及采用bottleneck来降低分辨率，GCNet采用的方法是直接回避掉相互计算attention部分，所以它可以放在任意一层。但是都不太适用于Transformer，毕竟ViT打出的旗帜就是“没有卷积”，这些操作都需要卷积参与。

Transformer针对这个问题采用的就是**划分patch**的方法，把patch替换pixel作为一个token。如果 $(P, P)$ 是一个patch的分辨率的话，对于一个 $x\in \mathbb{R}^{H×W×C}$ 就可以划分成 $x_p\in \mathbb{R}^{N×(P^2\cdot C)}$ 的序列，其中 $N=HW/P^2$ 。假设patch边长是16的话，一个256×256的图片就是有16×16个patch，因为nlp里面的token对应的是一个单词，所以就有了本文标题：**一张图片值16×16个单词**。

这个划分patch来降低token个数的方法基本被后面的Transformer继承。

## 如何适配Transformer

我们已经知道了ViT是如何构造输入token，一个token的维度为$p^2\cdot C$，所以和标准Transformer一样，需要首先进行一个线性映射来把维度映射到Transformer的维度D。

文章还沿用了BERT的方法，加入了一个class token,并且把这个token定义为了$z^0_0=X_{class}$，相当于图像在(0,0)位置增加了一个token。在训练过程中，这个token和其他token是等价的，所以会逐渐融合到其他token的信息，在最后做分类人物的时候就不需要提取所有的输出了，直接把这个token的信息提取出来接上一个分类头就可以了。

在预训练的时候分类头是一个MLP，在微调的时候分类头是一个单层线性层。

对position encoding，作者也做了实验，发现一个1D的可学习的position embedding并不比2D的position embedding表现差，所以文章后面的实验就直接使用了这个1D的position embedding。

ViT的Encoder和标准的Transformer的Encoder基本没有区别，Feed Forward Networks采用了一个双层的GELU激活的MLP，其中Decoder直接采用了一个MLP分类头，因为这个任务比较简单。整体架构如下图：

![](/src/ViT.png)

在训练中，通常在预训练时期采用较低分辨率，在fine-tune时期采用较高分辨率。这时候如果保持patch的size不变，就会产生更长的序列，但是更长的序列的position是没有意义的。比如预训练的时候最长为196，然后fine-tune的时候长度到了256，Transformer是学习不到后面197~256的position的意义的。作者这里采用了2D的插值，把position都缩放到预训练的长度范围中。

作者还分析了ViT的归纳偏置，ViT或者说attention本身是不具有针对图像的归纳偏置的。对于CNN来说，它天然具有局部性(locality)和平移不变性(translation equivariance)，局部性可以理解为它的操作是针对局部进行的，平移不变性可以理解为对于不同位置的特征，只要特征值和卷积参数保持不变，算出来的结果没有区别。而对于ViT，MLP给它提供了平移不变性，但是相邻结构是非常稀疏的(反着讲就是它很global)。position embedding在开始也是不含任何信息的，所有的归纳偏置都要靠学习获得。

说白了，Transformer出厂是白纸一张，需要大量的数据帮助它完成初始学习，而CNN出厂自带一些偏置，所以预训练更为简单。

## 实验分析

作者采用的几个模型如下：

|Model|Layers|Hidden size D|MLP size|Heads|Params|
|-----|------|-------------|--------|-----|------|
|Vit-Base|12|768|3072|12|86M|
|Vit-Large|24|1024|4096|16|307M|
|Vit-Huge|32|1280|5120|16|632M|

作者采用的预训练数据集分别为：JFT(18k classes, 303M images)、ImageNet-21k(21k classes, 14M images)、ImageNet(1k classes, 1.3M images)

![](/src/ViT_classification.png)
在分类任务上，对比BiT-L和Noisy Student, 在Image-21k上预训练的ViT并没有超过BiT-L，但是在JFT上预训练的两个大模型都达到或者超过了BiT-L。并且预训练的算力消耗更小，只需要TPU-v3训练“区区2500天”而已。

![](/src/ViT_breakdown.png)
在VTAB任务上，基本也是差不多的结果，超大规模预训练的ViT-H/14横扫战场。

分析这两个实验，作者发现在JFT数据集上预训练的ViT表现明显好于其他的ViT,这说明预训练数据集对Transformer至关重要。为了验证，作者接着做了两个实验。第一个是在不同数据集上预训练，然后再ImageNet上面fine-tune的结果：

![](/src/ViT_imagenet_finetune.png)
在ImageNet上训练，Vit还打不过ResNet(BiT)；在ImageNet-21k上预训练，虽然ViT涨点了，但是不同大小的ViT性能仍然拉不开差距；在JFT上预训练，就能够很明显发现大模型的优势。

第二个实验是随机选取部分JFT图像做预训练，然后在ImageNet上fune-tune
![](/src/ViT_sample.png)
ResNet在较小预训练集上表现优于Transformer，在大预训练集上表现不如ViT；同时依然可以得出结论：越大的ViT性能越好。

在训练数据是饱和的情况下，对于不同大小的模型，实验结果如下：
![](/src/ViT_modelsize.png)
在相同计算费用的情况下，ViT表现优于ResNet。这里说一下实验中的Hybrid，指的就是把ViT的patches换成先经过CNN处理过的feature map，杂交了两个模型。Hybrid开始表现优于Transformer，但是在大参数量的情况下Transformer成功反超。观察数据还可以发现一点，Transformer不会随着参数量的增加而性能饱和，这是很重要的一个结论。

文章最后对ViT做了一个检测的实验：
![](/src/ViT_analysis.png)
左图代表ViT最开始把patch变成低维向量的线性映射的filter,类似基函数。中间的图片代表position embedding之间的余弦相似度，可以很明显的观察出来，对于某个位置的position embedding，与它最相似的是和它同行以及同列的position embedding，并且物理距离越远余弦相似度越低，说明position encoding确实学到了位置信息。最右边是在不同深度的block中，multi-head attention所关联的patch的距离，可以发现在开始的时候self-attention既关联到了距离很近的patch，也关联到了距离很远的patch，底层的全局注意力很丰富。深度越深，学到的距离越远，说明long-range在ViT的后面发挥了很重要的作用。