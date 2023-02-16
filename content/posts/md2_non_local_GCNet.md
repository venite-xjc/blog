---
title: "attention in CV学习笔记：Non-local Net 与 GCNet"
date: 2022-11-26T01:50:11+08:00
math: true
---


---

# Non-local Neural Networks 

- 论文地址：<https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.html>


感觉这篇算视觉领域引入attention的先例了，主要的工作就是介绍了什么是non-local，non-local的技术细节，以及non-local在具体任务中的实现。

## 什么是non-local?

这篇文章是受到non-local mean算法的启发，定义了一个non-local operation，原文中定义是这样的：

>In this paper, we present non-local operations as a generic family of building blocks for capturing long-range dependencies. Inspired by the classical non-local means method in computer vision, our non-local operation computes the response at a position as a weighted sum of the features at all positions.

>在这篇文章中，我们将非局部操作作为一个用于捕获**长程依赖关系**的通用模块家族。受到计算机视觉中非局部均值的经典算法的启发，我们的非局部操作将一个位置（在操作中的）的响应计算为所有位置的**加权和**。

看到long-range和weighted sum of the features at all positions，相信大家也都明白了，这不就是attention嘛，只不过这一篇的思想来源是non-local means算法。不过attention跟non-local means这两个算法，你说它们是一个东西，它们的思想又不是完全一样的；你说他们不是一个东西吧，这两个的形式又基本一致……

## non-local means算法推导

non-local means是一个用于降噪的算法。我们通常的图像去噪方法就是用一个像素附近像素的均值来代替它。对于一个像素来说，如果我们在一张图片里面找到与这个像素最相似的九个像素，我们就可以把这个噪声降低三倍。但是最相近的像素不一定隔得很近，所以这篇文章的方法就是通过扫描窗口得到最相似的像素，然后用它们的均值来进行去噪。这个新的filter就是：
$$
NLu(o)=\frac{1}{C(p)}\int f(d(B(p), B(q)))u(q)dq\\\
$$
$d(B(p), B(q))$代表以$p, q$为中心的patch的欧氏距离,$f$是一个减函数，$C(p)$是一个归一化因子,$B$代表一个patch。算出来的欧式距离越小，$u(q)$前面的系数越高。

对应到像素离散的图片上面时，公式表示为：
$$
\hat{u_i}(p)=\frac{1}{C(p)}\sum\limits_{q\in B(p, r)}u_i(q)w(p, q)\\\
C(p)=\sum\limits_{q\in B(p, r)}w(p, q)
$$
$u(p)$代表像素p处的值(包括RGB三个值)，B(p, r)代表以p为中心，边长为$2r+1$的patch。
两个patch之间的欧式距离计算为：
$$
d^2(B(p, f), B(q, f))=\frac{1}{3(2f+1)^2}\sum\limits^3_{i=1}\sum\limits_{j\in B(0, f)}(u_i(p+j)-u_i(q+j))^2
$$
权重采用指数核计算:
$$
w(p, q)=e^{-\frac{max(d^2)-2\sigma^2, 0}{h^2}}
$$

后面的推导基本基于类比这上面的公式。

## 为什么要用non-local operation?

1. long-range dependencies很重要，non-local operation可以很好的捕获。
2. CNN的感受野太局限了，只有通过多次叠加卷积操作才能扩大感受野，但是这会造成信息的long-range dependencies很困难。
3. non-local operation输入大小可变，跟其他模块的耦合度很好。

## non-local具体内容介绍

首先根据non-local mean operation，作者定义了通用的non-local operation:
$$
y(i)=\frac{1}{\mathcal C(x)}\sum\limits_{\forall j} f(x_i, x_j)g(x_j)
$$

很眼熟啊，很attention is all you need 里面我推导的公式几乎一摸一样，$f$计算两个量之间的关系，$x$代表输入，$y$代表输出，$\mathcal{C(x)}$代表归一化因子。那么它表示的意义也和self-attention几乎一摸一样了。

相比于卷积，non-local operation 可以连接全局信息，而convolutional oparation只能在附近进行加权求和操作。相比于全连接层，non-local是通过相对关系来计算的，而全连接层是通过可学习的权重来完成的，这之间不能称为一个函数，或者说不能称之为一个确定的函数。（那我是不是可以叫non-local operation为specific connection guided fully connection layer哈哈）

然后这里作者分析了多种$f$的情况，对于$g$只考虑了线性变换的情况：$g(x_j)=W_gx_j$，并且采用了$1\times1$卷积来代替这个$W_q$。

对于$f$，作者考虑了4种情况：
1. Gaussian:
$$
f(x_i, x_j)=e^{x^T_ix_j}
$$
2. Embeded Gaussian:
$$
f(x_i, x_j)=e^{\theta(x_i)^T\phi(x_j)}\\\
\theta(x_i)=W_{\theta}x_i, \phi(x_i)=W_{\phi}x_i\\\
\mathcal C(x)=\sum\limits_{\forall j} f(x_i, x_j)
$$

等等，这不就是self-attention嘛(oﾟvﾟ)ノ

3. Dot product:
$$
f(x_i, x_j)={\theta(x_i)^T\phi(x_j)}\\\
\theta(x_i)=W_{\theta}x_i, \phi(x_i)=W_{\phi}x_i\\\
\mathcal C(x)=N(N是position的数量)
$$
4. Concatenation：
$$
f(x_i, x_j)=ReLU(w_f^T[\theta(x_i), \phi(x_j)])\\\
\theta(x_i)=W_{\theta}x_i, \phi(x_i)=W_{\phi}x_i, w_f是一个权重向量\\\
\mathcal C(x)=N(N是position的数量)
$$

然后考虑给$y_i$加上一个线性变换，相当于Multi-head Attention block 里面的Feed Forward Networks部分，然后加上残差连接，最终输出$$z_i=W_zy_i+x_i$$残差连接使得non-local block可以插入任何网络而不改变网络的表现。

下图是一个non-local block，可以看到它的基本形式与self-attention block没有差异，只是把全连接层换成了卷积层，然后没有包括layerNorm。我们从后来者的角度可以看出来，这个block的优化空间还是蛮大的，比如他没有借鉴到self-attention的LayerNorm和Position Encoding，没有用到我们后来熟悉的patch，以及计算复杂度还有很大的优化空间。
![](/src/non-local_block.png)


## 实验部分

如何降低计算复杂度？
- 经过$W_g$、$W_{\theta}$、$W_{\phi}$的变换之后将X的channel数减半
- 在$\theta$和$\phi$操作之后使用pool layer减少position，不会对模型性能有影响

实验结果结论：
- 即使只加上一个non-local block也能涨点！
- 四种block表现很接近，换句话说，相似度计算方式不重要，但是non-local结构很重要。
- non-local越多越好
- non-local表现好不是因为它增加了网络深度，是因为它的long-range起效果了。

---

# GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond

- 论文地址：<https://openaccess.thecvf.com/content_ICCVW_2019/html/NeurArch/Cao_GCNet_Non-Local_Networks_Meet_Squeeze-Excitation_Networks_and_Beyond_ICCVW_2019_paper.html>
- 参考代码：<https://github.com/xvjiarui/GCNet>

看这个标题就知道GCNet follow的是non-local和SENet两篇文章了。

于我所见，GCNet很大程度上算是Non-Local Net的延续性工作，但是我觉得整个思路很惊艳，所以选择把这两篇文章放到一起。

## non-local有什么问题呢？

首先，non-local计算量非常大，设想一个128$\times$128$\times$16的输入特征，如果经过1$\times$1卷积，则卷积核大小为1$\times$1$\times$16,参数量为128$\times$128$\times$16$\times$16+16=4194320（算上bias），而non-local的embedded-Gaussian版本需要四个这样的卷积层。所以原始的non-local block是显得笨重的。

另外一点，作者经过实验发现，non-local block中，对于不同的query点，所产生的注意力图都是几乎一样的,如下图所示。

![红点是query点，注意力图基本看不出来差别](/src/attention_map.png)

这个事情就很有意思了，说明对于non-local来说，query是不必要的，因为query是啥根本不影响注意力图。同时，non-local能够很好地找到全局的注意力图，也就是一整张图片上面最值得关注的部分。作者对non-local的量化分析如下：

non-local block可以表示成
$$
z_i=x_i+W_z\sum^{N_p}_{j=1}\frac{f(x_i, x_j)}{\mathcal{C}(x)}(W_v\cdot x_j)
$$
其中$x_i$表示输入，$z_i$表示输出，对于中间计算权重的那一部分，作者在后面采用的是non-local里面提到的embedded-Gaussian。

为了量化不同的query对应的attention-map之间的差别，采用了平均距离：
$$
avg\_dist=\frac{1}{{N_p}^2}\sum_{i=1}^{N_p}\sum_{j=1}^{N_p}dist(\boldsymbol{v_i}, \boldsymbol{v_j})
$$
采用余弦相似度$dist(\boldsymbol{v_i}, \boldsymbol{v_j})=\frac{1-cos(\boldsymbol{v_i}, \boldsymbol{v_j})}{2}$来分别计算输入向量的差别$(v_i=x_i)$，non-local在融合$x_i$之前的输出的差别$(v_i=z_i-x_i)$，以及attention map之间的差别。
采用JS散度$dist(\boldsymbol{v_i}, \boldsymbol{v_j})=\frac{1}{2}\sum_{k=1}^{N_p}\left(v_{ik}log\frac{2v_{ik}}{v_{ik}+v_{jk}}+v_{jk}log\frac{2v_{jk}}{v_{ik}+v_{jk}}\right)$计算attention map之间的差别。

最终结果如下表：
![](/src/non-local_analysis.png)

表的最后两列发现，non-local的输入之间还是具有很大差别的，但是经过non-local计算出来还没有融合进去的部分差别非常小，同时余弦相似度和JS散度计算出来的attention map之间的差别也很小。经过数据分析坐实了non-local的query几乎没有作用。

从这里入手，作者一步一步对non-local block进行了优化。

*需要注意一点，这里提到的query无用的观点是本文的实验结果，但是部分文章里面有提到non-local block对不同位置的attention map是有所不一样的*

## 从 non-local block 到 global context block

![](/src/arch_of_blocks.png)

第一步，由于query不起作用，说明$W_k$可以直接得到全局注意力，我们在embedde-Gaussian的基础上只采用$W_k$就够了，non-local变成了
$$
z_i=x_i+W_z\sum^{N_p}_{j=1}\frac{e^{W_kx_j}}{\sum^{N_p}_{m=1}e^{W_kx_m}}(W_v\cdot x_j)
$$

第二步，也是我认为很巧妙的一步，由于$W_v$,$W_z$都是线性变换，$\frac{e^{W_kx_j}}{\sum^{N_p}_{m=1}e^{W_kx_m}}$代表softmax是一个标量，所以对于$\sum^{N_p}_{j=1}\frac{e^{W_kx_j}}{\sum^{N_p}_{m=1}e^{W_kx_m}}(W_v\cdot x_j)$可以提出$W_k$到前面合并$W_z$,相当于又减少了一个线性变换计算。简化后的non-local block(Simplified NL block)表示为：
$$
z_i=x_i+W_v\sum^{N_p}_{j=1}\frac{e^{W_kx_j}}{\sum^{N_p}_{m=1}e^{W_kx_m}}x_j
$$

![](/src/non-local_arch12.png)

如上图的(b)所示。

现在回顾一下，由于我们并不考虑query的作用，所以对于每个position的，实际上生成的是一样的$1\times 1\times C$的向量，这个向量是由所有向量的加权和得来的。换句话说，相当于对输入的每个channel做了一个**带权值的global pooling操作**……等等，这个操作，很难不让人想到SE block的结构啊!

![](/src/SEblock.png)

毕竟SENet就是通过一个global pooling捕获每个channel的全局关系的。那来都来了，为什么不直接采用SE block后面的线性变换呢，反正SE block是非常轻量的，前面已经算过了pixel attention，后面还能接着算channel attention。顺着这个思想，作者将Simplified NL block的Transformer改成了squeeze-excitation的结构，并且加上了LayerNorm来帮助模型优化，图与公式表达如下：

![](/src/GCblock.png)

$$
z_i=x_i+W_{v2}\text{ReLU}\left(\text{LN}\left(W_{v1}\sum\limits_{j=1}^{N_p}\frac{e^{W_kx_j}}{\sum\limits_{m=1}^{N_p}e^{W_kx_m}}\right)\right)
$$

事实上，按我个人理解，前面Context Modeling计算attention的部分也有一种很SENet的感觉。就是非常暴力的**直接计算出一个全局的权重然后直接融合到原始输入**上去。只不过这俩一个是pixel-wise一个是channel-wise的。

## GCNet的分析及实验部分

在消融实验中，作者对比了单层non-local block，simplifyed non-local block,和GC block加在ResNet最后一层的效果，发现在参数量更小的情况下GCNet取得了相同的效果，并且发现如果在每层残差网络中添加GC block，会使得指标大幅度提升，同时只相当于添加了一个non-local block的参数量。

经过消融实验发现GC block加在residual block的加法操作之后和加在最后一个1$\times$1卷积之后的效果是一样的，本文默认为前一种方法。

消融实验证明GCNet最后采用加法融合的方法比SENet的缩放方法更有效。

我的GC block复现代码如下：

    import torch
    import torch.nn as nn

    class GCblock(nn.Module):
        
        def __init__(self, channel_in, Clr, fuse_type='add'):
            '''
            channel_in: channel num of inputs
            Clr: channel num of SE part
            '''
            super(GCblock, self).__init__()

            self.Wk = nn.Conv2d(in_channels=channel_in, out_channels=1, kernel_size=1, stride=1)
            self.Wv1 = nn.Conv2d(in_channels=channel_in, out_channels=Clr, kernel_size=1, stride=1)
            self.Wv2 = nn.Conv2d(in_channels=Clr, out_channels=channel_in, kernel_size=1)
            self.LN = nn.LayerNorm([Clr, 1, 1])
            self.relu = nn.ReLU()

            if fuse_type in ('add', 'scale'):
                self.fuse_type = fuse_type
            else:
                raise Exception('this fuse type is not supported')

        def forward(self, input):
            '''
            input: [B, C, H, W]
            '''
            B, C, H, W = input.shape
            weights=self.Wk(input)#[B, 1, H, W]
            weights = weights.reshape(B, H*W)
            weights = torch.softmax(weights, dim=-1)
            global_context=torch.sum(input.reshape(B, C, H*W)*weights, dim=-1)
            global_context=global_context.unsqueeze(-1).unsqueeze(-1)
            
            x = self.Wv1(global_context)
            x = self.LN(x)
            x = self.relu(x)
            x = self.Wv2(x)
            print(x)
            if self.fuse_type == 'add':
                x = x+input
            elif self.fuse_type == 'scale':
                x = x*input
            return x