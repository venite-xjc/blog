---
title: "attention in CV 学习笔记：Swin Transformer及部分代码复现"
date: 2022-11-26T01:50:11+08:00
math: true
---


---

# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

- 论文地址: <https://arxiv.org/abs/2103.14030>
- 参考代码：<https://github.com/microsoft/Swin-Transformer>

2021 ICCV best paper，提出了一个Transformer的通用backbone

远看是Transformer, 近看是CNN。

## Transformer in CV的问题

ViT出来之后，虽然follow它的工作很多，但是很少有人能够提出一个通用的骨干网络来解决CV的各种任务。作者认为Transformer转换到CV领域的主要困难来自NLP和CV两大领域的形态不同，比如：
1. NLP中的问题规模和CV不一样，NLP的scale通常是固定的，但是CV中的scale是变化的。
2. NLP里面的token就是一个单词，整个序列往往不会太大，但是CV的分辨率是非常大的，计算复杂度以二次方增加。

Swin Transformer解决的方法：

1. 在局部计算self-attention。
2. 采用层级结构。
3. 采用shift-window打通不同的window之间的信息。(模型的名字就来自**s**hift-**win**dow)

![](/blog/src/swin_vit.png)

左图是Swin Transformer的结构，把图像划分为patch，对在同一window之类的patch计算attention，然后通过合并patch降低分辨率，不断重复操作。右图是ViT的结构，把图像划分为patch，然后始终在所有patch之间做self-attention。

对比两个结构，Swin Transformer基本放弃使用了ViT的思想，转而借鉴了很多CNN的结构。比如说局部self-attention对比局部卷积域，shift-window对比滑动窗口，并且两者都采用了层叠结构。ViT讲求的是始终考虑全局所有信息计算attenntion，而Swin Transformer仅考虑窗口内的信息计算attention。两者都有降低计算复杂度的目的，但是一个希望保持注意力计算域在整张图片上，所以着力减少token数量，另一个选择了减小注意力计算域，然后通过层叠结构逐步扩展到整张图片上，所以可以把patch划得更小。

## Swin主要结构

![](/blog/src/swin_archtecture.png)

参照上图，因为Transformer操作的基本单位都是patch，所以Swin Transformer也是首先进行patch partition，文中采用的是4×4的大小。这样一个patch就会包含4×4×3=48的维度，经过linear embedding layer把维度投影到C，分辨率变为$\frac{H}{4}\times\frac{W}{4}$。然后会经过两个Swin Transformer Block计算注意力。这个过程属于上图的stage1。

在stage2、stage3、stage4中，首先会把相邻的2×2个patch 融合到一起成为新patch，这样得到的patch维度会是4C，需要使用一个linear layer把维度投影到2C。之后经过数量不等的Swin Transformer Block计算注意力。经过这样的一个流程，特征图的分辨率会从$\frac{H}{4}\times\frac{W}{4}$依次下降到$\frac{H}{8}\times\frac{W}{8}$、$\frac{H}{16}\times\frac{W}{16}$、$\frac{H}{32}\times\frac{W}{32}$，也就是一个分层结构。

## Swin Transformer Block结构

Swin Transformer Block是Swin Transformer的关键。标准的Transformer都是在全局进行注意力计算的，但是这样会导致计算复杂度二次上升。Swin Transformer采用了在window中进行计算的方式。

A×B大小的矩阵和B×C大小的矩阵相乘计算复杂度为ABC，在self-attention中，所有的线性变换都是以矩阵乘法的形式得到的。self-attention的公式如下：

$$\text{Attention}(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

对于分辨率(patch数量)为$h \times w$的特征图，它的输入为$hw\times C$，首先需要经过三个$C\times C$的矩阵分别得到Q，K，V，那么复杂度一共就为$3hwC^2$。
计算$QK^T$的时候，是$hw\times C$的矩阵和$C\times hw$的矩阵相乘，计算复杂度为$(hw)^2C$。
计算$(QK^T)V$的时候，是$hw\times hw$的矩阵和$hw\times C$的矩阵相乘，计算复杂度为$(hw)^2C$。
在输出的时候还需要经过一次线性变换，乘上一个$C\times C$的矩阵，计算复杂度为$hwC^2$。

所以self-attention的复杂度一共为:$4hwC^2+2(hw)^2C$。

Window based Self-Attention前面得到QKV的过程和最后的线性变换和self-attention保持一致，但是计算$QK^T$变成了$\frac{h}{M}\times\frac{w}{M}$个$(M\times M)^2C$，计算$(QK^T)V$变成了$\frac{h}{M}\times\frac{w}{M}$个$(M\times M)^2C$。一共是$2M^2hwC$。

所以Window based Self-Attention复杂度一共为$4hwC^2+2M^2hwC$。

![](/blog/src/W-MSA.png)

由于划分window的方法虽然能够降低计算复杂度，但是相当于破坏了attention的全局性，计算局限于window中，为了打破不同window之间的壁垒，作者采用了shift-window的方法。具体来说，就是把window移动$(\lfloor\frac{M}{2}\rfloor, \lfloor\frac{M}{2}\rfloor)$之后重新划分。参见下图：

![](/blog/src/shift-window.png)

但是这样会带来问题，第一是window的个数由原来的4个变成了9个，第二是window中的patch数量不固定，所以作者采用了efficient batch computation的方法，如下图:

![](/blog/src/Efficient_batch_computation.png)

其实也就是通过类似circular padding的方式把不完整的几个window给补全。

但是在比如右下的window中，一个window包含了4个不同位置的patch，计算他们之间的attention是没有意义的，所以需要给它们添加mask，保证来自A的patch只跟来自A的patch算attention,来自B的patch只跟来自B的patch算attention。

另外，Swin Transformer里面还添加了relative position bias，具体添加位置参照文中的公式，就是那个B：

$$
\text{Attention}(Q,K,V)=\text{SoftMax}\left(QK^T/\sqrt{d}+B\right)V
$$

源码里面是构造一个可以学习的位置参数表，然后再建立一个相对位置索引表。对于两个patch，首先计算它们的相对位置索引，然后根据相对位置索引去找到对应的bias。

在一个Swin Transformer Block中，会首先进行一次不移动窗口的MSA计算以及依次移动窗口的MSA计算，这样整个过程表述为：

$$
\hat{z}^l=W-MSA(LN(z^{l-1}))+z^{l-1},\\\
z^l = MLP(LN(\hat{z}^l))+\hat{z}^l,\\\
\hat{z}^{l+1} = SW-MSA(LN(z^l))+z^l,\\\
z^{l+1} = MLP(LN(\hat{z}^{l+1}))+\hat{z}^{l+1},
$$
![](/blog/src/swin_archtecture.png)

## 代码复现

划分window以及逆过程代码:

    def partition(input, window_size):
        '''
        input: [B, C, H, W]
        windows_size: int
        把图片变成window
        return: [B, N, C, window_size, window_wize] N=H//window_size*W//window_size
        '''
        assert len(input.shape) == 4
        B, Channel, H, W = input.shape
        assert H % window_size == 0 and W % window_size == 0
        
        #不考虑B和C，由于reshape不会改变底层一维数组分布，所以需要把H改成[窗口数，窗口里面的H数]
        #W改成[窗口数，窗口里面的W数]才符合按行排布的一维数组情况,顺序不能反。
        output = input.reshape(B, Channel, H//window_size, window_size, W//window_size, window_size)
        #由于我们想要的输出最后两维应该是一个patch的二维形式，所以需要把维度变换过去
        output = output.permute(0, 2, 1, 4, 3, 5).reshape(B, Channel, -1, window_size, window_size).permute(0, 2, 1, 3, 4)
        return output

    def reverse(input):
        '''
        input: [B, N, C, window_size, window_size]
        partition的逆变换
        return: [B, C, H, W]
        '''
        assert len(input.shape) == 5
        
        B, N, C, window_size, _ = input.shape
        x = input.permute(0, 2, 1, 3, 4)
        x = x.reshape(B, C, int(N**0.5), int(N**0.5), window_size, window_size)
        x = x.permute(0, 1, 2, 4, 3, 5)
        x = x.reshape(B, C, int(N**0.5)*window_size, int(N**0.5)*window_size)

        return x

patch embedding代码:

    class Patch_Linear_Embedding(nn.Module):
        def __inti__(self, patch_size, dim_in, dim_out, norm = None):
            '''
            根据patch_size划分并且投影到dim_out上，仍然是二维的
            '''
            super(Patch_Linear_Embedding, self).__init__()
            self.linear_embedding = nn.Conv2d(dim_in, dim_out, kernel_size=patch_size, stride=patch_size, padding=0)
            self.norm = norm
        
        def forward(self, input):
            '''
            input: [B, C, H, W]
            '''
            x = self.linear_embedding(x)#[B, C, H, W]->[B, dim_out, num_patch, num_patch]
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)

            if self.norm is not None:
                x = self.norm(x)

            return x

MLP代码：

    class MLP(nn.Module):
        def __init__(self, dim_in, dim_out, dim_hidden, active_layer = nn.GELU, dropout = 0):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(dim_in, dim_hidden)
            self.active_layer = active_layer()
            self.fc2 = nn.Linear(dim_hidden, dim_out)
            self.Dropout = nn.Dropout(dropout)

        def forward(self, x):
            x = self.fc1(x)
            x = self.active_layer(x)
            x = self.Dropout(x)#这里面使用了两个Dropout
            x = self.fc2(x)
            x = self.Dropout(x)
            return x

添加relative position bias：

    class Relative_Postion_Bias(nn.Module):
        '''
        输入: [B*N*num_head, L, C]
        '''
        def __init__(self, num_head, M):
            super(Relative_Postion_Bias, self).__init__()
            self.bias = nn.Parameter(torch.zeros((2*M-1, 2*M-1, num_head)))#构造一个bias表
            self.num_head = num_head
            #以下查询方式都是基于window中的patch是按照行优先顺序展开的
            coords = torch.stack(torch.meshgrid([torch.arange(M), torch.arange(M)]))
            coords = torch.flatten(coords, 1)#绝对坐标
            relative_coords = coords[:, :, None]-coords[:, None, :]#相对坐标
            relative_coords = relative_coords.permute(1, 2, 0)#[M**2, M**2, 2]
            relative_coords += M-1#索引置为非负数
            self.relative_coords = relative_coords

        def forward(self, input):
            '''
            input: [B*N*num_head, N, N]
            '''
            relative_position_bias = self.bias[self.relative_coords[:, :, 0], self.relative_coords[:, :, 1]]
            relative_position_bias = relative_position_bias.permute(2, 0, 1)
            relative_position_bias = relative_position_bias.unsqueeze(0).repeat(input.shape[0]//self.num_head)
            relative_position_bias = relative_position_bias.reshape(input.shape[0], input.shape[1], input.shape[2])
            
            return input + relative_position_bias

patch merging代码：

    class Patch_Merging(nn.Module):
        def __init__(self, dim_in, dim_out, stride):
            super(Patch_Merging, self).__init__()
            self.fc = nn.Linear(dim_in*stride**2, dim_out)
            self.stride = stride
        
        def forward(self, input):
            x = partition(input, self.stride)
            x.resahpe(x.shape[0], x.shape[1], -1)
            x = self.fc(x)
            x = x.reshape(x.shape[0], x.shape[1], -1, 1, 1)
            x = reverse(x)

            return x

W-MSA及SW-MSA：

    class Window_Multihead_Attention(nn.Module):
        def __init__(self, dim, num_head, window_size):
            self.dim = dim
            self.num_head = num_head
            #Swin默认这里的线性变换不改变维度，所以我也直接用了3*dim
            #Swin后面做multi-head的时候是直接把3*dim拆成num_head个来用，我认为在这里变成3*dim*num_head然后拆成num_head个效果是一样的
            self.qkv = nn.Linear(dim, 3*dim)

            self.relative_position_bias = Relative_Postion_Bias(num_head, window_size)


        def forward(self, input, mask = None):
            '''
            input: [B*N, L, C], L是把window中的patch按行优先展开的数量
            mask: [B*N, L, L], L同上
            '''

            #[B*N, L, C]->[B*N, L, 3, num_head, C//num_head]
            qkv = self.qkv(input).\
                reshape(input.shape[0], input.shape[1], 3, self.num_head, input.shape[2]//self.num_head)
            qkv = qkv.permute(0, 3, 2, 1, 4).reshape(input.shape[0]*self.num_head, input.shape[1], 3, -1)
            
            Q = qkv[:, :, 0, :]#[B*N*num_head, L, C//num_head]
            K = qkv[:, :, 1, :]
            V = qkv[:, :, 2, :]

            attention = torch.bmm(Q, K.permute(0, 2, 1))/(self.dim//self.num_head)**0.5#计算相似度
            attention = self.relative_position_bias(attention)#加上Bias
            
            if mask is not None:
                weights = torch.exp(attention)
                weights = weights*(mask.unsqueeze(1).repeat(1, self.num_head, 1, 1)).reshape(-1, mask.shape[1], mask.shape[2])
                weights = weights/torch.sum(weights, -1)
            else:
                weights  =torch.softmax(weights, -1)
            
            output = torch.bmm(weights, V)

            return output+input

SwinTransformerBlock：

    class SwinTransformerBlock(nn.Moduel):
        def __init__(self, dim, num_head, window_size, mlp_dim_hidden):
            super(SwinTransformerBlock, self).__init__()

            self.LN1 = nn.LayerNorm(dim)
            self.LN2 = nn.LayerNorm(dim)
            self.LN3 = nn.LayerNorm(dim)
            self.LN4 = nn.LayerNorm(dim)

            self.WMSA = Window_Multihead_Attention(dim, num_head, window_size)
            self.SWMSA = Window_Multihead_Attention(dim, num_head, window_size)

            self.MLP1 = MLP(dim, dim, mlp_dim_hidden)
            self.MLP2 = MLP(dim, dim, mlp_dim_hidden)

            self.window_size = window_size

        def forward(self, input):
            '''
            input: [B, C, H, W]
            '''
            B, C, H, W = input.shape
            L = self.window_size**2
            N = H*W/L

            x = partition(x, self.window_size)
            x = x.reshape(B*N, C, L).permute(0, 2, 1)#[B*N, L, C]
            x = self.LN1(input)
            x = self.WMSA(x)
            x = x+input
            x = self.LN2(x)
            x = self.MLP1(x)
            x = x+input#[B*N, L, C]

            x = x.reshape(B, N, L, C).permute(0, 1, 3, 2).reshape(B, N, C, self.window_size, -1)
            x = reverse(x)
            shift = self.window_size//2
            x = torch.cat([torch.cat([x[:, :, shift:, shift:], x[:, :, shift:, :shift]], dim = -1), 
            torch.cat([x[:, :, :shift, shift:], x[:, :, :shift, :shift]], dim = -1)], dim = -2)

            mask = torch.ones(B, 1, H, W)
            mask[:, :, :, -shift:] = 2
            mask[:, :, -shift:, :] = 3
            mask[:, :, -shift:, -shift:] = 4
            origin = mask#[B, 1, H, W]
            mask = partition(mask, self.window_size)
            mask = mask.reshape(B, mask.shape[1], -1)#[B, N, L]
            mask = mask.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.window_size, self.window_size)#[B, N, L, ws, ws]
            mask = reverse(mask)#[B, L, H, W]
            mask = torch.where(mask==origin.repeat(1, L, 1, 1), 1, 0)

            x = self.LN3(input)
            x = self.SWMSA(x, mask)
            x = x+input
            x = self.LN4(x)
            x = self.MLP2(x)
            x = x+input#[B*N, L, C]
            
            return x

## 实验结论

![](/blog/src/swin_classification.png)
在分类任务上，Swin表现大幅超过其他模型。在Image-21k预训练的结果也是更好。

![](/blog/src/swin_object_detection.png)
在目标检测任务上，效果也是非常好。

![](/blog/src/swin_segmentation.png)
swin实现了在这种dense prediction任务上采用Transformer的backbone，效果也是大幅超越其他模型。

![](/blog/src/swin_pos.png)
关于shift-window以及relative position bias的消融实验。有了这两个东西都可以涨点。

值得一提的是，作者后面把Swin的架构用到MLP-Mixer上面，发现了效果也很好，说明Swin的结构是具有通用性的。但是这是不是也说明了Swin的结构中，Transformer的地位其实没有那么高呢，我个人感觉Swin其实融合了很多文章的trick，比如滑动窗口也是一些CNN在用的提速方法，所以Swin刚出来的时候其实有很多质疑的声音。但是不得不说Swin作为通用backbone的地位非常高，我拿这个backbone也在某个workshop上干到过第二（虽然后面掉了），Transformer的潜力还是很大的。


