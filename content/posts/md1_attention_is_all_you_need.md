---
title: "attention in CV学习笔记：Attention is All you Need"
date: 2022-11-26T01:50:11+08:00
math: true
---


从NLP到CV，attention机制以及Transformer模型近几年大放异彩，大有一统江湖之势。Transformer作为一个long-range模型，一方面相对于CNN在很多需要全局信息的任务上更加有优势，另一方面具有统一NLP和CV的潜力，所以被人广泛研究（灌水）。作为一个深度学习方向的的实习生，在过去的一年零零散散看了不少attention的paper，但是始终对这个模型没有形成体系的认知，学习也颇为囫囵吞枣。所以希望重新整理一下自己看过的paper，同时记录一下自己的笔记。


---

# Attention is All you Need

- 论文地址：<https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf>
- 参考代码：<https://github.com/jadore801120/attention-is-all-you-need-pytorch>

这篇首先提出了Transformer这个完全依赖注意力机制的模型结构。

## 啥叫self-attention?

>Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence.

>self-attention是一种把单个序列的不同位置关联起来来计算序列表示的一种注意力机制

这句话已经说得很明确了，其一说明了self-attention这个self代表序列内部的计算，其二说明了self-attention会考虑到序列的所有位置，也就是global information。

## 啥叫attention?

>An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

>一个注意力函数的作用相当于把**一个**查询向量和**一系列**键值对向量映射成一个输出。这个输出是所有**值向量**的加权和，每个值向量的权重由其**对应的键向量**和**这个查询向量**得到。

**查询向量**，**键向量**，**值向量**是什么我们暂且按下不表，暂时只需要知道attention的形式是这样。


## 如何理解attention的定义?

这一部分参考了大佬的文章 

<https://zhuanlan.zhihu.com/p/410776234>

按照paper里面定义说的，我们可以考虑一个向量$\vec q=\begin{pmatrix}q_1&q_2&...&q_k\end{pmatrix}$以及一组键值对向量，其中$\vec k_i=\begin{pmatrix}k_{i1}&k_{i2}&...&k_{ik}\end{pmatrix},\vec v=\begin{pmatrix}v_{i1}&v_{i2}&...&v_{id}\end{pmatrix}$,那么上面的那一段话可以表示成$\vec {output}=\sum_{i=1}^nf(\vec q\cdot \vec k_i^T)\cdot\vec v_i$，$f$表示某种映射，不难想到$\vec {output}$应该和$\vec {v_i}$的shape一样。


知道了上面那个算法，我们直接来看下面这个公式

$$\text{Attention}(Q,K,V)=softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

这个公式在Transformer领域也算大名鼎鼎了，看起来很复杂，其实只是把很多个$\vec q$拼接成一个矩阵，相当于刚刚那一系列键值对和很多个$\vec q$做这个attention，$Q=\begin{pmatrix}q_1\\\ q_2\\\ ...\\\ q_n\end{pmatrix}，K=\begin{pmatrix}k_1\\\ k_2\\\ ...\\\ k_n\end{pmatrix}，V=\begin{pmatrix}v_1\\\ v_2\\\ ...\\\ v_n\end{pmatrix}$。

![](/blog/src/attention.png)

接下来的问题是关键了。

## Q, K, V是啥，attention为啥要这样算
 
其实Q, K, V不是凭空得出来的，他们实质上都是同一个矩阵的变形。

假设现在有一个序列$X$，序列分成了很多块（token），每个token都是一个向量,也就是$X=\begin{pmatrix}\vec x_1\\\ \vec x_2\\\ ...\\\ \vec x_n\end{pmatrix}$。现在把上面的$Q, K,V都替换成X$。那么$QK^T$算出来的结果$ANS$就是$X$中的列向量两两计算内积的结果，$ANS_{ij}$代表的就是$x_i$和$x_j$作内积(如果两个向量越接近，算出来的内积就越大)，也可以考虑成这些向量之间两两计算相似度。

$$ANS=\begin{pmatrix}\vec x_1\\\ \vec x_2\\\ ...\\\ \vec x_n\end{pmatrix}\cdot\begin{pmatrix}\vec x_1&\vec x_2&...&\vec x_n\end{pmatrix}=\\\
\begin{pmatrix}
\vec x_1和\vec x_1的相似度&\vec x_1和\vec x_2的相似度&\vec x_1和\vec x_3的相似度&...&\vec x_1和\vec x_n的相似度 \\\
\vec x_2和\vec x_1的相似度&\vec x_2和\vec x_2的相似度&\vec x_2和\vec x_3的相似度&...&\vec x_2和\vec x_n的相似度 \\\
\vec x_3和\vec x_1的相似度&\vec x_3和\vec x_2的相似度&\vec x_3和\vec x_3的相似度&...&\vec x_3和\vec x_n的相似度 \\\
...&...&...&...&...\\\
\vec x_n和\vec x_1的相似度&\vec x_n和\vec x_2的相似度&\vec x_n和\vec x_3的相似度&...&\vec x_n和\vec x_n的相似度 \\\
\end{pmatrix}
$$

然后乘上$V=\begin{pmatrix}x_1\\\ x_2\\\ ... \\\ x_n\end{pmatrix}$，算出来的结果为：

$$
\begin{pmatrix}
\sum_{i=1}^n\vec x_1和\vec x_i的相似度\cdot \vec x_i\\\
\sum_{i=1}^n\vec x_2和\vec x_i的相似度\cdot \vec x_i\\\
\sum_{i=1}^n\vec x_3和\vec x_i的相似度\cdot \vec x_i\\\
...\\\
\sum_{i=1}^n\vec x_n和\vec x_i的相似度\cdot \vec x_i\\\
\end{pmatrix}
$$

也就是说，我们通过这一通操作，把原来的每个token变成了其他所有token加权求和的形式，把所有token两两相互关联，引入了全局信息。

Q, K, V都是由X分别经过某种映射计算来的，为什么不直接采用X进行运算呢，我认为一个是把X映射到更高维度有助于计算attention，低维的向量之间相似度算出来波动很大，比如“书”和“饮料”几乎没有办法计算相似性；另外一个我认识是让整个self-attention具有了学习功能，否则整个模块在给出X的情况下算出来的解都是确定的，可以类比conv的卷积核。

至于为什么要加入一个softmax，首先当然是为了归一化，不然原本的token的值域会跟attention计算出来的token值域天差地别。另外一个我认为是为了突出高相关性的token，抑制低相关性的token，让整个attention趋于一种finetune的形式，不至于直接崩溃掉。

为什么要除以一个$\sqrt d_k$?$d_k$是$\vec q、\vec k$的维数。主要是为了归一化，消除相似性计算中对方差的影响，保证输入输出的分布一致。
不过我个人觉得这里其实并不绝对，变成余弦相似度计算$cos(\theta)=\frac{\vec x\cdot \vec y}{||\vec x||\cdot||\vec y||}$好像也行，有时间找找有没有这样做的paper。

现在可以回顾一下query,key,value的含义了，query指的就是把一个token丢进去查询。怎么查询呢？就是匹配不同的key来计算key对应的value的权重。是不是感觉一下子理解了为啥要这么叫它们，我曾经完全不理解这一堆名字，现在明白了self-attention之后才感觉这三个名字取得真的传神。

我写得self-attention/Scaled Dot-Product Attention部分代码如下：

    import torch
    import torch.nn as nn

    class Self_Attention(nn.Module):
        def __init__(self, d_input, d_k, d_v):
            '''
            d_input: 输入维度
            d_k: 论文中的d_k
            d_v: 论文中的d_v
            '''
            super(Self_Attention, self).__init__()
            
            #论文中的线性变换矩阵Wq,Wk,Wv使用线性层实现，Linear只会改变张量的最后一个维度
            self.Wq = nn.Linear(d_input, d_k)#注意，我参考的github项目这里把bias设为了False,但是我觉得线性变换应该考虑bias
            self.Wk = nn.Linear(d_input, d_k)
            self.Wv = nn.Linear(d_input, d_v)
            self.d_k = d_k

        def forward(self, input):
            '''
            input: [batch, n_num, dim]
            '''
            Q = self.Wq(input)
            K = self.Wk(input)
            V = self.Wv(input)

            x = torch.bmm(Q, K.permute(0, 2, 1))
            x = x/torch.sqrt(self.d_k)
            x = torch.softmax(x, dim=2)
            x = torch.bmm(x, V)
            return x


## multi-head attention

![](/blog/src/multi_head_attention.png)

所谓的multi-head attention其实就是把多个attention结合起来让模型学的更好。之前提到过QKV的生成是靠的X映射，现在把这个映射用线性变换代替了。只要我采用不同的线性变换，我就可以生成不同的QKV，算出不同的结果，增强模型的学习能力。

$$
\text{MultiHead}(Q, K, V)=\text{Concat}(head_1, ..., head_h)W^Q\\
head_i=\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

这个部分类比卷积中的卷积核个数，卷积核数量越多输出的维度越高，学习到的特征也更多。

我的代码如下：

    import torch
    import torch.nn as nn

    class Multi_Head(nn.Module):
        def __init__(self, d_input, d_k, d_v, n_head):
            '''
            d_input: 输入维度
            d_k: 论文中的d_k
            d_v: 论文中的d_v
            n_head: 多头注意力的头的数量
            '''
            super(Multi_Head, self).__init__()
            
            #把所有的head都集中起来，一次性变换完
            self.Wq = nn.Linear(d_input, d_k*n_head)#参考项目中这里为bias=False
            self.Wk = nn.Linear(d_input, d_k*n_head)
            self.Wv = nn.Linear(d_input, d_v*n_head)
            self.final_Linear = nn.Linear(d_v*n_head, d_input)
            self.d_k = d_k
            self.n_head = n_head

        def forward(self, input):
            '''
            input: [batch, n_num, dim]
            '''
            Q = self.Wq(input)
            K = self.Wk(input)
            V = self.Wv(input)

            Q = Q.reshape(Q.shape[0]*self.n_head, Q.shape[1], Q.shape[2]//self.n_head)
            K = K.reshape(Q.shape[0]*self.n_head, Q.shape[1], Q.shape[2]//self.n_head)
            V = V.reshape(Q.shape[0]*self.n_head, Q.shape[1], Q.shape[2]//self.n_head)

            x = torch.bmm(Q, K.permute(0, 2, 1))
            x = x/torch.sqrt(self.d_k)
            x = torch.softmax(x, dim=2)
            x = torch.bmm(x, V)

            x = x.reshape(input.shape[0], input.shape[1], -1)
            x = self.final_Linear(x)
            
            return x

## position encoding

position embedding是Transformer中不可或缺的组件。根据上面的分析我们知道了self-attention是如何引入全局知识的，很明显可以观察到一点，如果我把不同的token换个位置，好像并不会影响到self-attention的结果。但是对于实际的学习而言相对位置需要对attention有影响，举个例子：“I really want to stay at your house”和“痛，太痛了”如果放在一个句子里面，我们可以认为它们相关性很强；但是如果两个在不同的句子里面，我们就认为这两个之间的关联性没有那么强。所以额外给Transformer引入position是非常重要的。

![痛，太痛了！](/blog/src/cyberpunk_edgerunner.webp )


那么怎么引入呢？方法很简单粗暴，就是造一个跟输入一样大小的矩阵，矩阵每个位置的值就是这个位置的编码，然后把它跟输入加起来（embedding）.

编码采用以下方法：
$$
PE_{(pos,2i)}=sin\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)\\\
PE_{(pos,2i+1)}=cos\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)\\\
pos代表位置，i代表所在维度，d_{model}代表总维度
$$

至于为什么需要这样的编码，文章中提到的观点是希望模型能够学习**相对位置**（绝对位置没意义）,同时能够轻松计算**偏移量**。以下是我的理解：

首先考虑到序列可能具有不同长度的原因，周期函数更加适合对其进行编码。最典型的周期函数就是sin和cos。另外考虑到我们的模型要能够计算两个token之间的相对偏移量，单独使用sin和cos都没有办法计算，最好的办法就是同时使用，此时我们的position encoding应该具有以下形式：
$$
pe(pos)=\begin{cases}
    sin\left(\frac{pos}{wavelength}\right)\\\
    cos\left(\frac{pos}{wavelength}\right)\\\
\end{cases}
$$
假设现在有$pe(pos)$和$pe(pos+k)$，满足以下关系：

$$
sin\left(\frac{k}{wavelength}\right)=sin\left(\frac{pos+k}{wavelength}\right)cos\left(\frac{pos}{wavelength}\right)-sin\left(\frac{pos}{wavelength}\right)cos\left(\frac{pos+k}{wavelength}\right)\\\
$$

有

$$
k=wavelength*arcsin(a)
$$

然而这玩意有无穷多解，所以需要采用多个波长来编码，最后存在的公共解就是相对位置k。

我的代码如下：

    class Position_Encoding(nn.Module):
        def __init__(self, d_model):
            super(Position_Encoding, self).__init__()
            self.d_model = d_model
        
        def forward(self, n_num, dim):
            encoding = torch.zeros(n_num, dim)
            for pos in range(n_num):
                for i in range(dim):
                    if i%2 == 0:
                        encoding[pos, i] = torch.sin(pos/(10000**(i/self.d_model)))
                    else:
                        encoding[pos, i] = torch.sin(pos/(10000**((i-1)/self.d_model)))
            return encoding

## Transformer架构分析

![](/blog/src/Transformer.webp)

Transformer主要分为Encoder和Decoder两个部分组成，本质上是一个seq2seq模型，两个的区别在于Encoder的每一层只会使用上一层的信息，而Decoder的每一层除了使用上一层的输出之外还是使用了Encoder的输出，最大化利用了Encoder的深层编码信息，整个模型并不复杂，但是有些NLP方向的知识我并不了解，所以对模型细节我就不妄言了。

![](/blog/src/Transformer.png)

观察这两个结构可以很明显的看出有一个block复用的频率很高，这是一个由Position Encoding到Normalization的部分，其中最关键的部分就是一个多头注意力模块。

这个block的结构非常重要，是Transformer结构的核心，曹越老师(Swin Transformer和GCNet的作者)在VALSE的tutorial中提到，Transformer block中重要的部分就是
- Multi-head attention
- Position Encoding
- skip connection
- Layer Normalization
- Feed Forward Networks

尤其是后面CV中的几篇文章，很多都是在这个block的基础上进行改进，但是这个基本结构不会变。相对于整体的Transformer结构，我认为这个block更加重要，毕竟Transformer的结构是为了解决翻译问题，和视觉问题并不是完全耦合。

