# 基于表征的ReID方法

## Deep Transfer Learning for Person Re-identification

文章地址：<https://arxiv.org/pdf/1611.05244.pdf>

该论文针对数据稀缺问题，提出了一些迁移学习的模型，这些模型包含以下优点：
- 能够更好地将从大型图片分类数据集中学习到的特征提取方法迁移到ReID的特征提取中；
- 结合了分类损失和验证损失，每种损失函数使用不同的 dropout 策略；
- 采用了 two-stepped fine-tuning 策略；
- 给出一个无标签的 Re-ID 数据集，通过互训练(co-training),构建了一种全新的无监督深度迁移学习模型；

该模型的性能在当时(2016/11)是 state-of-the-art的：
||CUHK03|Market-1501|VIPeR|
|---|---|---|---|
|Rank-1|85.4%|83.7%|56.3%|

其中对 VIPeR 的无监督学习方式(Rank-1=45.1%)甚至优于大多数的有监督模型；

Person re-identification 的定义：在对视频监控的应用中被提出，在无重叠区域的摄像机之间匹配行人；

之前的相关工作包括：设计视角不变的行人特征，学习高效的行人相似度测量方法，或者两者兼而有之；

ReID 非常困难的原因：数据量非常少，而数据量越多，性能越好，Market1501训练出来的模型要远优于VIPeR。不过，**收集行人匹配对是非常困难的任务。**即使是最大的公开数据集也大小有限。Market-1501:1501个行人；CUHK03:1360个行人。

结论：在含有大量标签数据的数据集上训练，之后将特征提取迁移到使用场景的数据集上是必须的。通常的做法是在大的ReID数据集上学习，在小的目标数据集上微调(fine-tuning)，不过效果不好，毕竟每个数据集相对而言都非常小；而且，由于摄像头的视角等因素，导致了每个数据集中照片的差异很大；

我们采用了从更大的分类数据集（如 ImageNet）中完成学习的模型作为特征提取模型。不过将 ImageNet 中学习到的 “知识” 迁移到 Re-ID 中存在如下困难：
- 任务属性不同：分类任务和对象重识别任务差别非常大；
- 输入不同：ReID任务中，因为人通常是直立站立的，因此基本上有固定的长宽比；
- 模型结构不同：近期大多数的 Re-ID 模型结构与 ImageNet 上使用的模型结构非常不同；ReID 模型的的过滤器(filter)更小，并删去了大多数的池化层，使得网络深度更小。这些模型设计之初就是打算从0开始进行训练的，并不适合从 ImageNet 进行知识迁移；

文章中提出的 deep Re-ID network 结构，就是为了将从 ImageNet 中学习到的可泛化特征迁移到 Re-ID 任务中；

最后还做了两个设计上的选择：
- 使用经过 ImageNet 预训练的 GoogleNet 结构作为基本网络主体；
- 将 classification loss 和 verification loss 相结合；其中应用 classification loss 是因为模型需要在 ImageNet 上进行预训练，verification loss 是为了学习用于行人匹配的特征表示；通过两个 loss 的结合，我们可以缩小ImageNet 和 Re-ID数据集特征的差异；

与传统的 ReID 模型相比，我们的模型有如下明显的特点：
- 之前的模型通常有上百万的模型参数，而所有模型的目标都是小数据集，因此避免过拟合就至关重要，Dropout 被广泛应用。在我们的模型中，针对不同的损失函数，两种不同的 dropout 策略被应用；
- 两步微调训练策略；

pairwise verification loss


# 基于度量学习的ReID方法

## No Fuss Distance Metric Learning using Proxies

文章地址：<https://arxiv.org/pdf/1703.07464.pdf>

问题形式化描述：需要学习一个距离度量函数 $d(x, y; \theta)$，其中 $x$ 和 $y$ 是两个数据点(data points)。假设通过一个深度神经网络，将 $x$ 变为特征 $e(x;\theta)$，则 $d(x, y; \theta)=||e(x,\theta)-e(y,\theta)||^2_2$，其中 $\theta$ 是神经网络的参数。为了简化表示，$x$ 和 $e(x;\theta)$ 可以互换；

DML（distance metric learning）任务是学习一种在编码 $D$ 下的距离计算方式：$$d(x,y; \theta)\le d(x,z;\theta)$$ 其中 $(x,y,z)\in D$ ;

**hinge function:** $h(x) = \max(0, x)$

**NCA Loss:** $L_{NCA}(x,y,Z)=-\log(\frac{exp(-d(x,y))}{\sum_{z\in Z}exp(-d(x, z))})$

在一个图片量为n，共有 k 个类的数据集中，triplet 选择的方式大约有 $O(n^3)$ 种；


### Proxy Ranking Loss

为了解决 Triplet 采集的问题，我们提示了通过学习得到一个数据点集合 $P$，使得 $|P|\ll |D|$。我们希望 $P$ 能够近似于整体数据集，即 对于每一个 $x$，在 $P$ 中都有一个与之近似的元素。我们称这样的一个元素为 $x$ 的 proxy(代理)：$$p(x)=\arg\min_{p\in P}d(x, p)$$

并且将代理近似误差记为：$$\epsilon=\max_x d(x, p(x))$$

现在我们希望的是：$d(x,y)\le d(x,z)$，根据三角表达式有：
$$
|d(x, y)-d(x,p(y))|\le d(y,p(y))\le\epsilon \\
|d(x,p(z))-d(x,z)|\le d(z,p(z))\le\epsilon \\
=>|\{d(x,y)-d(x,z)\}-\{d(x,p(y))-d(x,p(z))\}|\le 2\epsilon
$$
只要 $|d(x,p(y))-d(x,p(z))|\gt 2\epsilon$ 时：

- 当 $d(x, p(y))-d(x,p(z))\gt\epsilon$ 时，有 $d(x,y)-d(x,z)\gt0$
- 当 $d(x,p(y))-d(x,p(z))\lt\epsilon$ 时，有 $d(x,y)-d(x,z)\lt0$

故，即使将 $y$、$z$ 用 $p(y)$、$p(z)$ 进行替换，$d(x, y)$ 和 $d(x, z)$ 的大小关系也不会改变；

记 $Pr[|d(x, p(y))-d(x, p(z))|\le 2\epsilon]$ 为不满足 $|d(x,p(y))-d(x,p(z))|\gt 2\epsilon$ 的情况数，则有：
$$E[L_{Ranking}(x;y,z)]\le E[L_{Ranking}(x;p(y), p(z))]+Pr[|d(x,p(y))-d(x,p(z))|\le2\epsilon]$$ 

假设代理的范数为 $||p||=N_p$，所有的特征数据有相同的范数 $||x||=N_x$,上面的不等式可以写得更加简洁(注：$H(\cdot)$ 为阶跃函数)：
$$L_{Ranking}(x,y,z) \\
= H(||\alpha x-p(y)||-||\alpha x-p(z)||) \\
= H(||\alpha x-p(y)||^2-||\alpha x-p(z)||^2) \\
= H(2\alpha(x^Tp(z)-x^Tp(y)))=H(x^Tp(z)-x^Tp(y))
$$

### 训练

使用 NCA 进行训练


# Other

**Zero-Shot Learning(零次学习)**：我们的模型能够对从来没有见过的类别进行分类，其中的 Zero-shot 是指对要分类的类别对象一次也不学习。

**Contrastive Loss(对比损失)**：假设共有 N 对图片，其中的图片对有相似的，也有不相似的，对比损失函数可以表示为 $$L=\frac1{2N}\sum_{n=1}^Ny_n\cdot d_n^2+(1-y_n)\cdot max(0, margin-d_n)^2;$$ 
其中 $d_n=||a_n - b_n||_2$

**Triplet Loss**：$$L=\sum_i^N\max(0, ||f(x_i^a)-f(x_i^p)||_2^2+\alpha-||f(x_i^a)-f(x_i^n)||_2^2)$$

Contrastive Loss 和 Triplet Loss 存在的问题：选择典型的 pairs 或者 triplets 对优化以及收敛速度都非常重要；

在所有的图片对中，类别不同但非常相似的图片称为 **hard negative**，而类别相同但不太相似的图片称为 **hard positive**，挑选 hard positive 和 hard negative 有两种方法，offline 和 online，具体的差别只是在训练上。

选择 hard negative 容易导致在训练中很快陷入局部最优，为了避免这个问题，在选择 negative 的时候，要满足：$$||f(x_i^a)-f(x_i^p)||^2_2 < ||f(x_i^a)-f(x_i^n)||^2_2$$
该约束条件称为：semi-hard