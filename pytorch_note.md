## 第四章 神经网络工具 nn

总结：

torch.nn 的核心模块是 Module，它是一个抽象概念，既可以表示神经网络的某个层(layer)，也可以表示一个包含很多层的神经网络。在实际使用中，最常见的做法是继承 nn.Module，编写自己的网络/层。

全连接层又被称为仿射层：$y=Wx+b$

自定义仿射层总结：
- 初始化w、b参数，使用函数 `torch.randn`
- 在定义网络的parameter时，需要用 `torch.nn.Parameter` 进行封装，否则通过 `layer.parameters()` 不能获取参数，并且该类型默认自动求导
- torch.tensor 对象能够自动进行广播操作，无需使用 `expand_as` 函数
- 如果想替换神经网络中参数的值，必须使用 `data` 属性（除了初始化外，不要设置可学习的参数）
- 图像的 weight 参数为 (out_channels, in_channels, H, W)
- BatchNorm: 批规范化；InstanceNorm: 实例规范化；
- Dropout: 用于防止过拟合
- BatchNorm1d 在 doc 中的公式是：$y=\frac{x-E[x]}{\sqrt{Var[x]+\epsilon}}*\gamma+\beta$，其中 $Var[x]$ 是批上的有偏估计，可以通过 `a.var(dim=0, unbiased=False)` 计算有偏估计的方差
- Dropout: 该层使每个输出有p的概率变为0，例如 `dropout=torch.nn.Dropout(0.5)` 表示每个元素有0.5的概率变为0，但最终变为 0 的个数并不一定刚好等于一半数量；另外，其他不为0的输出，需要变为原来的 $1/p$ 倍，以大概保持期望不变；
- ReLU: 该函数有一个 inplace 参数，如果设置为 True，它会把输出直接覆盖到输入中，这样可以节省内存/显存。返回的也是该输入参数；
- Sequential 构造的3种方式：
    - `torch.nn.Sequential(conv1, batchnorm, relu)`
    - `seq = torch.nn.Sequential(); seq.add_module('conv', conv1);...`
    - `torch.nn.Sequential(OrderedDict([('conv', conv1), ...]))`
- LSTM的使用：input 的shape为 (sequence_size, batch_size, in_features), hx.shape 为 (batch_size, out_features), cx.shape 为 (batch_size, out_features), LSTM 的构造函数为 (in_features, out_features, layers)
- CrossEntropyLoss 最终的值是在批上平均过的；在传入 CrossEntropyLoss 之前，输入不需要softmax；
- 具有可学习参数的，如 linear、conv，使用 `nn.Linear`、`nn.Conv2d`，对于不具有可学习参数的，使用torch.nn.functional 中的 `relu/sigmoid/tanh/pooling`
- 对于 dropout 操作，输入没有可学习参数，但是 `train`模式和`eval`模式的结果不同，所以还是使用 `nn.Dropout`构建(在train时，dropout生效，在 eval 时，dropout 无效)
- `torch.nn.DataParallel` 可以将网络放到多 CPU 上面进行加速
- 在 `torch.nn.Conv2d` 后面如果存在 `torch.nn.BatchNorm2d` 层，则 `torch.nn.Conv2d` 的 `bias` 参数应该设置为 False，因为 BatchNorm2d 就有调节偏置的作用；

## 微调 Torchvision 模型

我们可以在原来的 torchvision 的基础上进行两个操作：**finetune** 和 **feature extract**。

因为模型架构的不同，所以开发者想要针对特定的模型进行调整，没有一个万能的模板；

**finetune** 和 **feature extraction** 都属于迁移学习。

- finetune：我们首先获取经过了预训练的模型，然后用我们新任务的数据对训练好的模型进行微调，以适配我们的新任务；
- feature extraction: 我们只调整最后获得预测的层（全连接层），我们只使用之前的卷积层进行特征提取，修改的是输出层；

通常来说，两种迁移学习都包含下面的步骤：
- 初始化预训练模型
- 修改最后一层，使得输出层与我们想要预测的种类数量相同
- 定义优化算法，确定哪些参数在训练时需要修改。对不需要修改的参数调用`param.requires_grad=False`
- 进行训练