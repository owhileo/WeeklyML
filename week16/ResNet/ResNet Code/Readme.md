## cifar数据集的ResNet

`trainer.py` 是训练测试框架

常用参数：

```
--arch 模型名称，可选：'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202'，去掉引号
--epochs 迭代轮数
-b batch size
--lr 初始学习率
--save-dir 结果保存路径
```

其余可阅读`trainer.py`文件

可运行`run.sh`

`resnet.py`是模型文件

## ImageNet数据集的ResNet

模型见`ResNet_Imagenet.py`，从`pytorch`库中复制来的

