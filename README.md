# 2024Fall_3ZS082006

该仓库存放了上海大学2024年秋季学期图像处理与理解的实验代码。本实验主要工作设计了**FasterNet**，**MobilenetV1**和**ResNet18**的微调模型，并测试在kaggle数据集狗的品种分类任务上Top-1测试精度。

本实验的数据集链接：https://www.kaggle.com/competitions/dog-breed-identification/

## 1. 配置依赖环境

```bash
conda create -n finetunenet python=3.9.20 -y
conda activate finetunenet
```

### 2. 数据集准备：

本实验要求数据集的格式如下所示：

```
/train
   /n01440764
       images
   /n01443537
       images
    .....
/val
   /n01440764
       images
   /n01443537
       images
    .....
```

数据集压缩

```bash
cd dataset # 先转到dataset目录下
mkdir dog_breed_identification # 在dataset目录下创建一个存放数据的目录
# 将kaggle上下载的数据压缩到dog_breed_identification目录中
python dataprocess.py -v 0.1 # 进行训练集与测试集的分割
cd ..
```

### 3. 模型训练

这一条也可以不做，本仓库的model_ckpt目录下提供了预训练好的模型。

```bash
python main.py -b 256 -n 15 -m fasternet  
python main.py -b 256 -n 15 -m resnet 
python main.py -b 256 -n 15 -m mobilenet
```
### 4. 模型测试
```bash
python main.py -b 256 -t -m fasternet --model_name fasternet_finetune-epoch=15-val_acc1=74.61.pth   # 74.61
python main.py -b 256 -t -m resnet --model_name resnet_finetune-epoch=15-val_acc1=76.76.pth         # 76.76
python main.py -b 256 -t -m mobilenet --model_name mobilenet_finetune-epoch=15-val_acc1=74.22.pth   # 74.22
```



