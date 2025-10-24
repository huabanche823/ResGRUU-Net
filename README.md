# ResGRUU-Net 图像分割算法实现

基于论文《基于ResGRUU-Net网络的图像分割算法》的PyTorch实现。

## 项目概述

ResGRUU-Net是一种结合残差网络、门控循环单元和U-Net架构的改进型图像分割网络，特别适用于医学图像分割任务。

## 论文要点

- **网络结构**: 继承U-Net的U形结构，包含下采样和上采样路径
- **ResGRU块**: 每个卷积块由2-3个ResGRU块组成，每个ResGRU块包含两个GRU模块
- **数据增强**: 采用随机切片和Mosaic切片方法进行数据扩增
- **实验数据集**: DRIVE、STARE、CHASE_DB1视网膜血管分割数据集

## 项目结构

```
resgruu_net/
├── data_utils.py      # 数据加载和预处理模块
├── model.py           # ResGRUU-Net模型定义
├── trainer.py         # 训练器模块
├── utils.py           # 工具函数
├── main.py            # 主程序入口
└── README.md          # 使用说明
```

## 环境要求

- Python 3.7+
- PyTorch 1.10.1
- torchvision
- numpy
- pillow
- scikit-learn
- matplotlib
- tqdm
- psutil

## 安装依赖

```bash
pip install torch==1.10.1 torchvision==0.11.2
pip install numpy pillow scikit-learn matplotlib tqdm psutil
```

## 数据集准备

### DRIVE数据集结构

```
DRIVE/
├── training/
│   ├── images/          # 训练图像 (20幅)
│   ├── 1st_manual/      # 手动标注 (20幅)
│   └── mask/            # 掩模图像 (20幅)
└── test/
    ├── images/          # 测试图像 (20幅)
    ├── 1st_manual/      # 手动标注 (20幅)
    └── mask/            # 掩模图像 (20幅)
```

## 使用方法

### 训练模型

```bash
python main.py --data_dir ./DRIVE --epochs 150 --batch_size 32 --lr 0.01 --save_dir ./results
```

### 恢复训练

```bash
python main.py --data_dir ../DRIVE --resume ./results/checkpoints/epoch_50.pth --save_dir ./results
```

### 仅测试模式

```bash
python main.py --data_dir ../DRIVE --test_only --save_dir ./results
```

## 主要特性

### 1. 模块化设计
- **数据处理模块**: 包含数据加载、预处理、增强等功能
- **模型模块**: ResGRUU-Net网络结构定义
- **训练器模块**: 完整的训练流程管理
- **工具模块**: 各种辅助功能

### 2. 内存优化
- 实时监控内存使用情况
- 定期清理GPU缓存
- 梯度裁剪防止梯度爆炸
- 高效的数据加载机制

### 3. 训练监控
- 实时显示训练进度条
- 显示当前精度、损失函数值
- 监控内存利用率
- 自动保存最佳模型

### 4. 数据增强
- **随机切片**: 从图像中随机提取48×48的切片
- **Mosaic切片**: 参考YOLO-v4的Mosaic数据增强方法
- **多种变换**: 翻转、旋转、亮度调整、对比度调整等

### 5. 评估指标
- **敏感度 (SE)**: 真正例率
- **特异性 (SP)**: 真负例率
- **准确率 (AC)**: 总体准确率
- **AUC**: ROC曲线下面积

## 训练参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --data_dir | DRIVE数据集根目录 | ../DRIVE |
| --epochs | 训练轮数 | 150 |
| --batch_size | 批次大小 | 32 |
| --lr | 学习率 | 0.01 |
| --momentum | 动量 | 0.9 |
| --weight_decay | 权重衰减 | 1e-4 |
| --init_features | 初始特征图数量 | 64 |
| --save_dir | 结果保存目录 | ./results |
| --checkpoint_interval | 检查点保存间隔 | 10 |
| --num_workers | 数据加载器工作进程数 | 4 |
| --resume | 恢复训练的检查点路径 | None |
| --test_only | 仅测试模式 | False |

## 输出结果

训练完成后，在`save_dir`目录下会生成以下文件：

```
results/
├── training.log          # 训练日志
├── main.log              # 主程序日志
├── best_model.pth        # 最佳模型权重
├── checkpoints/          # 检查点目录
│   ├── epoch_10.pth
│   ├── epoch_20.pth
│   └── ...
├── predictions/          # 预测结果图像
│   ├── val_epoch_0_img_0.png
│   ├── val_epoch_0_label_0.png
│   ├── val_epoch_0_pred_0.png
│   └── ...
├── training_curve_final.png  # 最终训练曲线
├── metrics_curve_final.png   # 最终指标曲线
└── final_test_results.txt    # 最终测试结果
```

## 性能指标

根据论文，ResGRUU-Net在DRIVE数据集上的预期性能：
- **准确率 (AC)**: 95.59%
- **AUC**: 0.9784
- **敏感度 (SE)**: 0.7664
- **特异性 (SP)**: 0.9927

## 注意事项

1. **内存要求**: 建议使用至少6GB显存的GPU
2. **数据路径**: 确保DRIVE数据集路径正确
3. **训练时间**: 完整训练150个epoch可能需要较长时间
4. **早停机制**: 当验证损失连续20个epoch没有改善时，训练会自动停止

## 参考文献

丁璇. 基于ResGRUU-Net网络的图像分割算法[J]. 计算技术与自动化, 2025, 44(2): 1-6.

## 联系信息

如有问题，请参考论文原文或相关技术文档。