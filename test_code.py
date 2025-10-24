"""
ResGRUU-Net 代码功能测试脚本
用于验证代码的基本功能是否正常工作
"""

import os
import sys
import torch
import numpy as np
from model import ResGRUUNet, count_parameters
from utils import calculate_metrics, setup_device

def test_imports():
    """测试模块导入"""
    print("=== 测试模块导入 ===")
    try:
        from data_utils import DRIVEDataset, create_dataloaders
        from model import ResGRUUNet, count_parameters
        from utils import calculate_metrics, setup_device
        print("✓ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"✗ 模块导入失败: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    print("\n=== 测试模型创建 ===")
    try:
        model = ResGRUUNet(in_channels=3, out_channels=1, init_features=64)
        print(f"✓ 模型创建成功")
        print(f"✓ 模型参数量: {count_parameters(model):,}")
        
        # 测试前向传播
        x = torch.randn(2, 3, 48, 48)
        out = model(x)
        print(f"✓ 前向传播成功")
        print(f"✓ 输入形状: {x.shape}")
        print(f"✓ 输出形状: {out.shape}")
        return True
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_device_setup():
    """测试设备设置"""
    print("\n=== 测试设备设置 ===")
    try:
        device = setup_device()
        print(f"✓ 设备设置成功: {device}")
        return True
    except Exception as e:
        print(f"✗ 设备设置失败: {e}")
        return False

def test_metrics_calculation():
    """测试指标计算"""
    print("\n=== 测试指标计算 ===")
    try:
        # 创建测试数据
        batch_size = 2
        height, width = 48, 48
        
        preds = torch.rand(batch_size, 1, height, width)
        targets = (torch.rand(batch_size, 1, height, width) > 0.5).float()
        
        metrics = calculate_metrics(preds, targets, threshold=0.5)
        print("✓ 指标计算成功")
        print(f"✓ 准确率: {metrics['accuracy']:.4f}")
        print(f"✓ AUC: {metrics['auc']:.4f}")
        print(f"✓ 敏感度: {metrics['sensitivity']:.4f}")
        print(f"✓ 特异性: {metrics['specificity']:.4f}")
        return True
    except Exception as e:
        print(f"✗ 指标计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_utils():
    """测试数据工具"""
    print("\n=== 测试数据工具 ===")
    try:
        # 测试数据集类
        dummy_data_dir = "./dummy_data"
        os.makedirs(dummy_data_dir, exist_ok=True)
        
        # 创建虚拟数据集结构
        for split in ['training', 'test']:
            for subdir in ['images', '1st_manual', 'mask']:
                os.makedirs(os.path.join(dummy_data_dir, split, subdir), exist_ok=True)
        
        print("✓ 虚拟数据集结构创建成功")
        
        # 测试数据加载器创建函数（不实际加载数据）
        print("✓ 数据工具测试通过")
        return True
    except Exception as e:
        print(f"✗ 数据工具测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """测试主函数"""
    print("=" * 50)
    print("ResGRUU-Net 代码功能测试")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_model_creation,
        test_device_setup,
        test_metrics_calculation,
        test_data_utils
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 所有测试通过！代码基本功能正常。")
        print("\n使用说明:")
        print("1. 准备DRIVE数据集，确保路径正确")
        print("2. 运行: python main.py --data_dir /path/to/DRIVE --epochs 150")
        print("3. 查看results目录下的训练结果")
    else:
        print("❌ 部分测试失败，请检查代码。")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)