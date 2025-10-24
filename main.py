import os
import argparse
import torch
import warnings
warnings.filterwarnings('ignore')

from data_utils import create_dataloaders
from model import ResGRUUNet
from trainer import Trainer
from utils import setup_device, ConsoleLogger

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ResGRUU-Net 训练脚本')
    
    # 数据集参数
    parser.add_argument('--data_dir', type=str, default='./DRIVE', 
                      help='DRIVE数据集根目录')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=150, 
                      help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, 
                      help='批次大小')
    parser.add_argument('--lr', type=float, default=0.01, 
                      help='学习率')
    parser.add_argument('--momentum', type=float, default=0.9, 
                      help='动量')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                      help='权重衰减')
    
    # 模型参数
    parser.add_argument('--init_features', type=int, default=16, 
                      help='初始特征图数量（简化模型默认16）')
    
    # 保存参数
    parser.add_argument('--save_dir', type=str, default='./results', 
                      help='结果保存目录')
    parser.add_argument('--checkpoint_interval', type=int, default=10, 
                      help='检查点保存间隔')
    
    # 其他参数
    parser.add_argument('--num_workers', type=int, default=4, 
                      help='数据加载器工作进程数')
    parser.add_argument('--resume', type=str, default=None, 
                      help='恢复训练的检查点路径')
    parser.add_argument('--test_only', action='store_true', 
                      help='仅测试模式')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    device = setup_device()
    
    # 创建日志记录器
    logger = ConsoleLogger(os.path.join(args.save_dir, 'main.log'))
    logger.log(f"命令行参数: {args}")
    
    try:
        # 检查数据集目录
        if not os.path.exists(args.data_dir):
            raise ValueError(f"数据集目录不存在: {args.data_dir}")
        
        logger.log(f"使用数据集目录: {args.data_dir}")
        
        # 创建数据加载器
        logger.log("创建数据加载器...")
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        logger.log(f"训练集大小: {len(train_loader.dataset)}")
        logger.log(f"验证集大小: {len(val_loader.dataset)}")
        logger.log(f"测试集大小: {len(test_loader.dataset)}")
        
        # 创建模型
        logger.log("创建ResGRUU-Net模型...")
        model = ResGRUUNet(
            in_channels=3,
            out_channels=1,
            init_features=args.init_features
        )
        
        # 训练配置
        config = {
            'device': device,
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'momentum': args.momentum,
            'weight_decay': args.weight_decay,
            'save_dir': args.save_dir,
            'checkpoint_interval': args.checkpoint_interval,
            'early_stopping_patience': 20
        }
        
        # 创建训练器
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config
        )
        
        # 恢复训练
        if args.resume:
            if os.path.exists(args.resume):
                from utils import load_checkpoint
                start_epoch, _ = load_checkpoint(model, trainer.optimizer, args.resume)
                logger.log(f"从检查点恢复训练: {args.resume}, 起始epoch: {start_epoch}")
            else:
                logger.log(f"检查点文件不存在: {args.resume}")
        
        # 仅测试模式
        if args.test_only:
            logger.log("进入仅测试模式...")
            # 加载最佳模型
            best_model_path = os.path.join(args.save_dir, 'best_model.pth')
            if os.path.exists(best_model_path):
                from utils import load_checkpoint
                load_checkpoint(model, trainer.optimizer, best_model_path)
                logger.log(f"加载最佳模型: {best_model_path}")
                
                # 进行测试
                test_metrics = trainer.test()
                if test_metrics:
                    logger.log("测试完成!")
                    logger.log(print_metrics_summary(test_metrics, "Final Test"))
            else:
                logger.log(f"最佳模型文件不存在: {best_model_path}")
            return
        
        # 开始训练
        logger.log("开始训练ResGRUU-Net...")
        history = trainer.train()
        
        logger.log("训练完成!")
        
        # 训练结束后进行测试
        if test_loader is not None:
            logger.log("使用最佳模型进行测试...")
            best_model_path = os.path.join(args.save_dir, 'best_model.pth')
            if os.path.exists(best_model_path):
                from utils import load_checkpoint
                load_checkpoint(model, trainer.optimizer, best_model_path)
                
                test_metrics = trainer.test()
                if test_metrics:
                    logger.log("最终测试结果:")
                    logger.log(print_metrics_summary(test_metrics, "Final Test"))
                    
                    # 保存测试结果
                    with open(os.path.join(args.save_dir, 'final_test_results.txt'), 'w', encoding='utf-8') as f:
                        f.write(print_metrics_summary(test_metrics, "Final Test"))
        
        logger.log("所有任务完成!")
        
    except Exception as e:
        logger.log(f"训练过程中发生错误: {str(e)}")
        import traceback
        logger.log(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()