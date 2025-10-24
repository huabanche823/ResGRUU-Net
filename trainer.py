import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from utils import (
    ConsoleLogger, MemoryMonitor, calculate_metrics, save_checkpoint,
    save_predictions, plot_training_curve, plot_metrics_curve, clear_memory
)

class Trainer:
    """训练器类"""
    def __init__(self, model, train_loader, val_loader, test_loader=None, config=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # 配置参数
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        # 设备设置
        self.device = self.config['device']
        self.model.to(self.device)
        
        # 损失函数和优化器
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            momentum=self.config['momentum'],
            weight_decay=self.config['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
        
        # 监控器
        self.logger = ConsoleLogger(os.path.join(self.config['save_dir'], 'training.log'))
        self.memory_monitor = MemoryMonitor()
        
        # 训练历史
        self.history = defaultdict(list)
        
        # 最佳模型记录
        self.best_val_loss = float('inf')
        self.best_val_auc = 0.0
        
        # 创建保存目录
        os.makedirs(self.config['save_dir'], exist_ok=True)
        os.makedirs(os.path.join(self.config['save_dir'], 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.config['save_dir'], 'predictions'), exist_ok=True)
        
        # 记录配置信息
        self.logger.log(f"训练配置: {self.config}")
        self.logger.log(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    def _get_default_config(self):
        """获取默认配置"""
        return {
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'epochs': 150,
            'learning_rate': 0.01,
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'save_dir': './results',
            'checkpoint_interval': 10,
            'early_stopping_patience': 20,
            'threshold': 0.5,
            'gradient_clipping': 1.0
        }
    
    def train_one_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        epoch_metrics = defaultdict(list)
        
        # 创建进度条
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            # 移动到设备
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            
            # 梯度裁剪
            if self.config['gradient_clipping'] > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clipping'])
            
            self.optimizer.step()
            
            # 计算指标
            metrics = calculate_metrics(outputs, labels, self.config['threshold'])
            
            # 累加损失和指标
            total_loss += loss.item()
            for key, value in metrics.items():
                epoch_metrics[key].append(value)
            
            # 获取内存使用情况
            mem_usage = self.memory_monitor.get_memory_usage()
            gpu_mem_usage = self.memory_monitor.get_gpu_memory_usage()
            
            # 更新进度条
            current_lr = self.optimizer.param_groups[0]['lr']
            avg_loss = total_loss / (batch_idx + 1)
            
            pbar.set_postfix({
                'loss': f"{avg_loss:.4f}",
                'acc': f"{np.mean(epoch_metrics['accuracy']):.4f}",
                'lr': f"{current_lr:.6f}",
                'mem': f"{mem_usage['rss']:.2f}GB",
                'gpu_mem': f"{gpu_mem_usage['allocated']:.2f}GB" if gpu_mem_usage else "N/A"
            })
            
            # 定期清理内存
            if batch_idx % 100 == 0:
                clear_memory()
        
        # 计算平均指标
        avg_loss = total_loss / len(self.train_loader)
        train_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
        train_metrics['loss'] = avg_loss
        
        # 记录历史
        self.history['train_loss'].append(avg_loss)
        for key, value in train_metrics.items():
            if key != 'loss':
                self.history[f'train_{key}'].append(value)
        
        return train_metrics
    
    def validate(self, epoch):
        """验证"""
        self.model.eval()
        total_loss = 0.0
        val_metrics = defaultdict(list)
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch+1}")
            
            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # 计算指标
                metrics = calculate_metrics(outputs, labels, self.config['threshold'])
                
                # 累加
                total_loss += loss.item()
                for key, value in metrics.items():
                    val_metrics[key].append(value)
                
                # 保存预测结果（每10个epoch保存一次）
                if epoch % 10 == 0 and batch_idx < 5:  # 只保存前5个样本
                    save_predictions(
                        images, labels, outputs,
                        os.path.join(self.config['save_dir'], 'predictions'),
                        prefix=f"val_epoch_{epoch}"
                    )
        
        # 计算平均指标
        avg_loss = total_loss / len(self.val_loader)
        val_metrics_avg = {key: np.mean(values) for key, values in val_metrics.items()}
        val_metrics_avg['loss'] = avg_loss
        
        # 记录历史
        self.history['val_loss'].append(avg_loss)
        for key, value in val_metrics_avg.items():
            if key != 'loss':
                self.history[f'val_{key}'].append(value)
        
        return val_metrics_avg
    
    def test(self):
        """测试"""
        if not self.test_loader:
            self.logger.log("测试集未提供")
            return None
        
        self.model.eval()
        test_metrics = defaultdict(list)
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Testing")
            
            for batch_idx, (images, labels) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                metrics = calculate_metrics(outputs, labels, self.config['threshold'])
                
                for key, value in metrics.items():
                    test_metrics[key].append(value)
                
                # 保存测试预测结果
                save_predictions(
                    images, labels, outputs,
                    os.path.join(self.config['save_dir'], 'predictions'),
                    prefix=f"test_{batch_idx}"
                )
        
        test_metrics_avg = {key: np.mean(values) for key, values in test_metrics.items()}
        return test_metrics_avg
    
    def train(self):
        """完整训练过程"""
        self.logger.log("开始训练...")
        early_stopping_counter = 0
        
        try:
            for epoch in range(self.config['epochs']):
                start_time = time.time()
                
                # 训练
                train_metrics = self.train_one_epoch(epoch)
                
                # 验证
                val_metrics = self.validate(epoch)
                
                # 更新学习率调度器
                self.scheduler.step(val_metrics['loss'])
                
                # 打印 epoch 总结
                epoch_time = time.time() - start_time
                self.logger.log(f"Epoch {epoch+1} 总结:")
                self.logger.log(f"训练损失: {train_metrics['loss']:.4f}, 准确率: {train_metrics['accuracy']:.4f}")
                self.logger.log(f"验证损失: {val_metrics['loss']:.4f}, 准确率: {val_metrics['accuracy']:.4f}")
                self.logger.log(f"验证AUC: {val_metrics['auc']:.4f}, 耗时: {epoch_time:.2f}秒")
                
                # 保存检查点
                if (epoch + 1) % self.config['checkpoint_interval'] == 0:
                    checkpoint_path = os.path.join(
                        self.config['save_dir'], 'checkpoints',
                        f"epoch_{epoch+1}.pth"
                    )
                    save_checkpoint(
                        self.model, self.optimizer, epoch+1,
                        {'train': train_metrics, 'val': val_metrics},
                        checkpoint_path
                    )
                    self.logger.log(f"检查点已保存到: {checkpoint_path}")
                
                # 保存最佳模型（基于AUC）
                if val_metrics['auc'] > self.best_val_auc:
                    self.best_val_auc = val_metrics['auc']
                    best_model_path = os.path.join(
                        self.config['save_dir'], 'best_model.pth'
                    )
                    save_checkpoint(
                        self.model, self.optimizer, epoch+1,
                        {'train': train_metrics, 'val': val_metrics},
                        best_model_path
                    )
                    self.logger.log(f"最佳模型已更新 (AUC: {self.best_val_auc:.4f})")
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= self.config['early_stopping_patience']:
                        self.logger.log(f"早停机制触发，停止训练 (耐心值: {self.config['early_stopping_patience']})")
                        break
                
                # 绘制训练曲线
                if (epoch + 1) % 10 == 0:
                    plot_training_curve(
                        self.history,
                        os.path.join(self.config['save_dir'], f"training_curve_epoch_{epoch+1}.png")
                    )
                    plot_metrics_curve(
                        self.history,
                        os.path.join(self.config['save_dir'], f"metrics_curve_epoch_{epoch+1}.png")
                    )
                
                # 清理内存
                clear_memory()
        except Exception as e:
            self.logger.log(f"训练过程中发生异常: {str(e)}")
            # 保存紧急检查点
            emergency_checkpoint_path = os.path.join(
                self.config['save_dir'], 'checkpoints', 'emergency_checkpoint.pth'
            )
            save_checkpoint(
                self.model, self.optimizer, epoch + 1,
                {'train': train_metrics, 'val': val_metrics},
                emergency_checkpoint_path
            )
            self.logger.log(f"紧急检查点已保存至: {emergency_checkpoint_path}")
            raise  # 重新抛出异常以便用户查看错误详情
            
            # 训练结束
            self.logger.log("训练完成!")
            
            # 绘制最终训练曲线
            plot_training_curve(
                self.history,
                os.path.join(self.config['save_dir'], "training_curve_final.png")
            )
            plot_metrics_curve(
                self.history,
                os.path.join(self.config['save_dir'], "metrics_curve_final.png")
            )
            
            # 测试
            if self.test_loader:
                self.logger.log("开始测试...")
                test_metrics = self.test()
                if test_metrics:
                    self.logger.log("测试结果:")
                    self.logger.log(print_metrics_summary(test_metrics, "Test"))
                    
                    # 保存测试结果
                    with open(os.path.join(self.config['save_dir'], 'test_results.txt'), 'w', encoding='utf-8') as f:
                        f.write(print_metrics_summary(test_metrics, "Test"))
            
            # 保存训练历史
            torch.save(self.history, os.path.join(self.config['save_dir'], 'training_history.pth'))
            
        except KeyboardInterrupt:
            self.logger.log("训练被用户中断")
            
            # 保存最近的日志
            self.logger.save_buffer(os.path.join(self.config['save_dir'], 'recent_logs.txt'))
            
            # 保存当前模型
            interrupted_path = os.path.join(self.config['save_dir'], 'interrupted_model.pth')
            save_checkpoint(
                self.model, self.optimizer, epoch+1,
                {'train': train_metrics, 'val': val_metrics} if 'train_metrics' in locals() else None,
                interrupted_path
            )
            self.logger.log(f"当前模型已保存到: {interrupted_path}")
            
            # 绘制当前训练曲线
            plot_training_curve(
                self.history,
                os.path.join(self.config['save_dir'], "training_curve_interrupted.png")
            )
            plot_metrics_curve(
                self.history,
                os.path.join(self.config['save_dir'], "metrics_curve_interrupted.png")
            )
        
        return self.history