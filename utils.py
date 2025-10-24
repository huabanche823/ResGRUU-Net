import os
import sys
import time
import psutil
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image

class ConsoleLogger:
    """控制台日志记录器"""
    def __init__(self, log_file=None):
        self.log_file = log_file
        self.buffer = []
        self.max_buffer_size = 20
        
        # 创建日志目录
        if log_file:
            log_dir = os.path.dirname(log_file)
            os.makedirs(log_dir, exist_ok=True)
    
    def log(self, message):
        """记录日志"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_message = f"[{timestamp}] {message}"
        
        # 打印到控制台
        print(log_message)
        
        # 添加到缓冲区
        self.buffer.append(log_message)
        if len(self.buffer) > self.max_buffer_size:
            self.buffer.pop(0)
        
        # 写入文件
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_message + '\n')
    
    def get_recent_logs(self):
        """获取最近的日志"""
        return self.buffer
    
    def save_buffer(self, filename):
        """保存缓冲区内容"""
        with open(filename, 'w', encoding='utf-8') as f:
            for log in self.buffer:
                f.write(log + '\n')

class MemoryMonitor:
    """内存监控器"""
    def __init__(self):
        self.process = psutil.Process()
    
    def get_memory_usage(self):
        """获取当前内存使用情况"""
        mem_info = self.process.memory_info()
        return {
            'rss': mem_info.rss / (1024 ** 3),  # GB
            'vms': mem_info.vms / (1024 ** 3),  # GB
            'percent': self.process.memory_percent()
        }
    
    def get_gpu_memory_usage(self):
        """获取GPU内存使用情况"""
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / (1024 ** 3)
            gpu_mem_cached = torch.cuda.memory_reserved() / (1024 ** 3)
            return {
                'allocated': gpu_mem,
                'cached': gpu_mem_cached,
                'percent': gpu_mem / torch.cuda.get_device_properties(0).total_memory * 100
            }
        return None

def calculate_metrics(pred, target, threshold=0.5):
    """计算分割指标"""
    # 将概率转换为二值图像
    pred_binary = (pred > threshold).float()
    
    # 展平张量
    pred_flat = pred_binary.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()
    
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(target_flat, pred_flat).ravel()
    
    # 计算指标
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # SE
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # SP
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0  # AC
    
    # 计算AUC
    try:
        auc = roc_auc_score(target_flat, pred.view(-1).cpu().numpy())
    except:
        auc = 0.0
    
    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy,
        'auc': auc,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

def save_checkpoint(model, optimizer, epoch, metrics, filename):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename):
    """加载模型检查点"""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['metrics']

def save_predictions(images, labels, preds, save_dir, prefix=""):
    """保存预测结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(len(images)):
        # 转换为PIL图像
        img = tensor_to_image(images[i])
        label = tensor_to_image(labels[i], is_binary=True)
        pred = tensor_to_image(preds[i], is_binary=True, threshold=0.5)
        
        # 保存图像
        img.save(os.path.join(save_dir, f"{prefix}_img_{i}.png"))
        label.save(os.path.join(save_dir, f"{prefix}_label_{i}.png"))
        pred.save(os.path.join(save_dir, f"{prefix}_pred_{i}.png"))

def tensor_to_image(tensor, is_binary=False, threshold=0.5):
    """将张量转换为PIL图像"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # 移动到CPU并转换为numpy数组
    img_np = tensor.cpu().detach().numpy()
    
    if img_np.shape[0] == 3:  # RGB图像
        img_np = np.transpose(img_np, (1, 2, 0))
        # 反归一化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        img_np = (img_np * 255).astype(np.uint8)
    else:  # 单通道图像
        img_np = img_np.squeeze(0)
        if is_binary:
            img_np = (img_np > threshold).astype(np.uint8) * 255
        else:
            img_np = (img_np * 255).astype(np.uint8)
    
    return Image.fromarray(img_np)

def plot_training_curve(history, save_path):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 4))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Curve')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_curve(history, save_path):
    """绘制指标曲线"""
    plt.figure(figsize=(15, 10))
    
    metrics = ['sensitivity', 'specificity', 'accuracy', 'auc']
    titles = ['Sensitivity (SE)', 'Specificity (SP)', 'Accuracy (AC)', 'AUC']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        plt.subplot(2, 2, i + 1)
        plt.plot(history[f'train_{metric}'], label='Train')
        plt.plot(history[f'val_{metric}'], label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.title(f'Training {title} Curve')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def print_metrics_summary(metrics, prefix=""):
    """打印指标摘要"""
    summary = f"{prefix} Metrics Summary:\n"
    summary += f"  Sensitivity (SE): {metrics['sensitivity']:.4f}\n"
    summary += f"  Specificity (SP): {metrics['specificity']:.4f}\n"
    summary += f"  Accuracy (AC): {metrics['accuracy']:.4f}\n"
    summary += f"  AUC: {metrics['auc']:.4f}\n"
    summary += f"  TP: {metrics['tp']}, TN: {metrics['tn']}, FP: {metrics['fp']}, FN: {metrics['fn']}"
    return summary

def setup_device():
    """设置设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    return device

def clear_memory():
    """清理内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()