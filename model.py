import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUModule(nn.Module):
    """GRU模块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(GRUModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 更新门
        self.update_gate = nn.Conv2d(
            in_channels + out_channels, 
            out_channels, 
            kernel_size, 
            padding=padding
        )
        
        # 重置门
        self.reset_gate = nn.Conv2d(
            in_channels + out_channels, 
            out_channels, 
            kernel_size, 
            padding=padding
        )
        
        # 候选隐藏状态
        self.candidate = nn.Conv2d(
            in_channels + out_channels, 
            out_channels, 
            kernel_size, 
            padding=padding
        )
        
        # 批量归一化
        self.bn_update = nn.BatchNorm2d(out_channels)
        self.bn_reset = nn.BatchNorm2d(out_channels)
        self.bn_candidate = nn.BatchNorm2d(out_channels)
        self.bn_hidden = nn.BatchNorm2d(out_channels)
    
    def forward(self, x, h_prev):
        """
        x: 当前输入特征图 (B, C_in, H, W)
        h_prev: 上一时刻隐藏状态 (B, C_out, H, W)
        """
        # 连接输入和隐藏状态
        combined = torch.cat([x, h_prev], dim=1)
        
        # 更新门
        z = torch.sigmoid(self.bn_update(self.update_gate(combined)))
        
        # 重置门
        r = torch.sigmoid(self.bn_reset(self.reset_gate(combined)))
        
        # 候选隐藏状态
        combined_candidate = torch.cat([x, r * h_prev], dim=1)
        h_candidate = torch.tanh(self.bn_candidate(self.candidate(combined_candidate)))
        
        # 更新隐藏状态
        h_new = (1 - z) * h_prev + z * h_candidate
        h_new = self.bn_hidden(h_new)
        
        return h_new

class ResGRUBlock(nn.Module):
    """ResGRU块"""
    def __init__(self, in_channels, out_channels):
        super(ResGRUBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 第一个GRU模块
        self.gru1 = GRUModule(in_channels, out_channels)
        
        # 第二个GRU模块
        self.gru2 = GRUModule(out_channels, out_channels)
        
        # 残差连接调整
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()
        
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        x: 输入特征图 (B, C_in, H, W)
        """
        # 残差连接
        residual = self.residual(x)
        
        # 初始化隐藏状态
        batch_size, _, height, width = x.size()
        h1 = torch.zeros(batch_size, self.out_channels, height, width, device=x.device)
        
        # 第一个GRU模块
        h1 = self.gru1(x, h1)
        
        # 第二个GRU模块
        h2 = self.gru2(h1, h1)  # 使用h1作为初始隐藏状态
        
        # 残差连接 + ReLU
        out = self.relu(h2 + residual)
        
        return out

class DownBlock(nn.Module):
    """下采样块"""
    def __init__(self, in_channels, out_channels, num_resgru_blocks=2):
        super(DownBlock, self).__init__()
        
        # ResGRU块序列
        blocks = []
        blocks.append(ResGRUBlock(in_channels, out_channels))
        for _ in range(num_resgru_blocks - 1):
            blocks.append(ResGRUBlock(out_channels, out_channels))
        
        self.blocks = nn.Sequential(*blocks)
        
        # 下采样：最大池化 + 1x1卷积
        self.downsample = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        x: 输入特征图
        返回: (下采样特征图, 跳连接特征图)
        """
        # 特征提取
        out = self.blocks(x)
        
        # 保存跳连接特征
        skip = out
        
        # 下采样
        out = self.downsample(out)
        
        return out, skip

class UpBlock(nn.Module):
    """上采样块"""
    def __init__(self, in_channels, out_channels, num_resgru_blocks=2):
        super(UpBlock, self).__init__()
        
        # 上采样：反卷积
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # ResGRU块序列
        blocks = []
        blocks.append(ResGRUBlock(out_channels * 2, out_channels))  # 考虑跳连接
        for _ in range(num_resgru_blocks - 1):
            blocks.append(ResGRUBlock(out_channels, out_channels))
        
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x, skip):
        """
        x: 输入特征图
        skip: 跳连接特征图
        """
        # 上采样
        x = self.upsample(x)
        
        # 调整跳连接大小以匹配
        if x.size() != skip.size():
            skip = F.interpolate(skip, size=x.size()[2:], mode='bilinear', align_corners=True)
        
        # 融合跳连接
        out = torch.cat([x, skip], dim=1)
        
        # 特征提取
        out = self.blocks(out)
        
        return out

class ResGRUUNet(nn.Module):
    """ResGRUU-Net网络"""
    def __init__(self, in_channels=3, out_channels=1, init_features=32):  # 初始特征数从64减少到32
        super(ResGRUUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_features = init_features
        
        # 减少下采样路径从4个到3个
        self.down1 = DownBlock(in_channels, init_features, num_resgru_blocks=1)  # 减少ResGRU块数量
        self.down2 = DownBlock(init_features, init_features * 2, num_resgru_blocks=1)
        self.down3 = DownBlock(init_features * 2, init_features * 4, num_resgru_blocks=2)
        
        # 简化瓶颈层
        self.bottleneck = ResGRUBlock(init_features * 4, init_features * 8)
        
        # 减少上采样路径从4个到3个
        self.up1 = UpBlock(init_features * 8, init_features * 4, num_resgru_blocks=2)
        self.up2 = UpBlock(init_features * 4, init_features * 2, num_resgru_blocks=1)
        self.up3 = UpBlock(init_features * 2, init_features, num_resgru_blocks=1)
        
        # 输出层保持不变
        self.output_layer = nn.Sequential(
            nn.Conv2d(init_features, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        x: 输入图像 (B, C, H, W)
        """
        # 调整下采样路径
        x1, skip1 = self.down1(x)
        x2, skip2 = self.down2(x1)
        x3, skip3 = self.down3(x2)
        
        # 简化瓶颈层前向传播
        x = self.bottleneck(x3)
        
        # 调整上采样路径
        x = self.up1(x, skip3)
        x = self.up2(x, skip2)
        x = self.up3(x, skip1)
        
        # 输出
        out = self.output_layer(x)
        
        return out

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # 测试模型
    model = ResGRUUNet()
    print(f"模型参数量: {count_parameters(model):,}")
    
    # 测试前向传播
    x = torch.randn(2, 3, 48, 48)
    out = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")