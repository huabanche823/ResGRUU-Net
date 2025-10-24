import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class DRIVEDataset(Dataset):
    """DRIVE数据集加载器"""
    def __init__(self, data_dir, train=True, transform=None, augment=True):
        self.data_dir = data_dir
        self.train = train
        self.transform = transform
        self.augment = augment
        
        # 数据集路径
        if train:
            self.image_dir = os.path.join(data_dir, 'training', 'images')
            self.label_dir = os.path.join(data_dir, 'training', '1st_manual')
            self.mask_dir = os.path.join(data_dir, 'training', 'mask')
        else:
            self.image_dir = os.path.join(data_dir, 'test', 'images')
            self.label_dir = os.path.join(data_dir, 'test', '1st_manual')
            self.mask_dir = os.path.join(data_dir, 'test', 'mask')
        
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.tif')])
        self.label_files = sorted([f for f in os.listdir(self.label_dir) if f.endswith('.gif')])
        
        # 验证文件数量匹配
        assert len(self.image_files) == len(self.label_files), "图像和标签数量不匹配"
        
        # 数据增强参数
        self.patch_size = 48
        self.num_patches_per_image = 9500  # 每幅图像提取的切片数量
    
    def __len__(self):
        if self.train and self.augment:
            return len(self.image_files) * self.num_patches_per_image
        return len(self.image_files)
    
    def __getitem__(self, idx):
        if self.train and self.augment:
            # 计算原始图像索引
            img_idx = idx // self.num_patches_per_image
            patch_idx = idx % self.num_patches_per_image
            
            image_path = os.path.join(self.image_dir, self.image_files[img_idx])
            label_path = os.path.join(self.label_dir, self.label_files[img_idx])
            # 使用图像文件名生成mask路径
            image_num = self.image_files[img_idx].split('_')[0]
            mask_filename = f"{image_num}_{'training' if self.train else 'test'}_mask.gif"
            mask_path = os.path.join(self.mask_dir, mask_filename)
            
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            label = Image.open(label_path).convert('L')
            mask = Image.open(mask_path).convert('L')
            
            # 随机选择切片方法
            if random.random() < 0.5:
                # 随机切片
                image_patch, label_patch = self.random_patch(image, label, mask)
            else:
                # Mosaic切片
                image_patch, label_patch = self.mosaic_patch(image, label, mask, img_idx)
            
            # 数据增强
            image_patch, label_patch = self.augment_patch(image_patch, label_patch)
            
        else:
            # 不增强时返回完整图像
            image_path = os.path.join(self.image_dir, self.image_files[idx])
            label_path = os.path.join(self.label_dir, self.label_files[idx])
            # 使用图像文件名生成mask路径
            image_num = self.image_files[idx].split('_')[0]
            mask_filename = f"{image_num}_{'training' if self.train else 'test'}_mask.gif"
            mask_path = os.path.join(self.mask_dir, mask_filename)
            
            image = Image.open(image_path).convert('RGB')
            label = Image.open(label_path).convert('L')
            mask = Image.open(mask_path).convert('L')
            
            # 调整图像大小以匹配模型输出尺寸 (584x560)
            image = image.resize((560, 584), Image.BILINEAR)
            label = label.resize((560, 584), Image.NEAREST)
            
            # 应用mask
            label = self.apply_mask(label, mask)
            
            image_patch, label_patch = image, label
        
        # 预处理
        if self.transform:
            image_patch = self.transform(image_patch)
            label_patch = transforms.ToTensor()(label_patch)
            label_patch = (label_patch > 0.5).float()  # 二值化
        
        return image_patch, label_patch
    
    def random_patch(self, image, label, mask):
        """随机切片"""
        w, h = image.size
        
        # 在mask区域内随机选择中心
        mask_np = np.array(mask)
        valid_coords = np.where(mask_np > 0)
        
        if len(valid_coords[0]) == 0:
            # 如果没有有效区域，使用整个图像
            return image.crop((0, 0, self.patch_size, self.patch_size)), \
                   label.crop((0, 0, self.patch_size, self.patch_size))
        
        # 随机选择有效区域内的中心
        center_x = random.choice(valid_coords[1])
        center_y = random.choice(valid_coords[0])
        
        # 计算切片坐标
        x1 = max(0, center_x - self.patch_size // 2)
        y1 = max(0, center_y - self.patch_size // 2)
        x2 = min(w, x1 + self.patch_size)
        y2 = min(h, y1 + self.patch_size)
        
        # 调整坐标确保切片大小
        if x2 - x1 < self.patch_size:
            x1 = max(0, x2 - self.patch_size)
        if y2 - y1 < self.patch_size:
            y1 = max(0, y2 - self.patch_size)
        
        # 裁剪
        image_patch = image.crop((x1, y1, x2, y2))
        label_patch = label.crop((x1, y1, x2, y2))
        
        return image_patch, label_patch
    
    def mosaic_patch(self, image, label, mask, img_idx):
        """Mosaic切片"""
        # 随机选择其他3幅图像
        other_indices = [i for i in range(len(self.image_files)) if i != img_idx]
        if len(other_indices) < 3:
            other_indices = other_indices * 3  # 确保有足够的图像
        
        selected_indices = random.sample(other_indices, 3)
        selected_indices.insert(0, img_idx)  # 包含当前图像
        
        # 获取4个随机切片
        patches = []
        for idx in selected_indices:
            img_path = os.path.join(self.image_dir, self.image_files[idx])
            lbl_path = os.path.join(self.label_dir, self.label_files[idx])
            image_num = self.image_files[idx].split('_')[0]
            mask_filename = f"{image_num}_{'training' if self.train else 'test'}_mask.gif"
            msk_path = os.path.join(self.mask_dir, mask_filename)
            
            img = Image.open(img_path).convert('RGB')
            lbl = Image.open(lbl_path).convert('L')
            msk = Image.open(msk_path).convert('L')
            
            patch_img, patch_lbl = self.random_patch(img, lbl, msk)
            patches.append((patch_img, patch_lbl))
        
        # 创建Mosaic图像
        mosaic_img = Image.new('RGB', (self.patch_size * 2, self.patch_size * 2))
        mosaic_lbl = Image.new('L', (self.patch_size * 2, self.patch_size * 2))
        
        # 放置4个切片
        positions = [(0, 0), (self.patch_size, 0), 
                    (0, self.patch_size), (self.patch_size, self.patch_size)]
        
        for i, ((img, lbl), (x, y)) in enumerate(zip(patches, positions)):
            # 随机变换
            img, lbl = self.random_transform(img, lbl)
            mosaic_img.paste(img, (x, y))
            mosaic_lbl.paste(lbl, (x, y))
        
        # 随机裁剪到48x48
        x = random.randint(0, self.patch_size)
        y = random.randint(0, self.patch_size)
        
        mosaic_img = mosaic_img.crop((x, y, x + self.patch_size, y + self.patch_size))
        mosaic_lbl = mosaic_lbl.crop((x, y, x + self.patch_size, y + self.patch_size))
        
        return mosaic_img, mosaic_lbl
    
    def augment_patch(self, image, label):
        """图像增强"""
        # 随机水平翻转
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        # 随机垂直翻转
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            label = label.transpose(Image.FLIP_TOP_BOTTOM)
        
        # 随机旋转
        if random.random() < 0.5:
            angle = random.choice([90, 180, 270])
            image = image.rotate(angle)
            label = label.rotate(angle)
        
        # 随机亮度调整
        if random.random() < 0.3:
            enhancer = ImageEnhance.Brightness(image)
            factor = random.uniform(0.7, 1.3)
            image = enhancer.enhance(factor)
        
        # 随机对比度调整
        if random.random() < 0.3:
            enhancer = ImageEnhance.Contrast(image)
            factor = random.uniform(0.7, 1.3)
            image = enhancer.enhance(factor)
        
        return image, label
    
    def random_transform(self, image, label):
        """随机变换"""
        # 随机翻转
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        # 随机缩放
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            new_size = (int(self.patch_size * scale), int(self.patch_size * scale))
            image = image.resize(new_size, Image.BILINEAR)
            label = label.resize(new_size, Image.NEAREST)
            
            # 重新裁剪到48x48
            if scale > 1.0:
                x = random.randint(0, new_size[0] - self.patch_size)
                y = random.randint(0, new_size[1] - self.patch_size)
                image = image.crop((x, y, x + self.patch_size, y + self.patch_size))
                label = label.crop((x, y, x + self.patch_size, y + self.patch_size))
            else:
                # 填充
                new_img = Image.new('RGB', (self.patch_size, self.patch_size), (0, 0, 0))
                new_lbl = Image.new('L', (self.patch_size, self.patch_size), 0)
                x = (self.patch_size - new_size[0]) // 2
                y = (self.patch_size - new_size[1]) // 2
                new_img.paste(image, (x, y))
                new_lbl.paste(label, (x, y))
                image, label = new_img, new_lbl
        
        return image, label
    
    def apply_mask(self, label, mask):
        """应用mask"""
        label_np = np.array(label)
        mask_np = np.array(mask)
        label_np[mask_np == 0] = 0
        return Image.fromarray(label_np)

def create_dataloaders(data_dir, batch_size=32, num_workers=4):
    """创建数据加载器"""
    # 预处理变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 创建数据集
    train_dataset = DRIVEDataset(data_dir, train=True, transform=transform, augment=True)
    val_dataset = DRIVEDataset(data_dir, train=True, transform=transform, augment=False)
    test_dataset = DRIVEDataset(data_dir, train=False, transform=transform, augment=False)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # 完整图像验证
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # 完整图像测试
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader