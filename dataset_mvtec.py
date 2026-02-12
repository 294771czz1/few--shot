#!/usr/bin/env python3
"""
MVTec AD数据集加载器
MVTec包含15个类别，每个类别有多种异常类型
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class MVTecDataset(Dataset):
    """
    MVTec AD数据集加载器
    
    数据结构:
    root/
        bottle/
            test/
                broken_large/
                    000.png
                    ...
                good/
                    000.png
                    ...
            ground_truth/
                broken_large/
                    000_mask.png
                    ...
    """
    
    def __init__(self, root, categories=None, split='test', transform=None, target_size=(512, 512)):
        """
        Args:
            root: MVTec数据集根目录
            categories: 要使用的类别列表，None表示使用所有类别
            split: 'test' (MVTec只有test集)
            transform: 图像变换
            target_size: 目标尺寸
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_size = target_size
        
        # MVTec的15个类别
        all_categories = [
            'bottle', 'cable', 'capsule', 'carpet', 'grid',
            'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
            'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
        ]
        
        self.categories = categories if categories is not None else all_categories
        
        # 加载数据
        self.samples = []
        self._load_dataset()
        
        print(f"MVTec数据集加载完成:")
        print(f"  类别数: {len(self.categories)}")
        print(f"  样本数: {len(self.samples)}")
    
    def _load_dataset(self):
        """加载数据集"""
        for category in self.categories:
            category_dir = self.root / category / self.split
            gt_dir = self.root / category / 'ground_truth'
            
            if not category_dir.exists():
                print(f"警告: {category_dir} 不存在，跳过")
                continue
            
            # 遍历异常类型
            for defect_type in category_dir.iterdir():
                if not defect_type.is_dir():
                    continue
                
                defect_name = defect_type.name
                
                # 遍历图像
                for img_path in sorted(defect_type.glob('*.png')):
                    # 确定mask路径
                    if defect_name == 'good':
                        # 正常样本没有mask，创建全0 mask
                        mask_path = None
                    else:
                        # 异常样本有mask，文件名一一对应
                        # 图像: 006.png -> mask: 006_mask.png
                        mask_name = img_path.stem + '_mask.png'
                        mask_path = gt_dir / defect_name / mask_name
                        
                        # 文件名应该一一对应，如果不存在说明数据有问题
                        if not mask_path.exists():
                            print(f"警告: mask不存在，跳过 {img_path.name} -> {mask_path}")
                            continue
                    
                    self.samples.append({
                        'image': img_path,
                        'mask': mask_path,
                        'category': category,
                        'defect_type': defect_name
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图像
        image = Image.open(sample['image']).convert('RGB')
        
        # 加载mask
        if sample['mask'] is not None:
            mask = Image.open(sample['mask']).convert('L')
            # 二值化: 0-背景, 1-前景(异常)
            mask = np.array(mask)
            # 自动检测mask值范围来二值化 (与训练时保持一致)
            if mask.max() > 1:
                # 0-255范围，使用127阈值
                mask = (mask > 127).astype(np.float32)
            else:
                # 0-1范围（已经是二值），直接使用
                mask = mask.astype(np.float32)
        else:
            # 正常样本，全0 mask
            mask = np.zeros((image.size[1], image.size[0]), dtype=np.float32)
        
        # 转换为tensor
        if self.transform is None:
            # 默认变换: ToTensor (不进行归一化，由Dataset Wrapper处理)
            image = T.ToTensor()(image)
            mask = torch.from_numpy(mask)
        else:
            image = self.transform(image)
            mask = torch.from_numpy(mask)
        
        # 调整尺寸
        if image.shape[-2:] != self.target_size:
            image = T.Resize(self.target_size)(image)
        
        if mask.shape[-2:] != self.target_size:
            mask = mask.unsqueeze(0)  # [H, W] -> [1, H, W]
            mask = T.Resize(self.target_size, interpolation=T.InterpolationMode.NEAREST)(mask)
            mask = mask.squeeze(0)  # [1, H, W] -> [H, W]
        
        # 返回信息字典 (直接返回图像路径)
        info = str(sample['image'])
        
        return image, mask, info


class MVTecFewShotSampler:
    """MVTec数据集的Few-shot采样器 - 每个缺陷类型采样k个"""
    
    def __init__(self, dataset):
        self.dataset = dataset
        self._analyze_dataset()
    
    def _analyze_dataset(self):
        """分析数据集，按缺陷类型分组"""
        print("正在分析MVTec数据集...")
        
        # 按 (category, defect_type) 分组
        from collections import defaultdict
        self.defect_groups = defaultdict(list)  # {(category, defect_type): [(idx, fg_pixels), ...]}
        self.normal_samples = []  # 正常样本
        
        for idx in range(len(self.dataset)):
            sample = self.dataset.samples[idx]
            img, mask, _ = self.dataset[idx]
            
            mask_np = mask.numpy() if torch.is_tensor(mask) else np.array(mask)
            foreground_pixels = np.sum(mask_np > 0)
            
            if sample['defect_type'] == 'good':
                # 正常样本
                self.normal_samples.append(idx)
            else:
                # 缺陷样本，按类型分组
                key = (sample['category'], sample['defect_type'])
                self.defect_groups[key].append((idx, foreground_pixels))
        
        # 对每组内的样本按前景像素数排序（降序）
        for key in self.defect_groups:
            self.defect_groups[key].sort(key=lambda x: x[1], reverse=True)
        
        print(f"  找到 {len(self.defect_groups)} 个缺陷类型")
        print(f"  找到 {len(self.normal_samples)} 个正常样本")
        
        # 统计每个缺陷类型的样本数
        defect_counts = {key: len(samples) for key, samples in self.defect_groups.items()}
        min_count = min(defect_counts.values()) if defect_counts else 0
        max_count = max(defect_counts.values()) if defect_counts else 0
        print(f"  每个缺陷类型的样本数范围: {min_count} ~ {max_count}")
    
    def sample_k_shot(self, k_shot, strategy='top', include_normal=False):
        """
        采样k-shot样本 - 每个缺陷类型采样k个样本
        
        Args:
            k_shot: 每个缺陷类型采样的数量
            strategy: 'top' 选择前景最多的, 'diverse' 均匀分布采样
            include_normal: 是否包含正常样本（每个物体类别采样k个）
        
        Returns:
            selected: 选中的样本索引列表
        """
        selected = []
        
        print(f"\n采样策略: 每个缺陷类型采样 {k_shot} 个样本")
        
        # 对每个缺陷类型采样
        for (category, defect_type), samples in self.defect_groups.items():
            if strategy == 'top':
                # 选择前景像素最多的k个
                k = min(k_shot, len(samples))
                selected_from_group = [idx for idx, _ in samples[:k]]
            else:
                # 均匀分布采样
                k = min(k_shot, len(samples))
                step = len(samples) // k if k > 0 else 1
                selected_from_group = [samples[i * step][0] for i in range(k)]
            
            selected.extend(selected_from_group)
            
            if len(selected) <= 50:  # 只显示前50个
                print(f"  {category}/{defect_type}: 采样 {len(selected_from_group)} 个样本")
        
        if len(self.defect_groups) > 10:
            print(f"  ... (共 {len(self.defect_groups)} 个缺陷类型)")
        
        # 如果包含正常样本
        if include_normal and len(self.normal_samples) > 0:
            # 按物体类别分组正常样本
            normal_by_category = {}
            for idx in self.normal_samples:
                cat = self.dataset.samples[idx]['category']
                if cat not in normal_by_category:
                    normal_by_category[cat] = []
                normal_by_category[cat].append(idx)
            
            # 每个类别采样k个正常样本
            for cat, samples in normal_by_category.items():
                k = min(k_shot, len(samples))
                selected.extend(samples[:k])
                print(f"  {cat}/good: 采样 {k} 个正常样本")
        
        print(f"\n总计采样: {len(selected)} 个样本")
        print(f"  - {len(self.defect_groups)} 个缺陷类型 × {k_shot} = {len(self.defect_groups) * k_shot} (理论值)")
        if include_normal:
            print(f"  - 正常样本: {len(selected) - len(self.defect_groups) * k_shot}")
        
        return selected


if __name__ == '__main__':
    # 测试代码
    dataset = MVTecDataset(
        root='./segdata/mvtec',
        categories=['bottle'],  # 只测试bottle类别
        split='test'
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    
    # 测试采样器
    sampler = MVTecFewShotSampler(dataset)
    selected = sampler.sample_k_shot(k_shot=5, strategy='top')
    
    print(f"\n选中的样本索引: {selected}")
