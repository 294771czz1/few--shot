#!/usr/bin/env python3
"""
TN3K数据集的Few-shot学习脚本
基于改进的采样策略和增强技术
"""

import os
import sys
import argparse
import random
from pathlib import Path
import numpy as np
from collections import defaultdict
import shutil
import cv2
from scipy.ndimage import distance_transform_edt
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from dataset import FolderDataset
from dataset_mvtec import MVTecDataset, MVTecFewShotSampler
from dpt import DPT
from dpt_enhanced import DPTEnhanced


def set_seed(seed=42):
    """固定随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🎲 随机种子已设置为: {seed}")


def compute_dice_coefficient(pred, target, smooth=1e-6):
    """
    计算 Dice 系数
    
    Args:
        pred: 预测结果 (torch.Tensor or np.ndarray)
        target: 真实标签 (torch.Tensor or np.ndarray)
        smooth: 平滑项，防止除零
    
    Returns:
        dice: Dice系数 (float)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = np.sum(pred * target)
    dice = (2. * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)
    
    return dice


def compute_hd95(pred, target, voxel_spacing=(1.0, 1.0)):
    """
    计算 Hausdorff Distance 95% (HD95)
    
    Args:
        pred: 预测结果 (np.ndarray), shape [H, W]
        target: 真实标签 (np.ndarray), shape [H, W]
        voxel_spacing: 像素间距 (tuple)
    
    Returns:
        hd95: HD95距离 (float), 如果无法计算返回 np.inf
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    # 如果预测或真实标签全0或全1，无法计算HD
    if pred.sum() == 0 or target.sum() == 0:
        return np.inf
    if pred.sum() == pred.size or target.sum() == target.size:
        return np.inf
    
    # 计算边界
    pred_border = pred ^ cv2.erode(pred.astype(np.uint8), np.ones((3,3), np.uint8), iterations=1).astype(bool)
    target_border = target ^ cv2.erode(target.astype(np.uint8), np.ones((3,3), np.uint8), iterations=1).astype(bool)
    
    if pred_border.sum() == 0 or target_border.sum() == 0:
        return np.inf
    
    # 计算距离变换
    dt_pred = distance_transform_edt(~pred_border, sampling=voxel_spacing)
    dt_target = distance_transform_edt(~target_border, sampling=voxel_spacing)
    
    # 计算从预测边界到真实边界的距
    distances_pred_to_target = dt_target[pred_border]
    # 计算从真实边界到预测边界的距
    distances_target_to_pred = dt_pred[target_border]
    
    # 合并所有距
    all_distances = np.concatenate([distances_pred_to_target, distances_target_to_pred])
    
    if len(all_distances) == 0:
        return np.inf
    
    # 计算95百分位数
    hd95 = np.percentile(all_distances, 95)
    
    return hd95


# ===== ViSA Dataset =====
class ViSADataset(torch.utils.data.Dataset):
    """
    ViSA Dataset for Few-shot Learning
    
    Dataset structure:
    visa/
      ├── split_csv/
      │  └── 2cls_fewshot.csv
      ├── candle/
      │  └── Data/
      
      ├── Images/
      
      │  ├── Normal/
      
      │  └── Anomaly/
      
      └── Masks/
      
          └── Anomaly/  (Mask值 0-6, >0即为前景)
      └── capsules/
          └── ...
    """
    
    def __init__(self, root, csv_file, split='train', category=None, transform=None, target_size=(512, 512)):
        """
        Args:
            root: ViSA数据集根目录
            csv_file: CSV文件路径 (e.g. 'split_csv/2cls_fewshot.csv')
            split: 'train' or 'test'
            category: 特定类别 (e.g. 'candle'), None表示所有类别
            transform: 数据增强
            target_size: 目标尺寸 (H, W)
        """
        self.root = root
        self.split = split
        self.category = category
        self.transform = transform
        self.target_size = target_size
        
        # 加载CSV
        csv_path = os.path.join(root, csv_file)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # 过滤split
        df = df[df['split'] == split]
        
        # 过滤category
        if category is not None:
            df = df[df['object'] == category]
        
        # 收集样本
        self.samples = []
        for _, row in df.iterrows():
            img_path = os.path.join(root, row['image'])
            
            if not os.path.isfile(img_path):
                continue
            
            # 处理mask路径
            if pd.isna(row['mask']) or row['mask'] == '':
                mask_path = None  # 正常样本无mask
            else:
                mask_path = os.path.join(root, row['mask'])
                if not os.path.isfile(mask_path):
                    # 尝试.png扩展
                    mask_path_png = os.path.splitext(mask_path)[0] + '.png'
                    if os.path.isfile(mask_path_png):
                        mask_path = mask_path_png
                    else:
                        continue
            
            self.samples.append({
                'image': img_path,
                'mask': mask_path,
                'label': row['label'],  # 'normal' or 'anomaly'
                'object': row['object'],
            })
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found for {category}/{split}!")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 读取图像
        image = Image.open(sample['image']).convert('RGB')
        
        # 读取mask
        if sample['mask'] is not None:
            mask = Image.open(sample['mask'])
            mask_np = np.array(mask)
            # ViSA的mask值是0-6，>0即为前景 (转换为二值 0=背景, 1=前景)
            mask_binary = (mask_np > 0).astype(np.uint8)
        else:
            # 正常样本数 mask
            mask_binary = np.zeros(image.size[::-1], dtype=np.uint8)
        
        # Resize到目标尺
        image = image.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
        mask_img = Image.fromarray(mask_binary)
        mask_img = mask_img.resize((self.target_size[1], self.target_size[0]), Image.NEAREST)
        mask_binary = np.array(mask_img)
        
        # 转换为tensor
        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)
        
        mask_tensor = torch.from_numpy(mask_binary).long()
        
        # 返回3个值以保持一致(image, mask, label)
        # label: 0=normal, 1=anomaly
        label = 1 if sample['label'] == 'anomaly' else 0
        
        return image, mask_tensor, label


class FewShotSamplerTN3K:
    """TN3K数据集的Few-shot采样"""
    
    def __init__(self, dataset):
        self.dataset = dataset
        self._analyze_dataset()
    
    def _analyze_dataset(self):
        """分析数据集，按照前景像素数排"""
        print("正在分析TN3K数据集..")
        
        self.samples_with_target = []  # 存储 (idx, foreground_pixel_count)
        
        for idx in range(len(self.dataset)):
            img, mask, _ = self.dataset[idx]  # FolderDataset返回 (img_tensor, mask_tensor, _)
            
            # FolderDataset在没有transform时会返回已归一化的tensor
            if torch.is_tensor(mask):
                # mask已经是二值化的[C, H, W] 或 [H, W]
                if mask.dim() == 3:
                    mask = mask.squeeze(0)
                foreground_pixels = mask.sum().item()
            elif isinstance(mask, np.ndarray):
                # 原始numpy数组
                foreground_pixels = np.sum(mask > 127)
            else:
                foreground_pixels = 0
            
            if foreground_pixels > 0:
                self.samples_with_target.append((idx, int(foreground_pixels)))
        
        # 按前景像素数降序排序
        self.samples_with_target.sort(key=lambda x: x[1], reverse=True)
        
        print(f"  找到 {len(self.samples_with_target)} 个包含目标的样本")
        if len(self.samples_with_target) > 0:
            print(f"  前景像素数范围 {self.samples_with_target[-1][1]} ~ {self.samples_with_target[0][1]}")
    
    def sample_k_shot(self, k_shot, strategy='top'):
        """
        采样k-shot样本
        
        Args:
            k_shot: 采样数量
            strategy: 'top' 选择前景最多的, 'diverse' 均匀分布采样
        """
        if strategy == 'top':
            # 选择前景像素最多的k个样
            selected = [idx for idx, _ in self.samples_with_target[:k_shot]]
            print(f"\n采样策略: 选择前景像素最多的 {k_shot} 个样本")
        else:
            # 均匀分布采样
            step = len(self.samples_with_target) // k_shot
            selected = [self.samples_with_target[i * step][0] for i in range(k_shot)]
            print(f"\n采样策略: 均匀分布采样 {k_shot} 个样本")
        
        # 打印选中样本的信
        print(f"选中的样本索引和前景像素数")
        for idx in selected[:5]:  # 只显示前5
            pixel_count = next(cnt for i, cnt in self.samples_with_target if i == idx)
            print(f"  样本 {idx}: {pixel_count} 前景像素")
        if len(selected) > 5:
            print(f"  ... (共{len(selected)} 个样本")
        
        return selected


class FewShotSamplerViSA:
    """ViSA数据集的Few-shot采样"""
    
    def __init__(self, dataset):
        self.dataset = dataset
        self._analyze_dataset()
    
    def _analyze_dataset(self):
        """分析数据集，按照前景像素数排"""
        print("正在分析ViSA数据集..")
        
        self.anomaly_samples = []  # (idx, foreground_pixel_count)
        self.normal_samples = []   # idx
        
        for idx in range(len(self.dataset)):
            sample = self.dataset.samples[idx]
            
            if sample['label'] == 'anomaly':
                # 检查是否真的有异常区域
                if sample['mask'] is not None:
                    try:
                        _, mask, _ = self.dataset[idx]
                        foreground_pixels = mask.sum().item()
                        if foreground_pixels > 0:
                            self.anomaly_samples.append((idx, foreground_pixels))
                    except:
                        pass
            elif sample['label'] == 'normal':
                self.normal_samples.append(idx)
        
        # 按前景像素数降序排序
        self.anomaly_samples.sort(key=lambda x: x[1], reverse=True)
        
        print(f"  找到 {len(self.anomaly_samples)} 个异常样本")
        print(f"  找到 {len(self.normal_samples)} 个正常样本")
        if self.anomaly_samples:
            print(f"  异常前景像素数范围 {self.anomaly_samples[-1][1]} ~ {self.anomaly_samples[0][1]}")
    
    def sample_k_shot(self, k_shot, include_normal=False, strategy='top'):
        """
        采样k-shot样本
        
        Args:
            k_shot: 采样数量
            include_normal: 是否包含正常样本
            strategy: 'top' 选择前景最多的, 'diverse' 均匀分布采样
        """
        selected = []
        
        # 采样异常样本
        if strategy == 'top':
            # 选择前景像素最多的k个异常样
            anomaly_indices = [idx for idx, _ in self.anomaly_samples[:k_shot]]
            print(f"\n采样策略: 选择前景像素最多的 {k_shot} 个异常样本")
        else:
            # 均匀分布采样
            step = len(self.anomaly_samples) // k_shot
            anomaly_indices = [self.anomaly_samples[i * step][0] for i in range(k_shot)]
            print(f"\n采样策略: 均匀分布采样 {k_shot} 个异常样本")
        
        selected.extend(anomaly_indices)
        
        # 可选：添加正常样本
        if include_normal and self.normal_samples:
            normal_k = min(k_shot, len(self.normal_samples))
            normal_indices = random.sample(self.normal_samples, normal_k)
            selected.extend(normal_indices)
            print(f"  额外添加 {normal_k} 个正常样本")
        
        # 打印选中样本信息
        print(f"选中的异常样本索引和前景像素数")
        for idx in anomaly_indices[:5]:
            pixel_count = next(cnt for i, cnt in self.anomaly_samples if i == idx)
            print(f"  样本 {idx}: {pixel_count} 前景像素")
        if len(anomaly_indices) > 5:
            print(f"  ... (共{len(anomaly_indices)} 个异常样本")
        
        print(f"✅ ViSA采样完成: 共选中 {len(selected)} 个样本")
        
        return selected


class TN3KTestDataset(Dataset):
    """TN3K测试集的Wrapper，处理没有transform的原始数"""
    
    def __init__(self, base_dataset, target_size=(512, 512)):
        self.base_dataset = base_dataset
        self.target_size = target_size
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        image, mask, label = self.base_dataset[idx]
        
        # 处理图像
        if not torch.is_tensor(image):
            # numpy数组: BGR uint8 -> RGB float tensor [C,H,W]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            image_pil = image_pil.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
            image = T.ToTensor()(image_pil)
            image = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        else:
            # 已经是归一化的tensor，但没有标准
            if image.shape[-2:] != self.target_size:
                image = F.interpolate(image.unsqueeze(0), size=self.target_size, 
                                    mode='bilinear', align_corners=False).squeeze(0)
            image = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        
        # 处理mask
        if not torch.is_tensor(mask):
            # numpy数组
            mask_pil = Image.fromarray(mask)
            mask_pil = mask_pil.resize((self.target_size[1], self.target_size[0]), Image.NEAREST)
            mask_np = np.array(mask_pil)
            mask_binary = (mask_np > 127).astype(np.float32)
            mask = torch.from_numpy(mask_binary)
        else:
            # 已经是tensor
            if mask.dim() == 3:
                mask = mask.squeeze(0)
            if mask.shape[-2:] != self.target_size:
                mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), 
                                   size=self.target_size,
                                   mode='nearest').squeeze(0).squeeze(0)
        
        return image, mask, label


class DSIFNDataset(Dataset):
    """
    DSIFN遥感变化检测数据集 - 作为异常检测数据集使用
    
    数据结构
    train/val:
        - t1/: 正常图像 (512×512 JPG)
        - t2/: 异常图像 (512×512 JPG) 
        - mask/: t1标签 (512×512 PNG) - 不使
        - mask_256/: t2标签 (256×256 RGB PNG) - 使用此作为异常标
    
    test:
        - t1/: 正常图像 (512×512 JPG)
        - t2/: 异常图像 (512×512 JPG)
        - mask/: 共用标签 (512×512 PNG) - 使用此作为异常标
    
    异常检测范式：
    - t2 作为输入（异常图像）
    - mask_256 (train/val) 或 mask (test) 作为异常区域标注
    """
    
    def __init__(self, root, split='train', target_size=(512, 512), transform=None):
        """
        Args:
            root: 数据集根目录 (如 /home/czz/segdino/segdata/DSIFN)
            split: 'train', 'val', 或 'test'
            target_size: 输出图像尺寸
            transform: 图像变换
        """
        self.root = Path(root)
        self.split = split
        self.target_size = target_size
        self.transform = transform
        
        # 构建路径
        split_dir = self.root / split
        self.t1_dir = split_dir / 't1'  # 正常图像（不用于训练
        self.t2_dir = split_dir / 't2'  # 异常图像（用于训练）
        
        # 根据split选择正确的mask目录
        if split in ['train', 'val']:
            # train/val使用mask_256 (t2对应的256×256标签)
            self.mask_dir = split_dir / 'mask_256'
            self.is_train_val = True
        else:  # test
            # test使用mask (512×512标签)
            self.mask_dir = split_dir / 'mask'
            self.is_train_val = False
        
        # 获取所有样
        self.samples = []
        if self.t2_dir.exists():  # 主要使用t2（异常图像）
            for t2_path in sorted(self.t2_dir.glob('*.jpg')):
                sample_id = t2_path.stem
                mask_path = self.mask_dir / f'{sample_id}.png'
                
                if mask_path.exists():
                    self.samples.append({
                        't2_anomaly': str(t2_path),    # 异常图像（用于训练）
                        'mask': str(mask_path),        # 异常标注
                        'id': sample_id,
                        'is_train_val': self.is_train_val
                    })
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {split_dir}")
        
        mask_info = "mask_256 (256×256)" if self.is_train_val else "mask (512×512)"
        print(f"📊 DSIFN {split}: 加载 {len(self.samples)} 个样本")
        print(f"   📌 使用 t2 作为异常图像")
        print(f"   📌 使用 {mask_info} 作为异常标注")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 读取异常图像 (t2)
        t2_anomaly = Image.open(sample['t2_anomaly']).convert('RGB')
        
        # 读取异常区域标注
        mask = Image.open(sample['mask'])
        
        # 处理mask格式
        if mask.mode == 'RGB':
            # RGB格式（train/val的mask_256
            mask_np = np.array(mask)[:, :, 0]  # 取第一通道
        else:
            # 灰度格式（test的mask
            mask_np = np.array(mask)
        
        # 二值化: 0=正常, 1=异常
        mask_binary = (mask_np > 127).astype(np.uint8)
        
        # Resize到目标尺寸(统一512×512)
        t2_anomaly = t2_anomaly.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
        
        # mask也resize到512×512 (如果是mask_256则需要放大）
        mask_img = Image.fromarray(mask_binary)
        mask_img = mask_img.resize((self.target_size[1], self.target_size[0]), Image.NEAREST)
        mask_binary = np.array(mask_img)
        
        # 转换为tensor
        if self.transform:
            t2_anomaly = self.transform(t2_anomaly)
        else:
            t2_anomaly = T.ToTensor()(t2_anomaly)
        
        mask_tensor = torch.from_numpy(mask_binary).long()
        
        # 返回: (异常图像, 异常标注, 标签)
        # 标签1表示这是异常样本（用于异常检测任务）
        return t2_anomaly, mask_tensor, 1


class FewShotSamplerDSIFN:
    """
    DSIFN数据集的Few-shot采样器（异常检测模式）
    
    采样策略
    - 从异常样本（t2图像）中选择异常区域最显著的k个样
    - 这些样本用于训练模型识别异常/变化模式
    """
    
    def __init__(self, dataset):
        self.dataset = dataset
        self._analyze_dataset()
    
    def _analyze_dataset(self):
        """分析数据集，按照异常像素数排"""
        print("正在分析DSIFN数据集（异常检测模式）...")
        
        self.anomaly_samples = []  # 存储 (idx, anomaly_pixel_count)
        
        for idx in range(len(self.dataset)):
            _, mask, _ = self.dataset[idx]
            
            if torch.is_tensor(mask):
                anomaly_pixels = mask.sum().item()
            elif isinstance(mask, np.ndarray):
                anomaly_pixels = np.sum(mask > 0)
            else:
                anomaly_pixels = 0
            
            if anomaly_pixels > 0:
                self.anomaly_samples.append((idx, int(anomaly_pixels)))
        
        # 按异常像素数降序排序
        self.anomaly_samples.sort(key=lambda x: x[1], reverse=True)
        
        print(f"  ✅ 找到 {len(self.anomaly_samples)} 个包含异常的样本")
        if len(self.anomaly_samples) > 0:
            print(f"  📊 异常像素数范围 {self.anomaly_samples[-1][1]} ~ {self.anomaly_samples[0][1]}")
    
    def sample_k_shot(self, k_shot, strategy='top'):
        """
        采样k-shot异常样本
        
        Args:
            k_shot: 采样数量
            strategy: 'top' 选择异常最显著的 'diverse' 均匀分布采样
        """
        if strategy == 'top':
            selected = [idx for idx, _ in self.anomaly_samples[:k_shot]]
            print(f"\n🎯 采样策略: 选择异常区域最显著的{k_shot} 个样本")
        else:
            # 均匀分布采样
            step = len(self.anomaly_samples) // k_shot
            selected = [self.anomaly_samples[i * step][0] for i in range(k_shot)]
            print(f"\n🎯 采样策略: 均匀分布采样 {k_shot} 个样本")
        
        # 打印选中样本信息
        print(f"选中的异常样本索引和前景像素数")
        for idx in selected[:5]:
            pixel_count = next(cnt for i, cnt in self.anomaly_samples if i == idx)
            print(f"  样本 {idx}: {pixel_count} 前景像素")
        if len(selected) > 5:
            print(f"  ... (共{len(selected)} 个样本")
        
        return selected


class MassachusettsRoadsDataset(Dataset):
    """
    Massachusetts Roads 遥感道路分割数据
    
    数据结构
    - data/: 原始遥感图像 (1500×1500 TIFF, RGB)
    - label/: 道路标签 (1500×1500 TIFF, 二值/255)
    
    道路分割任务
    - 输入: 遥感图像
    - 输出: 道路区域分割 (0=背景, 1=道路)
    - 道路占比3-5%，类别不平衡
    """
    
    def __init__(self, root, split='train', train_ratio=0.8, target_size=(512, 512), 
                 transform=None, seed=42):
        """
        Args:
            root: 数据集根目录 (如 /home/czz/segdino/segdata/Massachusetts Roads)
            split: 'train' 或 'test'
            train_ratio: 训练集比例(默认0.8，即49张图像39张训练，10张测试)
            target_size: 输出图像尺寸
            transform: 图像变换
            seed: 随机种子
        """
        self.root = Path(root)
        self.split = split
        self.target_size = target_size
        self.transform = transform
        
        # 获取所有样本文
        data_dir = self.root / 'data'
        label_dir = self.root / 'label'
        
        # 收集所有配对的图像和标
        all_samples = []
        for img_path in sorted(data_dir.glob('*.tiff')):
            # 从 test_data_1.tiff 提取 1
            sample_id = img_path.stem.replace('test_data_', '')
            label_path = label_dir / f'test_label_{sample_id}.tiff'
            
            if label_path.exists():
                all_samples.append({
                    'image': str(img_path),
                    'label': str(label_path),
                    'id': sample_id
                })
        
        # 划分训练集和测试
        np.random.seed(seed)
        n_total = len(all_samples)
        n_train = int(n_total * train_ratio)
        
        indices = np.random.permutation(n_total)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        if split == 'train':
            self.samples = [all_samples[i] for i in train_indices]
        else:  # test
            self.samples = [all_samples[i] for i in test_indices]
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {split} split")
        
        print(f"📊 Massachusetts Roads {split}: 加载 {len(self.samples)} 个样本")
        print(f"   📐 图像尺寸: 1500×1500 → {target_size[0]}×{target_size[1]}")
        print(f"   🎯 任务: 道路分割 (二分类)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 读取TIFF图像
        image = Image.open(sample['image']).convert('RGB')
        label = Image.open(sample['label'])
        
        # 处理标签 (0/255 → 0/1)
        label_np = np.array(label)
        label_binary = (label_np > 127).astype(np.uint8)
        
        # Resize到目标尺
        image = image.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
        label_img = Image.fromarray(label_binary)
        label_img = label_img.resize((self.target_size[1], self.target_size[0]), Image.NEAREST)
        label_binary = np.array(label_img)
        
        # 转换为tensor
        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)
        
        label_tensor = torch.from_numpy(label_binary).long()
        
        # 返回: (图像, 标签, 类别)
        # 类别1表示道路分割任务
        return image, label_tensor, 1


class FewShotSamplerMassRoads:
    """
    Massachusetts Roads数据集的Few-shot采样
    
    采样策略
    - 从训练集中选择道路区域最显著的k个样
    - 这些样本用于训练模型识别道路模式
    """
    
    def __init__(self, dataset):
        self.dataset = dataset
        self._analyze_dataset()
    
    def _analyze_dataset(self):
        """分析数据集，按照道路像素数排"""
        print("正在分析Massachusetts Roads数据集..")
        
        self.road_samples = []  # 存储 (idx, road_pixel_count)
        
        for idx in range(len(self.dataset)):
            _, label, _ = self.dataset[idx]
            
            if torch.is_tensor(label):
                road_pixels = label.sum().item()
            elif isinstance(label, np.ndarray):
                road_pixels = np.sum(label > 0)
            else:
                road_pixels = 0
            
            if road_pixels > 0:
                self.road_samples.append((idx, int(road_pixels)))
        
        # 按道路像素数降序排序
        self.road_samples.sort(key=lambda x: x[1], reverse=True)
        
        print(f"  ✅ 找到 {len(self.road_samples)} 个包含道路的样本")
        if len(self.road_samples) > 0:
            print(f"  📊 道路像素数范围 {self.road_samples[-1][1]} ~ {self.road_samples[0][1]}")
    
    def sample_k_shot(self, k_shot, strategy='top'):
        """
        采样k-shot道路样本
        
        Args:
            k_shot: 采样数量
            strategy: 'top' 选择道路最显著的 'diverse' 均匀分布采样
        """
        if strategy == 'top':
            selected = [idx for idx, _ in self.road_samples[:k_shot]]
            print(f"\n🎯 采样策略: 选择道路区域最显著的{k_shot} 个样本")
        else:
            # 均匀分布采样
            step = len(self.road_samples) // k_shot
            selected = [self.road_samples[i * step][0] for i in range(k_shot)]
            print(f"\n🎯 采样策略: 均匀分布采样 {k_shot} 个样本")
        
        # 打印选中样本信息
        print(f"选中的道路样本索引和像素数")
        for idx in selected[:5]:
            pixel_count = next(cnt for i, cnt in self.road_samples if i == idx)
            print(f"  样本 {idx}: {pixel_count} 道路像素")
        if len(selected) > 5:
            print(f"  ... (共{len(selected)} 个样本")
        
        return selected


class SatelliteDataset(Dataset):
    """
    Satellite Dataset 遥感图像分割数据
    
    数据结构
    - image/: 遥感图像 (512×512 TIFF, RGB)
    - label/: 分割标签 (512×512 TIFF, RGB二值/255)
    
    遥感分割任务
    - 输入: 遥感图像
    - 输出: 区域分割 (0=背景, 1=前景)
    """
    
    def __init__(self, root, split='train', train_ratio=0.8, target_size=(512, 512), 
                 transform=None, seed=42):
        """
        Args:
            root: 数据集根目录 (如 /home/czz/segdino/segdata/Satellite dataset)
            split: 'train' 或 'test'
            train_ratio: 训练集比例(默认0.8)
            target_size: 输出图像尺寸
            transform: 图像变换
            seed: 随机种子
        """
        self.root = Path(root)
        self.split = split
        self.target_size = target_size
        self.transform = transform
        
        # 获取所有样本文
        image_dir = self.root / 'image'
        label_dir = self.root / 'label'
        
        # 收集所有配对的图像和标
        all_samples = []
        for img_path in sorted(image_dir.glob('*.tif')):
            sample_id = img_path.stem
            label_path = label_dir / f'{sample_id}.tif'
            
            if label_path.exists():
                all_samples.append({
                    'image': str(img_path),
                    'label': str(label_path),
                    'id': sample_id
                })
        
        # 划分训练集和测试
        np.random.seed(seed)
        n_total = len(all_samples)
        n_train = int(n_total * train_ratio)
        
        indices = np.random.permutation(n_total)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        if split == 'train':
            self.samples = [all_samples[i] for i in train_indices]
        else:  # test
            self.samples = [all_samples[i] for i in test_indices]
        
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {split} split")
        
        print(f"📊 Satellite Dataset {split}: 加载 {len(self.samples)} 个样本")
        print(f"   📐 图像尺寸: 512×512")
        print(f"   🎯 任务: 遥感分割 (二分类)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 读取TIFF图像
        image = Image.open(sample['image']).convert('RGB')
        label = Image.open(sample['label'])
        
        # 处理RGB格式的label (取第一通道并二值化)
        label_np = np.array(label)
        if len(label_np.shape) == 3:
            label_np = label_np[:, :, 0]  # 取第一通道
        label_binary = (label_np > 127).astype(np.uint8)
        
        # Resize到目标尺寸（如果需要）
        if image.size != (self.target_size[1], self.target_size[0]):
            image = image.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
        
        if label_binary.shape != self.target_size:
            label_img = Image.fromarray(label_binary)
            label_img = label_img.resize((self.target_size[1], self.target_size[0]), Image.NEAREST)
            label_binary = np.array(label_img)
        
        # 转换为tensor
        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)
        
        label_tensor = torch.from_numpy(label_binary).long()
        
        # 返回: (图像, 标签, 图像路径)
        return image, label_tensor, sample['image']


class FewShotSamplerSatellite:
    """
    Satellite Dataset的Few-shot采样
    
    采样策略
    - 从训练集中选择前景区域最显著的k个样
    """
    
    def __init__(self, dataset):
        self.dataset = dataset
        self._analyze_dataset()
    
    def _analyze_dataset(self):
        """分析数据集，按照前景像素数排"""
        print("正在分析Satellite Dataset...")
        
        self.fg_samples = []  # 存储 (idx, fg_pixel_count)
        
        for idx in range(len(self.dataset)):
            _, label, _ = self.dataset[idx]
            
            if torch.is_tensor(label):
                fg_pixels = label.sum().item()
            elif isinstance(label, np.ndarray):
                fg_pixels = np.sum(label > 0)
            else:
                fg_pixels = 0
            
            if fg_pixels > 0:
                self.fg_samples.append((idx, int(fg_pixels)))
        
        # 按前景像素数降序排序
        self.fg_samples.sort(key=lambda x: x[1], reverse=True)
        
        print(f"  ✅ 找到 {len(self.fg_samples)} 个包含前景的样本")
        if len(self.fg_samples) > 0:
            print(f"  📊 前景像素数范围 {self.fg_samples[-1][1]} ~ {self.fg_samples[0][1]}")
    
    def sample_k_shot(self, k_shot, strategy='top'):
        """
        采样k-shot样本
        
        Args:
            k_shot: 采样数量
            strategy: 'top' 选择前景最显著的 'diverse' 均匀分布采样
        """
        if strategy == 'top':
            selected = [idx for idx, _ in self.fg_samples[:k_shot]]
            print(f"\n🎯 采样策略: 选择前景区域最显著的{k_shot} 个样本")
        else:
            # 均匀分布采样
            step = len(self.fg_samples) // k_shot
            selected = [self.fg_samples[i * step][0] for i in range(k_shot)]
            print(f"\n🎯 采样策略: 均匀分布采样 {k_shot} 个样本")
        
        # 打印选中样本信息
        print(f"选中的样本索引和像素数")
        for idx in selected[:5]:
            pixel_count = next(cnt for i, cnt in self.fg_samples if i == idx)
            print(f"  样本 {idx}: {pixel_count} 前景像素")
        if len(selected) > 5:
            print(f"  ... (共{len(selected)} 个样本")
        
        return selected


class EnhancedFewShotDatasetTN3K(Dataset):
    """TN3K的增强型Few-shot数据"""
    
    def __init__(self, base_dataset, selected_indices, augment_factor=10, target_size=(512, 512)):
        self.base_dataset = base_dataset
        self.selected_indices = selected_indices
        self.augment_factor = augment_factor
        self.target_size = target_size
        
        # 图像归一化参数(ImageNet)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # 图像数据增强（包含颜色变换）
        self.image_augmentation = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])
        
        # Mask数据增强（只包含几何变换，不改变像素值）
        self.mask_augmentation = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15),
        ])
    
    def __len__(self):
        return len(self.selected_indices) * self.augment_factor
    
    def __getitem__(self, idx):
        # 确定基础样本索引
        base_idx = self.selected_indices[idx // self.augment_factor]
        aug_idx = idx % self.augment_factor
        
        # 获取原始数据
        image, mask, _ = self.base_dataset[base_idx]  # FolderDataset返回 (img_tensor, mask_tensor, _)
        
        # 处理图像
        if not torch.is_tensor(image):
            # numpy数组: BGR uint8 -> RGB float tensor [C,H,W]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            image_pil = image_pil.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
            image = T.ToTensor()(image_pil)
            image = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        else:
            # 已经是归一化的tensor [C, H, W]，但没有标准
            if image.shape[-2:] != self.target_size:
                image = F.interpolate(image.unsqueeze(0), size=self.target_size, 
                                    mode='bilinear', align_corners=False).squeeze(0)
            # 应用ImageNet标准
            image = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        
        # 处理mask
        if not torch.is_tensor(mask):
            # numpy数组: grayscale uint8 -> binary float tensor [H,W]
            mask_pil = Image.fromarray(mask)
            mask_pil = mask_pil.resize((self.target_size[1], self.target_size[0]), Image.NEAREST)
            mask_np = np.array(mask_pil)
            # 自动检测mask值范围来二值化
            if mask_np.max() > 1:
                # 0-255范围，使用27阈
                mask_binary = (mask_np > 127).astype(np.float32)
            else:
                # 0-1范围（已经是二值），直接使
                mask_binary = mask_np.astype(np.float32)
            mask = torch.from_numpy(mask_binary)
        else:
            # 已经是二值化的tensor [C, H, W] 或 [H, W]
            if mask.dim() == 3:
                mask = mask.squeeze(0)  # [1, H, W] -> [H, W]
            if mask.shape[-2:] != self.target_size:
                mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), 
                                   size=self.target_size,
                                   mode='nearest').squeeze(0).squeeze(0)
            # 确保mask是float类型，并且是二值的0/1
            mask = mask.float()
            # 如果mask值不是/1，需要二值化
            if mask.max() > 1:
                mask = (mask > 0.5).float()
        
        # 数据增强
        if aug_idx > 0:  # 第一个不增强，作为原始样
            seed = random.randint(0, 2**32 - 1)
            
            # 图像增强（包含颜色变换）
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.image_augmentation(image)
            
            # mask增强（只包含几何变换
            random.seed(seed)
            torch.manual_seed(seed)
            mask_3d = mask.unsqueeze(0)  # [H, W] -> [1, H, W]
            mask_3d = self.mask_augmentation(mask_3d)
            mask = mask_3d.squeeze(0)  # [1, H, W] -> [H, W]
            
            # 确保mask仍然是二值的（旋转可能产生插值）
            mask = (mask > 0.5).float()
        
        # 返回3个值保持一致性 (image, mask, label)
        # label设为1表示这是一个有效的训练样本
        return image, mask, 1


def visualize_specific_satellite_image(model, dataset, target_name='3_1.tif', save_path='vis_3_1.png', device='cuda'):
    """
    可视化特定的Satellite图像 (如 3_1.tif)
    """
    print(f"正在寻找并可视化特定图像: {target_name} ...")
    target_idx = -1
    
    # 查找图像
    if hasattr(dataset, 'samples'):
        for i, sample in enumerate(dataset.samples):
            # 使用 os.path.basename 确保精确匹配文件
            if os.path.basename(sample['image']) == target_name:
                target_idx = i
                break
    
    if target_idx == -1:
        print(f"  ⚠️ 未在当前数据集中找到 {target_name}")
        return False
        
    print(f"  ✅ 找到 {target_name} (Index: {target_idx})")
    
    # 获取数据
    # SatelliteDataset returns (image, label, path)
    image, mask, path = dataset[target_idx]
    
    # 预测
    model.eval()
    with torch.no_grad():
        image_input = image.unsqueeze(0).to(device)
        result = model(image_input)
        if isinstance(result, tuple):
            output = result[0]
        else:
            output = result
        
        pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()
        pred_mask = (pred_prob > 0.6).astype(np.uint8) # Satellite optimal threshold 0.6
    
    # 准备显示
    gt_mask = mask.cpu().numpy().astype(np.uint8)
    
    # 反归一
    image_np = image.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = image_np * std + mean
    image_np = np.clip(image_np, 0, 1)
    
    # Overlay calculation
    overlay = image_np.copy()
    tp_mask = np.logical_and(pred_mask == 1, gt_mask == 1)
    fp_mask = np.logical_and(pred_mask == 1, gt_mask == 0)
    fn_mask = np.logical_and(pred_mask == 0, gt_mask == 1)
    
    # Green for TP, Red for FP, Blue for FN
    overlay[tp_mask] = 0.6 * overlay[tp_mask] + 0.4 * np.array([0, 1, 0])
    overlay[fp_mask] = 0.6 * overlay[fp_mask] + 0.4 * np.array([1, 0, 0])
    overlay[fn_mask] = 0.6 * overlay[fn_mask] + 0.4 * np.array([0, 0, 1])

    # 绘图
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    axes[3].imshow(pred_prob, cmap='jet')
    axes[3].set_title('Probability')
    axes[3].axis('off')

    axes[4].imshow(overlay)
    axes[4].set_title('Overlay\nGreen=TP, Red=FP, Blue=FN')
    axes[4].axis('off')
    
    plt.suptitle(f'Visualization of {target_name}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  ✅ 可视化结果已保存: {save_path}")
    return True


def train_few_shot_tn3k(
    data_dir='./data/TN3K',
    output_dir='./tn3k_few_shot_results',
    k_shot=5,
    epochs=50,
    batch_size=4,
    lr=1e-4,
    augment_factor=10,
    sampling_strategy='top',
    use_internal_adapter=False,  # 是否在DINOv3内部注入适配器
    use_glcm=False,  # 是否使用GLCM模块
    use_hypergraph=False,
    use_layers='all',
    device='cuda',
    early_stopping_patience=10,
    seed=42,  # 添加随机种子参数
    dataset_type='tn3k',  # 新增: 数据集类
    mvtec_category=None,  # 新增: MVTec类别
    visa_category=None,  # 新增: ViSA类别
    visa_csv='split_csv/2cls_fewshot.csv',  # ViSA的CSV文件
    include_normal=False,  # ViSA是否包含正常样本
    num_classes=5,  # 类别数量 (TN3K=5, ViSA=2, MVTec=2)
    val_interval=10  # 验证间隔
):
    """
    Few-shot训练函数（支持TN3K、MVTec和ViSA数据集）
    
    Args:
        seed: 随机种子,默认42
        use_internal_adapter: 是否在DINOv3内部注入适配器
        use_glcm: 是否使用GLCM全局-局部校准模块
        dataset_type: 数据集类型'tn3k', 'mvtec' 或 'visa'
        mvtec_category: MVTec类别名称，如 'bottle'
        visa_category: ViSA类别名称，如 'candle'
        visa_csv: ViSA的CSV文件路径
        include_normal: ViSA是否包含正常样本
        num_classes: 类别数量 (TN3K=5, ViSA/MVTec=2)
        val_interval: 验证间隔（每N个epoch验证一次）
    """
    # 设置随机种子
    set_seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 根据数据集类型加载数
    if dataset_type == 'tn3k':
        print(f"\n加载TN3K数据集 {data_dir}")
        
        # TN3K不使用ResizeAndNormalize，因为它会用thr=0.5二值化
        # 直接使用FolderDataset，在数据增强阶段处理resize
        train_dataset = FolderDataset(
            root=data_dir,
            split='train',
            img_dir_name='image',
            label_dir_name='mask',
            transform=None  # 不使用transform，手动处
        )
        test_dataset = FolderDataset(
            root=data_dir,
            split='test',
            img_dir_name='image',
            label_dir_name='mask',
            transform=None  # 不使用transform，手动处
        )
        
        print(f"训练集样本数: {len(train_dataset)}")
        print(f"测试集样本数: {len(test_dataset)}")
        
        # 采样few-shot样本
        sampler = FewShotSamplerTN3K(train_dataset)
        selected_indices = sampler.sample_k_shot(k_shot, strategy=sampling_strategy)
        
    elif dataset_type == 'visa':
        print(f"\n加载ViSA数据集 {data_dir}")
        
        if visa_category is None:
            raise ValueError("使用ViSA数据集时必须指定 --visa_category")
        
        print(f"类别: {visa_category}")
        print(f"CSV文件: {visa_csv}")
        
        # 加载训练集和测试
        train_dataset = ViSADataset(
            root=data_dir,
            csv_file=visa_csv,
            split='train',
            category=visa_category,
            target_size=(512, 512)
        )
        
        test_dataset = ViSADataset(
            root=data_dir,
            csv_file=visa_csv,
            split='test',
            category=visa_category,
            target_size=(512, 512)
        )
        
        print(f"训练集样本数: {len(train_dataset)}")
        print(f"测试集样本数: {len(test_dataset)}")
        
        # 采样few-shot样本
        sampler = FewShotSamplerViSA(train_dataset)
        selected_indices = sampler.sample_k_shot(k_shot, include_normal=include_normal, strategy=sampling_strategy)
        
    elif dataset_type == 'mvtec':
        print(f"\n加载MVTec数据集 {data_dir}")
        
        # 支持加载单个类别或所有类
        if mvtec_category is None or mvtec_category == 'all':
            # 加载所有 5个类
            categories = [
                'bottle', 'cable', 'capsule', 'carpet', 'grid',
                'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
            ]
            print(f"加载所有 5个物体类型 {', '.join(categories)}")
        else:
            # 加载指定类别
            categories = [mvtec_category]
            print(f"加载类别: {mvtec_category}")
        
        # MVTec只有test集，我们将其分为训练和测
        full_dataset = MVTecDataset(
            root=data_dir,
            categories=categories,
            split='test',
            target_size=(512, 512)
        )
        
        # 采样few-shot样本（从有缺陷的样本中选择
        sampler = MVTecFewShotSampler(full_dataset)
        selected_indices = sampler.sample_k_shot(k_shot, strategy=sampling_strategy)
        
        # 使用剩余样本作为测试
        all_indices = set(range(len(full_dataset)))
        test_indices = list(all_indices - set(selected_indices))
        
        train_dataset = full_dataset
        test_dataset = Subset(full_dataset, test_indices)
        
        print(f"训练样本数(few-shot): {len(selected_indices)}")
        print(f"测试样本数 {len(test_dataset)}")
        
    elif dataset_type == 'dsifn':
        print(f"\n🌍 加载DSIFN遥感数据集（异常检测模式）: {data_dir}")
        print(f"   📁 数据结构:")
        print(f"      train/val: t1(正常) + t2(异常) + mask_256(t2的256×256标注)")
        print(f"      test:      t1(正常) + t2(异常) + mask(512×512标注)")
        print(f"   🎯 使用 t2 作为异常图像输入\n")
        
        # 加载训练集和验证
        train_dataset = DSIFNDataset(
            root=data_dir,
            split='train',
            target_size=(512, 512)
        )
        
        # 使用val作为测试
        test_dataset = DSIFNDataset(
            root=data_dir,
            split='val',
            target_size=(512, 512)
        )
        
        print(f"训练集样本数: {len(train_dataset)}")
        print(f"验证集样本数: {len(test_dataset)}")
        
        # 采样few-shot异常样本
        sampler = FewShotSamplerDSIFN(train_dataset)
        selected_indices = sampler.sample_k_shot(k_shot, strategy=sampling_strategy)
        
    elif dataset_type == 'massroads':
        print(f"\n🛣️ 加载Massachusetts Roads遥感道路分割数据集 {data_dir}")
        print(f"   📁 数据结构:")
        print(f"      data/: 遥感图像 (1500×1500 TIFF)")
        print(f"      label/: 道路标签 (1500×1500 TIFF, 二值/255)")
        print(f"   🎯 任务: 道路区域分割\n")
        
        # 加载训练集和测试集(80%训练集0%测试)
        train_dataset = MassachusettsRoadsDataset(
            root=data_dir,
            split='train',
            train_ratio=0.8,
            target_size=(512, 512),
            seed=seed
        )
        
        test_dataset = MassachusettsRoadsDataset(
            root=data_dir,
            split='test',
            train_ratio=0.8,
            target_size=(512, 512),
            seed=seed
        )
        
        print(f"训练集样本数: {len(train_dataset)}")
        print(f"测试集样本数: {len(test_dataset)}")
        
        # 采样few-shot道路样本
        sampler = FewShotSamplerMassRoads(train_dataset)
        selected_indices = sampler.sample_k_shot(k_shot, strategy=sampling_strategy)
        
    elif dataset_type == 'satellite':
        print(f"\n🛰️ 加载Satellite Dataset遥感分割数据集 {data_dir}")
        print(f"   📁 数据结构:")
        print(f"      image/: 遥感图像 (512×512 TIFF)")
        print(f"      label/: 分割标签 (512×512 TIFF, RGB二值/255)")
        print(f"   🎯 任务: 遥感区域分割\n")
        
        # 加载训练集和测试集(80%训练集0%测试)
        train_dataset = SatelliteDataset(
            root=data_dir,
            split='train',
            train_ratio=0.8,
            target_size=(512, 512),
            seed=seed
        )
        
        test_dataset = SatelliteDataset(
            root=data_dir,
            split='test',
            train_ratio=0.8,
            target_size=(512, 512),
            seed=seed
        )
        
        print(f"训练集样本数: {len(train_dataset)}")
        print(f"测试集样本数: {len(test_dataset)}")
        
        # 采样few-shot样本
        sampler = FewShotSamplerSatellite(train_dataset)
        selected_indices = sampler.sample_k_shot(k_shot, strategy=sampling_strategy)
        
    else:
        raise ValueError(f"未知的数据集类型: {dataset_type}")
    
    # 采样few-shot样本已在上面完成
    # selected_indices 已经获得
    
    # 创建增强数据
    enhanced_dataset = EnhancedFewShotDatasetTN3K(
        base_dataset=train_dataset,
        selected_indices=selected_indices,
        augment_factor=augment_factor,
        target_size=(512, 512)
    )
    
    print(f"\n增强后训练集大小: {len(enhanced_dataset)} (原始 {len(selected_indices)} × {augment_factor})")
    
    # 为所有数据集创建测试集wrapper (确保验证时进行标准化)
    # TN3KTestDataset 会对图像进行ImageNet标准化，这对DINOv3是必须的
    test_dataset_wrapped = TN3KTestDataset(test_dataset, target_size=(512, 512))
    
    # 创建数据加载
    train_loader = DataLoader(
        enhanced_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset_wrapped,
        batch_size=8,  # 增大batch size避免超图模块的维度问
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 创建模型
    print("\n创建DPT模型...")
    repo_dir = './dinov3'
    dino_ckpt = './web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
    backbone = torch.hub.load(repo_dir, 'dinov3_vits16', source='local', weights=dino_ckpt)
    
    # 所有数据集都使用二分类（背景vs 前景
    model_nclass = 1  # 二分类（使用BCEWithLogitsLoss
    
    print(f"  模型输出类别数 {model_nclass} (二分类 背景 vs 前景)")
    
    # 判断使用哪种模型配置
    if use_internal_adapter or use_glcm or use_hypergraph:
        # 使用增强版DPT
        modules_enabled = []
        if use_internal_adapter:
            modules_enabled.append("Internal Adapter")
        if use_glcm:
            modules_enabled.append("GLCM")
        if use_hypergraph:
            modules_enabled.append("超图GCN")
        
        print(f"  使用增强版DPT（{' + '.join(modules_enabled)}")
        
        model = DPTEnhanced(
            encoder_size='small',
            nclass=model_nclass,
            features=256,
            out_channels=[96, 192, 384, 768],
            use_bn=False,
            backbone=backbone,
            use_layers=use_layers,
            use_internal_adapter=use_internal_adapter,  # 内部适配器
            use_glcm=use_glcm,  # 根据参数启用GLCM
            use_hypergraph=use_hypergraph,  # 根据参数启用超图GCN
            fusion_strategy='sequential'  # Internal Adapter → GLCM → 超图GCN 顺序处理
        ).to(device)
    else:
        print("  使用基础版DPT")
        model = DPT(
            encoder_size='small',
            nclass=model_nclass,
            features=256,
            out_channels=[96, 192, 384, 768],
            use_bn=False,
            backbone=backbone,
            use_layers=use_layers
        ).to(device)
    
    # 优化器和学习率调
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 损失函数 - 根据数据集类型设置正样本权重
    # 遥感数据集（DSIFN, MassRoads）和工业缺陷数据集（ViSA）前景比例很小，需要增加前景权
    # Satellite数据集前景比例约50%，不需要权
    if dataset_type in ['dsifn', 'massroads', 'visa']:
        pos_weight = torch.tensor([10.0]).to(device)  # 前景权重10
        print(f"  使用正样本权重 10.0 (适用于{dataset_type}的类别不平衡)")
    else:
        pos_weight = None
        print(f"  使用标准BCE损失（无权重")
    
    if pos_weight is not None:
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        bce_loss = nn.BCEWithLogitsLoss()
    
    def dice_loss(pred, target, smooth=1e-6):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice
    
    # 训练循环
    best_iou = 0.0
    patience_counter = 0  # Early stopping计数
    print(f"\n开始训练 {k_shot}-shot 模型...")
    print(f"训练轮数: {epochs}, 批次大小: {batch_size}, 学习率 {lr}")
    print(f"Early stopping patience: {early_stopping_patience}")
    
    # 判断是否使用GLCM（根据参数决定）
    using_glcm = use_glcm
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_seg_loss = 0.0
        epoch_glcm_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, masks, labels in pbar:  # 接收3个值 images, masks, labels
            images = images.to(device)
            masks = masks.to(device).float()  # [B, H, W]
            
            # 确保mask是4D的[B, H, W]
            if masks.dim() == 4:  # [B, 1, H, W]
                masks = masks.squeeze(1)  # -> [B, H, W]
            
            # 扩展维度用于loss计算 [B, 1, H, W]
            masks_4d = masks.unsqueeze(1)  # [B, 1, H, W]
            
            # 前向传播（DPTEnhanced返回3个值：主输出、各层异常图、融合异常图
            result = model(images)
            if isinstance(result, tuple) and len(result) == 3:
                outputs, anomaly_maps, anomaly_map_fused = result
            elif isinstance(result, tuple) and len(result) == 2:
                outputs, anomaly_maps = result
                anomaly_map_fused = None
            else:
                outputs = result
                anomaly_maps = None
                anomaly_map_fused = None
            
            # 确保尺寸匹配
            if outputs.shape[-2:] != masks_4d.shape[-2:]:
                outputs = F.interpolate(outputs, size=masks_4d.shape[-2:],
                                      mode='bilinear', align_corners=False)
            
            # ===== 损失1: 主分割损失（BCE + Dice）====
            loss_bce = bce_loss(outputs, masks_4d)
            loss_dice = dice_loss(outputs, masks_4d)
            seg_loss = loss_bce + loss_dice
            
            # ===== 损失2: GLCM校准损失（可选）=====
            glcm_loss = 0.0
            if use_glcm and anomaly_maps is not None and len(anomaly_maps) > 0:
                # 对每一层的异常图计算监督损
                for anomaly_map in anomaly_maps:
                    if anomaly_map.shape[-2:] != masks_4d.shape[-2:]:
                        anomaly_map = F.interpolate(
                            anomaly_map,
                            size=masks_4d.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        )
                    # BCE损失
                    glcm_bce = F.binary_cross_entropy(anomaly_map, masks_4d)
                    # Dice损失
                    intersection = (anomaly_map * masks_4d).sum()
                    union = anomaly_map.sum() + masks_4d.sum()
                    glcm_dice = 1 - (2 * intersection + 1e-6) / (union + 1e-6)
                    glcm_loss += (glcm_bce + glcm_dice)
                
                # 平均多层损失
                glcm_loss = glcm_loss / len(anomaly_maps)
            
            # ===== 损失3: 融合GLCM异常图的损失（可选，与主分割输出融合）=====
            glcm_fused_loss = 0.0
            if use_glcm and anomaly_map_fused is not None:
                if anomaly_map_fused.shape[-2:] != masks_4d.shape[-2:]:
                    anomaly_map_fused = F.interpolate(
                        anomaly_map_fused,
                        size=masks_4d.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                glcm_fused_bce = F.binary_cross_entropy(anomaly_map_fused, masks_4d)
                intersection_fused = (anomaly_map_fused * masks_4d).sum()
                union_fused = anomaly_map_fused.sum() + masks_4d.sum()
                glcm_fused_dice = 1 - (2 * intersection_fused + 1e-6) / (union_fused + 1e-6)
                glcm_fused_loss = glcm_fused_bce + glcm_fused_dice
            
            # ===== 总损失=====
            # seg_loss: 主分割损失(权重1.0)
            # glcm_loss: 各层异常图辅助损失(权重0.15)
            # glcm_fused_loss: 融合异常图损失(权重0.25，参考AD-DINOv3)
            if use_glcm and (glcm_loss > 0 or glcm_fused_loss > 0):
                loss = seg_loss + 0.25 * glcm_fused_loss
            else:
                loss = seg_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_seg_loss += seg_loss.item()
            if use_glcm and glcm_loss > 0:
                epoch_glcm_loss += glcm_loss.item()
            
            # 更新进度条（显示各项损失
            if use_glcm and (glcm_loss > 0 or glcm_fused_loss > 0):
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'seg': f'{seg_loss.item():.4f}',
                    'glcm_l': f'{glcm_loss.item():.4f}',  # 各层GLCM
                    'glcm_f': f'{glcm_fused_loss.item():.4f}'  # 融合GLCM
                })
            else:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        avg_seg_loss = epoch_seg_loss / len(train_loader)
        avg_glcm_loss = epoch_glcm_loss / len(train_loader) if using_glcm else 0.0
        
        # 验证（根据val_interval
        if (epoch + 1) % val_interval == 0:
            model.eval()
            
            # ===== 二分类验证（所有数据集统一使用）====
            val_iou = 0.0
            val_accuracy = 0.0
            val_f1 = 0.0
            val_mae = 0.0
            val_ber = 0.0
            val_dice = 0.0
            val_hd95_list = []
            
            with torch.no_grad():
                    # 收集所有预测和真实标签用于整体计算
                    all_preds_list = []
                    all_masks_list = []
                    
                    # 用于逐样本计算Dice 和 HD95
                    sample_dice_list = []
                    sample_hd95_list = []
                    
                    for batch in tqdm(test_loader, desc="Validation", leave=False):
                        images, masks, _ = batch
                        images = images.to(device)
                        masks = masks.to(device).float()
                        
                        # 确保mask是4D
                        if masks.dim() == 4:  # [B, 1, H, W]
                            masks = masks.squeeze(1)  # -> [B, H, W]
                        
                        # 前向传播（处理元组或元组返回值）
                        result = model(images)
                        if isinstance(result, tuple) and len(result) >= 2:
                            outputs = result[0]  # 只取主输
                            # 可选：也可以使用融合的anomaly_map
                            # if len(result) == 3 and result[2] is not None:
                            #     outputs = 0.7 * outputs + 0.3 * result[2]  # 融合策略
                        else:
                            outputs = result
                        
                        if outputs.shape[-2:] != masks.shape[-2:]:
                            outputs = F.interpolate(outputs, size=masks.shape[-2:],
                                                  mode='bilinear', align_corners=False)
                        
                        # 根据数据集类型调整预测阈
                        # DSIFN/MassRoads/ViSA前景稀疏用0.3
                        # Satellite前景比例~23%，但模型倾向过度预测，用0.6
                        if dataset_type in ['dsifn', 'massroads', 'visa']:
                            threshold = 0.3
                        elif dataset_type == 'satellite':
                            threshold = 0.7
                        else:
                            threshold = 0.5
                        pred = (torch.sigmoid(outputs.squeeze(1)) > threshold).float()  # [B, H, W]
                        
                        # 逐样本计算Dice 和 HD95
                        for i in range(pred.shape[0]):
                            pred_np = pred[i].cpu().numpy()
                            mask_np = masks[i].cpu().numpy()
                            
                            # Dice
                            dice = compute_dice_coefficient(pred_np, mask_np)
                            sample_dice_list.append(dice)
                            
                            # HD95
                            hd95 = compute_hd95(pred_np, mask_np)
                            if not np.isinf(hd95):  # 只统计有效的HD95
                                sample_hd95_list.append(hd95)
                        
                        # 将每个样本的像素展平后添加（处理不同尺寸
                        for i in range(pred.shape[0]):
                            all_preds_list.append(pred[i].flatten())
                            all_masks_list.append(masks[i].flatten())
                    
                    # 拼接所有样本的所有像
                    all_preds = torch.cat(all_preds_list, dim=0)  # [总像素数]
                    all_masks = torch.cat(all_masks_list, dim=0)  # [总像素数]
                
                    # ========== 整体像素计算 ==========
                    # 计算 Micro IoU (整体所有像素，保留原有逻辑)
                    intersection = (all_preds * all_masks).sum()
                    union = all_preds.sum() + all_masks.sum() - intersection
                    val_iou_micro = (intersection / (union + 1e-6)).item()
                
                    # 计算标准 mIoU (背景和前景分别计算再平均)
                    # 前景 IoU (预测=1, 真实=1)
                    fg_preds = all_preds  # 前景预测
                    fg_masks = all_masks  # 前景真实
                    fg_intersection = (fg_preds * fg_masks).sum()
                    fg_union = fg_preds.sum() + fg_masks.sum() - fg_intersection
                    val_iou_fg = (fg_intersection / (fg_union + 1e-6)).item()
                
                    # 背景 IoU (预测=0,  真实=0)
                    bg_preds = 1 - all_preds  # 背景预测
                    bg_masks = 1 - all_masks  # 背景真实
                    bg_intersection = (bg_preds * bg_masks).sum()
                    bg_union = bg_preds.sum() + bg_masks.sum() - bg_intersection
                    val_iou_bg = (bg_intersection / (bg_union + 1e-6)).item()
                
                    # mIoU = 两类IoU的平
                    val_miou = (val_iou_bg + val_iou_fg) / 2.0
                
                    # 计算Accuracy
                    correct = (all_preds == all_masks).sum()
                    total = all_masks.numel()
                    val_accuracy = (correct / total).item()
                
                    # 计算F1-Score
                    tp = (all_preds * all_masks).sum()
                    fp = (all_preds * (1 - all_masks)).sum()
                    fn = ((1 - all_preds) * all_masks).sum()
                    tn = ((1 - all_preds) * (1 - all_masks)).sum()
                
                    precision = tp / (tp + fp + 1e-6)
                    recall = tp / (tp + fn + 1e-6)
                    val_f1 = (2 * precision * recall / (precision + recall + 1e-6)).item()
                
                    # 计算MAE
                    val_mae = torch.abs(all_preds - all_masks).mean().item()
                
                    # 计算BER
                    fpr = fp / (fp + tn + 1e-6)  # False Positive Rate
                    fnr = fn / (fn + tp + 1e-6)  # False Negative Rate
                    val_ber = (0.5 * (fpr + fnr)).item()
                
                    # ========== 逐样本平均计算==========
                    # Dice (平均)
                    val_dice = np.mean(sample_dice_list) if sample_dice_list else 0.0
                
                    # HD95 (平均)
                    val_hd95 = np.mean(sample_hd95_list) if sample_hd95_list else np.inf
            
            print(f"\nEpoch [{epoch+1}/{epochs}]")
            if using_glcm and avg_glcm_loss > 0:
                print(f"  Loss: {avg_loss:.4f} (Seg: {avg_seg_loss:.4f}, GLCM: {avg_glcm_loss:.4f})")
            else:
                print(f"  Loss: {avg_loss:.4f}")
            print(f"  Val mIoU (标准): {val_miou:.4f}  [背景: {val_iou_bg:.4f}, 前景: {val_iou_fg:.4f}]")
            print(f"  Val IoU (micro): {val_iou_micro:.4f}")
            print(f"  Val Dice: {val_dice:.4f}")
            print(f"  Val HD95: {val_hd95:.2f}" if not np.isinf(val_hd95) else f"  Val HD95: inf")
            print(f"  Val Accuracy: {val_accuracy:.4f}")
            print(f"  Val F1-Score: {val_f1:.4f}")
            print(f"  Val MAE: {val_mae:.4f}")
            print(f"  Val BER: {val_ber:.4f}")
            
            # ========== 可视化验证结果(每个epoch或最后一个epoch) ==========
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                print(f"\n📊 生成验证可视化(Epoch {epoch+1})...")
                vis_dir = os.path.join(output_dir, f'visualizations/epoch_{epoch+1}')
                os.makedirs(vis_dir, exist_ok=True)
                
                # 随机选择5个验证样本进行可视化
                num_vis_samples = min(5, len(test_dataset))
                vis_indices = np.random.choice(len(test_dataset), num_vis_samples, replace=False)
                
                model.eval()
                with torch.no_grad():
                    for vis_idx, sample_idx in enumerate(vis_indices):
                        # 获取样本
                        if isinstance(test_dataset, torch.utils.data.Subset):
                            image, mask, info = test_dataset.dataset[test_dataset.indices[sample_idx]]
                        else:
                            image, mask, info = test_dataset[sample_idx]
                        
                        image_input = image.unsqueeze(0).to(device)
                        
                        # 预测
                        result = model(image_input)
                        if isinstance(result, tuple):
                            output = result[0]
                        else:
                            output = result
                        
                        if output.shape[-2:] != mask.shape[-2:]:
                            output = F.interpolate(output, size=mask.shape[-2:],
                                                  mode='bilinear', align_corners=False)
                        
                        # 使用与验证相同的阈
                        if dataset_type in ['dsifn', 'massroads', 'visa']:
                            vis_threshold = 0.3
                        elif dataset_type == 'satellite':
                            vis_threshold = 0.9
                        else:
                            vis_threshold = 0.5
                        
                        pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()
                        pred_mask = (pred_prob > vis_threshold).astype(np.uint8)
                        gt_mask = mask.cpu().numpy().astype(np.uint8)
                        
                        # 确保gt_mask是2D的 [H, W]
                        if gt_mask.ndim == 3:
                            gt_mask = gt_mask.squeeze(0)
                        
                        # 反归一化图
                        image_np = image.cpu().numpy().transpose(1, 2, 0)
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        image_np = image_np * std + mean
                        image_np = np.clip(image_np, 0, 1)
                        
                        # 计算IoU
                        intersection = np.logical_and(pred_mask, gt_mask).sum()
                        union = np.logical_or(pred_mask, gt_mask).sum()
                        iou = intersection / (union + 1e-6)
                        
                        # 处理样本信息
                        sample_info = str(sample_idx)
                        full_path = ""
                        if isinstance(info, str):
                            full_path = info
                            # 如果是路径，尝试提取文件
                            if '/' in info or '\\' in info:
                                sample_info = os.path.basename(info)
                            else:
                                sample_info = info
                        elif isinstance(info, dict):
                            # 处理FolderDataset返回的字典信
                            if 'img_path' in info:
                                full_path = info['img_path']
                                sample_info = os.path.basename(full_path)
                            elif 'id' in info:
                                sample_info = info['id']

                        # 创建可视化(2x3布局)
                        import matplotlib.pyplot as plt
                        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                        
                        # Row 1
                        axes[0, 0].imshow(image_np)
                        axes[0, 0].set_title(f'Original Image\n{sample_info}', fontsize=14, fontweight='bold')
                        axes[0, 0].axis('off')
                        
                        axes[0, 1].imshow(gt_mask, cmap='gray')
                        axes[0, 1].set_title(f'Ground Truth\nForeground: {gt_mask.sum():,} ({100*gt_mask.mean():.1f}%)', 
                                           fontsize=14, fontweight='bold')
                        axes[0, 1].axis('off')
                        
                        axes[0, 2].imshow(pred_mask, cmap='gray')
                        axes[0, 2].set_title(f'Prediction\nForeground: {pred_mask.sum():,} ({100*pred_mask.mean():.1f}%)', 
                                           fontsize=14, fontweight='bold')
                        axes[0, 2].axis('off')
                        
                        # Row 2
                        axes[1, 0].imshow(pred_prob, cmap='jet', vmin=0, vmax=1)
                        axes[1, 0].set_title('Probability Map', fontsize=14, fontweight='bold')
                        axes[1, 0].axis('off')
                        plt.colorbar(axes[1, 0].images[0], ax=axes[1, 0], fraction=0.046, pad=0.04)
                        
                        # Overlay
                        overlay = image_np.copy()
                        tp_mask = np.logical_and(pred_mask == 1, gt_mask == 1)
                        fp_mask = np.logical_and(pred_mask == 1, gt_mask == 0)
                        fn_mask = np.logical_and(pred_mask == 0, gt_mask == 1)
                        
                        overlay[tp_mask] = [0, 1, 0]  # Green
                        overlay[fp_mask] = [1, 0, 0]  # Red
                        overlay[fn_mask] = [0, 0, 1]  # Blue
                        
                        axes[1, 1].imshow(overlay)
                        axes[1, 1].set_title('Overlay\nGreen=TP, Red=FP, Blue=FN', fontsize=14, fontweight='bold')
                        axes[1, 1].axis('off')
                        
                        # Metrics display
                        axes[1, 2].axis('off')
                        stats_text = f"""
Epoch {epoch+1} Validation

IoU: {iou:.4f}
Threshold: {vis_threshold}

Probability Stats:
  Min: {pred_prob.min():.4f}
  Max: {pred_prob.max():.4f}
  Mean: {pred_prob.mean():.4f}
  Std: {pred_prob.std():.4f}

Sample: {full_path if full_path else sample_idx}
                        """
                        axes[1, 2].text(0.1, 0.5, stats_text, fontsize=11, 
                                       verticalalignment='center',
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
                        
                        fig.suptitle(f'Epoch {epoch+1} - 验证样本 {vis_idx+1}/{num_vis_samples}', 
                                    fontsize=16, fontweight='bold')
                        plt.tight_layout()
                        
                        save_path = os.path.join(vis_dir, f'val_sample_{sample_idx:04d}.png')
                        plt.savefig(save_path, dpi=100, bbox_inches='tight')
                        plt.close()
                
                print(f"✅ 可视化已保存到 {vis_dir}")
            
            # 保存最佳模型(使用 mIoU 作为选择标准)
            if val_miou > best_iou:
                best_iou = val_miou
                patience_counter = 0  # 重置计数
                model_path = os.path.join(output_dir, f'best_model_{k_shot}shot.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_miou': val_miou,
                    'val_iou_micro': val_iou_micro,
                    'val_iou_bg': val_iou_bg,
                    'val_iou_fg': val_iou_fg,
                    'val_dice': val_dice,
                    'val_hd95': val_hd95 if not np.isinf(val_hd95) else -1,
                    'val_accuracy': val_accuracy,
                    'val_f1': val_f1,
                    'val_mae': val_mae,
                    'val_ber': val_ber,
                    'k_shot': k_shot,
                    'use_hypergraph': use_hypergraph,
                    'use_layers': use_layers,
                    'dataset_type': dataset_type,
                }, model_path)
                print(f"✅ 保存最佳模型(Val mIoU: {best_iou:.4f}, Dice: {val_dice:.4f}, HD95: {val_hd95:.2f})" 
                      if not np.isinf(val_hd95) else 
                      f"✅ 保存最佳模型(Val mIoU: {best_iou:.4f}, Dice: {val_dice:.4f})")
            else:
                patience_counter += 1
                print(f"Val mIoU未提升({patience_counter}/{early_stopping_patience})")
                
                # Early stopping检
                if patience_counter >= early_stopping_patience:
                    print(f"\n⚠️ Early stopping triggered at epoch {epoch+1}")
                    print(f"Best Val mIoU: {best_iou:.4f} (stopped after {early_stopping_patience} epochs without improvement)")
                    break
    
    print(f"\n训练完成! 最佳Val mIoU: {best_iou:.4f}")

    # 可视化特定图像(Satellite 3_1.tif)
    if dataset_type == 'satellite':
        best_model_path = os.path.join(output_dir, f'best_model_{k_shot}shot.pth')
        if os.path.exists(best_model_path):
            # Load best weights
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            vis_save_path = os.path.join(output_dir, f'vis_3_1_{k_shot}shot.png')
            # Try to find in test dataset first, then train dataset
            found = visualize_specific_satellite_image(model, test_dataset, '3_1.tif', vis_save_path, device)
            if not found:
                print("  在测试集中未找到，尝试训练集...")
                visualize_specific_satellite_image(model, train_dataset, '3_1.tif', vis_save_path, device)

    return os.path.join(output_dir, f'best_model_{k_shot}shot.pth')


def main():
    parser = argparse.ArgumentParser(description='Few-shot Learning (TN3K/MVTec/ViSA)')
    
    # 数据集相关参
    parser.add_argument('--dataset_type', type=str, default='tn3k',
                      choices=['tn3k', 'mvtec', 'visa', 'dsifn', 'massroads', 'satellite'],
                      help='数据集类型 tn3k, mvtec, visa, dsifn (遥感变化检测, massroads (遥感道路), satellite (遥感分割)')
    parser.add_argument('--data_dir', type=str, default='./segdata/tn3k',
                      help='数据集路')
    parser.add_argument('--mvtec_category', type=str, default=None,
                      choices=['all', 'bottle', 'cable', 'capsule', 'carpet', 'grid',
                              'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                              'tile', 'toothbrush', 'transistor', 'wood', 'zipper'],
                      help='MVTec类别: all=所有 5个类型 或指定单个类型(仅当dataset_type=mvtec时需要')
    parser.add_argument('--visa_category', type=str, default=None,
                      help='ViSA类别 (如 candle, capsules等) (仅当dataset_type=visa时需要')
    parser.add_argument('--visa_csv', type=str, default='split_csv/2cls_fewshot.csv',
                      help='ViSA的CSV文件路径')
    parser.add_argument('--include_normal', action='store_true',
                      help='ViSA是否包含正常样本（仅当dataset_type=visa时）')
    parser.add_argument('--output_dir', type=str, default='./runs/tn3k_fewshot',
                      help='输出目录')
    parser.add_argument('--val_interval', type=int, default=10,
                      help='验证间隔（每N个epoch验证一次）')
    
    # 训练参数
    parser.add_argument('--k_shots', type=int, nargs='+', default=[5, 10, 20],
                      help='Few-shot数量')
    parser.add_argument('--epochs', type=int, default=50,
                      help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='学习率')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                      help='Early stopping耐心值（连续多少个epoch不提升则停止')
    parser.add_argument('--augment_factor', type=int, default=10,
                      help='数据增强倍数')
    parser.add_argument('--sampling_strategy', type=str, default='top',
                      choices=['top', 'diverse'],
                      help='采样策略: top-前景最多 diverse-均匀分布')
    
    # 模型增强模块参数
    parser.add_argument('--use_internal_adapter', action='store_true',
                      help='在DINOv3内部注入适配器（使冻结骨干网络适应数据集）')
    parser.add_argument('--use_glcm', action='store_true',
                      help='使用GLCM全局-局部校准模块')
    parser.add_argument('--use_hypergraph', action='store_true',
                      help='使用超图GCN模块')
    parser.add_argument('--use_layers', type=str, default='6_9',
                      choices=['all', '6_9'],
                      help='DINOv2特征层: all-4层, 6_9-2层')
    parser.add_argument('--device', type=str, default='cuda',
                      help='设备')
    parser.add_argument('--seed', type=int, default=42,
                      help='随机种子（确保可复现')
    
    args = parser.parse_args()
    
    # 数据集验
    if args.dataset_type == 'mvtec':
        if args.mvtec_category is None:
            # 默认训练所有类
            print("未指定MVTec类别，将训练所有 5个类别")
            args.mvtec_category = 'all'
    elif args.dataset_type == 'visa':
        if args.visa_category is None:
            raise ValueError("使用ViSA数据集时必须指定 --visa_category")
    
    print("="*80)
    print(f"{args.dataset_type.upper()} Few-shot Learning 实验")
    print("="*80)
    print(f"数据集类型 {args.dataset_type}")
    if args.dataset_type == 'mvtec':
        if args.mvtec_category == 'all':
            print(f"MVTec类别: 所有 5个类别（联合训练）")
        else:
            print(f"MVTec类别: {args.mvtec_category}")
    elif args.dataset_type == 'visa':
        print(f"ViSA类别: {args.visa_category}")
        print(f"CSV文件: {args.visa_csv}")
        print(f"包含正常样本: {args.include_normal}")
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"K-shot设置: {args.k_shots}")
    print(f"训练轮数: {args.epochs}")
    print(f"验证间隔: {args.val_interval}")
    print(f"增强倍数: {args.augment_factor}")
    print(f"采样策略: {args.sampling_strategy}")
    print(f"使用Internal Adapter: {args.use_internal_adapter}")
    print(f"使用GLCM: {args.use_glcm}")
    print(f"使用超图GCN: {args.use_hypergraph}")
    print(f"特征层配置 {args.use_layers}")
    print(f"Early stopping: {args.early_stopping_patience}")
    print(f"随机种子: {args.seed}")
    print("="*80)
    
    # 对每个k_shot进行实验
    results = {}
    for k_shot in args.k_shots:
        print(f"\n{'='*80}")
        print(f"开始{k_shot}-shot 实验")
        print(f"{'='*80}")
        
        output_dir = os.path.join(args.output_dir, f'{k_shot}shot')
        
        model_path = train_few_shot_tn3k(
            data_dir=args.data_dir,
            output_dir=output_dir,
            k_shot=k_shot,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            augment_factor=args.augment_factor,
            sampling_strategy=args.sampling_strategy,
            use_internal_adapter=args.use_internal_adapter,  # 新增
            use_glcm=args.use_glcm,
            use_hypergraph=args.use_hypergraph,
            use_layers=args.use_layers,
            device=args.device,
            early_stopping_patience=args.early_stopping_patience,
            seed=args.seed,
            dataset_type=args.dataset_type,
            mvtec_category=args.mvtec_category,
            visa_category=args.visa_category,
            visa_csv=args.visa_csv,
            include_normal=args.include_normal,
            val_interval=args.val_interval
        )
        
        results[k_shot] = model_path
    
    # 打印结果总结
    print("\n" + "="*80)
    print("实验结果总结")
    print("="*80)
    for k_shot, model_path in results.items():
        print(f"{k_shot}-shot: {model_path}")
    print("="*80)


if __name__ == '__main__':
    main()
