#!/usr/bin/env python3
"""
增强版Few-shot SegDINO训练脚本
添加了数据增强技术来提高小样本学习效果
"""

import os
import sys
import argparse
import random
import logging
import time
from pathlib import Path
import numpy as np
from collections import defaultdict
import shutil
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torch.amp import autocast, GradScaler
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import pandas as pd
from scipy.ndimage import distance_transform_edt

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from dataset_3570_final import Dataset3570Final
from dpt import DPT
from predict_dinov3 import load_model, preprocess_image, postprocess_prediction, visualize_prediction

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiClassSegmentationLoss(nn.Module):
    """多类别分割损失函数：CrossEntropy + Multi-class Dice + Focal Loss"""
    def __init__(self, num_classes=5, ce_weight=1.0, dice_weight=1.0, focal_weight=0.5, alpha=1.0, gamma=2.0, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight 
        self.focal_weight = focal_weight
        
        # CrossEntropy损失（支持类别权重）
        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
        
        # Focal Loss参数
        self.alpha = alpha
        self.gamma = gamma
        
    def dice_loss_multiclass(self, pred, target, smooth=1e-6):
        """多类别Dice损失"""
        # pred: [B, C, H, W], target: [B, H, W]
        pred_softmax = torch.softmax(pred, dim=1)
        target_one_hot = torch.zeros_like(pred)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)
        
        dice_scores = []
        for c in range(self.num_classes):
            pred_c = pred_softmax[:, c]
            target_c = target_one_hot[:, c]
            
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice = (2 * intersection + smooth) / (union + smooth)
            dice_scores.append(dice)
        
        return 1 - torch.stack(dice_scores).mean()
    
    def focal_loss(self, pred, target, alpha=1.0, gamma=2.0):
        """Focal Loss"""
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def forward(self, pred, target):
        # 确保target是长整型
        target = target.long()
        
        # CrossEntropy损失
        ce_loss = self.ce_loss(pred, target)
        
        # Dice损失
        dice_loss = self.dice_loss_multiclass(pred, target)
        
        # Focal损失
        focal_loss = self.focal_loss(pred, target, self.alpha, self.gamma)
        
        # 组合损失
        total_loss = (self.ce_weight * ce_loss + 
                     self.dice_weight * dice_loss + 
                     self.focal_weight * focal_loss)
        
        return total_loss, {
            'ce_loss': ce_loss.item(),
            'dice_loss': dice_loss.item(),
            'focal_loss': focal_loss.item(),
            'total_loss': total_loss.item()
        }


# ===== ViSA Dataset =====
class ViSADataset(torch.utils.data.Dataset):
    """
    ViSA Dataset for Few-shot Learning with DINOv3
    
    Dataset structure:
    visa/
      ├── split_csv/
      │   └── 2cls_fewshot.csv
      ├── candle/
      │   └── Data/
      │       ├── Images/
      │       │   ├── Normal/
      │       │   └── Anomaly/
      │       └── Masks/
      │           └── Anomaly/  (Mask值: 0-6, >0即为前景)
      └── capsules/
          └── ...
    """
    
    def __init__(self, root, csv_file, split='train', category=None, transform=None, target_size=(518, 518)):
        """
        Args:
            root: ViSA数据集根目录
            csv_file: CSV文件路径 (如 'split_csv/2cls_fewshot.csv')
            split: 'train' or 'test'
            category: 特定类别 (如 'candle'), None表示所有类别
            transform: 数据增强
            target_size: 目标尺寸 (H, W) - DINOv3使用518x518
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
                    # 尝试.png扩展名
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
            # ViSA的mask值是0-6，>0即为前景 (转换为二值: 0=背景, 1=前景)
            mask_binary = (mask_np > 0).astype(np.uint8)
        else:
            # 正常样本全0 mask
            mask_binary = np.zeros(image.size[::-1], dtype=np.uint8)
        
        # Resize到目标尺寸
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
        
        # 返回3个值以保持与TN3K数据集一致 (image, mask, label)
        # label: 0=normal, 1=anomaly
        label = 1 if sample['label'] == 'anomaly' else 0
        
        return image, mask_tensor, label


def sample_k_shot_visa(visa_dataset, k, include_normal=False, verbose=True):
    """ViSA数据集的采样器 - 选择k个异常样本 + 可选k个正常样本"""
    if verbose:
        print(f"\n正在从 ViSA 数据集中采样 {k}-shot...")
        print(f"  include_normal={include_normal}")
    
    anomaly_indices = []
    normal_indices = []
    
    # 遍历所有样本，分类为anomaly和normal
    for idx in range(len(visa_dataset)):
        sample = visa_dataset.samples[idx]
        
        if sample['label'] == 'anomaly':
            # 检查是否真的有异常区域
            if sample['mask'] is not None:
                try:
                    _, mask, _ = visa_dataset[idx]  # 修复：现在返回3个值
                    if mask.sum() > 0:  # 有前景像素
                        anomaly_indices.append(idx)
                except:
                    pass
        elif sample['label'] == 'normal':
            normal_indices.append(idx)
    
    if verbose:
        print(f"  找到 {len(anomaly_indices)} 个异常样本, {len(normal_indices)} 个正常样本")
    
    # 采样k个异常样本
    if len(anomaly_indices) < k:
        print(f"  警告: 只有 {len(anomaly_indices)} 个异常样本，但需要 {k} 个")
        selected_anomaly = anomaly_indices
    else:
        selected_anomaly = random.sample(anomaly_indices, k)
    
    selected_indices = selected_anomaly
    
    # 可选: 添加k个正常样本
    if include_normal and len(normal_indices) > 0:
        k_normal = min(k, len(normal_indices))
        selected_normal = random.sample(normal_indices, k_normal)
        selected_indices.extend(selected_normal)
        if verbose:
            print(f"  添加 {k_normal} 个正常样本")
    
    if verbose:
        print(f"✓ ViSA采样完成: 共选中 {len(selected_indices)} 个样本")
        print(f"  - {len(selected_anomaly)} 个异常样本")
        if include_normal:
            print(f"  - {len(selected_indices) - len(selected_anomaly)} 个正常样本\n")
    
    return selected_indices


def compute_hd95(pred, gt, target_size=(256, 256)):
    """
    计算Hausdorff Distance 95th percentile (HD95)
    在256x256标准下计算
    使用Canny边缘检测 + 距离变换
    
    Args:
        pred: 预测mask (numpy array或torch tensor)
        gt: ground truth mask (numpy array或torch tensor)
        target_size: 计算HD95的标准尺寸，默认(256, 256)
    
    Returns:
        float: HD95值
    """
    # 转换为numpy
    if isinstance(pred, torch.Tensor):
        pred_np = pred.cpu().numpy().astype(np.uint8)
    else:
        pred_np = pred.astype(np.uint8)
    
    if isinstance(gt, torch.Tensor):
        gt_np = gt.cpu().numpy().astype(np.uint8)
    else:
        gt_np = gt.astype(np.uint8)
    
    # Resize到256x256标准
    if pred_np.shape != target_size:
        pred_np = cv2.resize(pred_np, target_size, interpolation=cv2.INTER_NEAREST)
    if gt_np.shape != target_size:
        gt_np = cv2.resize(gt_np, target_size, interpolation=cv2.INTER_NEAREST)
    
    # 如果预测或GT全为0或全为1，返回最大距离
    max_distance = np.sqrt(target_size[0]**2 + target_size[1]**2)  # 对角线长度
    
    if pred_np.sum() == 0 or gt_np.sum() == 0:
        return float(max_distance)
    if pred_np.sum() == pred_np.size or gt_np.sum() == gt_np.size:
        return float(max_distance)
    
    try:
        # 提取边缘
        pred_edges = cv2.Canny(pred_np * 255, 50, 150)
        gt_edges = cv2.Canny(gt_np * 255, 50, 150)
        
        if pred_edges.sum() == 0 or gt_edges.sum() == 0:
            return float(max_distance)
        
        # 距离变换
        pred_dt = distance_transform_edt(~pred_edges.astype(bool))
        gt_dt = distance_transform_edt(~gt_edges.astype(bool))
        
        # 计算边缘点到对方的距离
        pred_edge_points = np.argwhere(pred_edges > 0)
        gt_edge_points = np.argwhere(gt_edges > 0)
        
        distances_pred_to_gt = pred_dt[gt_edge_points[:, 0], gt_edge_points[:, 1]]
        distances_gt_to_pred = gt_dt[pred_edge_points[:, 0], pred_edge_points[:, 1]]
        
        # 合并距离并计算95分位数
        all_distances = np.concatenate([distances_pred_to_gt, distances_gt_to_pred])
        hd95 = np.percentile(all_distances, 95)
        
        return float(hd95)
    except:
        return float(max_distance)


def calculate_segmentation_metrics(pred_mask, gt_mask, num_classes=5, ignore_background=False):
    """
    计算分割评估指标
    
    Args:
        pred_mask: 预测掩码 [H, W]，值为类别索引 (0-5)
        gt_mask: 真实掩码 [H, W]，值为类别索引 (0-5)
        num_classes: 类别数量
        ignore_background: 是否在计算指标时忽略背景类（类别0）
    
    Returns:
        metrics: 包含各类指标的字典
    """
    pred_mask = pred_mask.flatten()
    gt_mask = gt_mask.flatten()
    
    # 确定计算的类别范围
    class_start = 1 if ignore_background else 0
    
    # 初始化统计量
    tp = np.zeros(num_classes)  # True Positive
    fp = np.zeros(num_classes)  # False Positive
    fn = np.zeros(num_classes)  # False Negative
    tn = np.zeros(num_classes)  # True Negative
    
    # 按类别计算混淆矩阵
    for cls in range(num_classes):
        pred_cls = (pred_mask == cls)
        gt_cls = (gt_mask == cls)
        
        tp[cls] = np.sum(pred_cls & gt_cls)
        fp[cls] = np.sum(pred_cls & ~gt_cls)
        fn[cls] = np.sum(~pred_cls & gt_cls)
        tn[cls] = np.sum(~pred_cls & ~gt_cls)
    
    # 计算每类的IoU
    iou_per_class = np.zeros(num_classes)
    for cls in range(num_classes):
        union = tp[cls] + fp[cls] + fn[cls]
        if union > 0:
            iou_per_class[cls] = tp[cls] / union
        else:
            iou_per_class[cls] = 0.0
    
    # 计算mIoU（可选择是否包含背景）
    valid_ious = iou_per_class[class_start:]
    miou = np.mean(valid_ious)
    
    # 计算像素准确率（Pixel Accuracy）
    correct_pixels = np.sum(pred_mask == gt_mask)
    total_pixels = len(pred_mask)
    pixel_accuracy = correct_pixels / total_pixels if total_pixels > 0 else 0.0
    
    # 计算每类的精确率（Precision）和召回率（Recall）
    precision_per_class = np.zeros(num_classes)
    recall_per_class = np.zeros(num_classes)
    f1_per_class = np.zeros(num_classes)
    
    for cls in range(num_classes):
        # 精确率 = TP / (TP + FP)
        if (tp[cls] + fp[cls]) > 0:
            precision_per_class[cls] = tp[cls] / (tp[cls] + fp[cls])
        
        # 召回率 = TP / (TP + FN)
        if (tp[cls] + fn[cls]) > 0:
            recall_per_class[cls] = tp[cls] / (tp[cls] + fn[cls])
        
        # F1分数 = 2 * (Precision * Recall) / (Precision + Recall)
        if (precision_per_class[cls] + recall_per_class[cls]) > 0:
            f1_per_class[cls] = 2 * precision_per_class[cls] * recall_per_class[cls] / \
                              (precision_per_class[cls] + recall_per_class[cls])
    
    # 计算平均指标（不包含背景）
    mean_precision = np.mean(precision_per_class[class_start:])
    mean_recall = np.mean(recall_per_class[class_start:])
    mean_f1 = np.mean(f1_per_class[class_start:])
    
    # ========== 二分类方式计算过检率和漏检率（背景 vs 所有前景类） ==========
    # 将所有前景类（类1-5）合并成一个"前景"类别，与背景类进行二分类评估
    
    # 创建二值掩码：0=背景，1=前景（任意非背景类）
    pred_foreground = (pred_mask > 0)  # 预测的前景像素
    gt_foreground = (gt_mask > 0)      # 真实的前景像素
    
    # 二分类混淆矩阵
    binary_tp = np.sum(pred_foreground & gt_foreground)    # 预测前景且真实前景
    binary_fp = np.sum(pred_foreground & ~gt_foreground)   # 预测前景但实际背景
    binary_fn = np.sum(~pred_foreground & gt_foreground)   # 预测背景但实际前景
    binary_tn = np.sum(~pred_foreground & ~gt_foreground)  # 预测背景且真实背景
    
    # 过检率 = 错误预测为前景的背景像素 / 所有背景像素
    # False Alarm Rate = FP / (FP + TN)
    false_alarm_rate = binary_fp / (binary_fp + binary_tn) if (binary_fp + binary_tn) > 0 else 0.0
    
    # 漏检率 = 错误预测为背景的前景像素 / 所有前景像素
    # Miss Rate = FN / (FN + TP)
    miss_rate = binary_fn / (binary_fn + binary_tp) if (binary_fn + binary_tp) > 0 else 0.0
    
    # 保留原来的多类别统计（用于详细分析）
    total_fp_multiclass = np.sum(fp[class_start:])  # 多类别方式的总FP
    total_fn_multiclass = np.sum(fn[class_start:])  # 多类别方式的总FN
    total_tp_multiclass = np.sum(tp[class_start:])  # 多类别方式的总TP
    total_tn_multiclass = np.sum(tn[class_start:])  # 多类别方式的总TN
    
    # 计算类别平衡准确率（Balanced Accuracy）
    balanced_accuracy_per_class = np.zeros(num_classes)
    for cls in range(num_classes):
        sensitivity = recall_per_class[cls]  # TPR = TP / (TP + FN)
        specificity = tn[cls] / (tn[cls] + fp[cls]) if (tn[cls] + fp[cls]) > 0 else 0.0  # TNR = TN / (TN + FP)
        balanced_accuracy_per_class[cls] = (sensitivity + specificity) / 2
    
    mean_balanced_accuracy = np.mean(balanced_accuracy_per_class[class_start:])
    
    # 组装结果
    metrics = {
        'mIoU': miou,
        'pixel_accuracy': pixel_accuracy,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'mean_f1': mean_f1,
        'false_alarm_rate': false_alarm_rate,  # 过检率（二分类方式）
        'miss_rate': miss_rate,  # 漏检率（二分类方式）
        'balanced_accuracy': mean_balanced_accuracy,
        # 二分类统计（背景 vs 前景）
        'binary_tp': binary_tp,
        'binary_fp': binary_fp,
        'binary_fn': binary_fn,
        'binary_tn': binary_tn,
        # 每类详细指标
        'iou_per_class': iou_per_class,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }
    
    return metrics


def print_metrics(metrics, class_names=None, show_per_class=True):
    """
    打印评估指标
    
    Args:
        metrics: calculate_segmentation_metrics返回的指标字典
        class_names: 类别名称列表
        show_per_class: 是否显示每类的详细指标
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(metrics['iou_per_class']))]
    
    print("\n" + "="*80)
    print("分割评估指标")
    print("="*80)
    
    # 总体指标
    print(f"【总体指标】")
    print(f"  mIoU (平均交并比):          {metrics['mIoU']:.4f} ({metrics['mIoU']*100:.2f}%)")
    print(f"  像素准确率 (Pixel Accuracy): {metrics['pixel_accuracy']:.4f} ({metrics['pixel_accuracy']*100:.2f}%)")
    print(f"  平均精确率 (Precision):      {metrics['mean_precision']:.4f} ({metrics['mean_precision']*100:.2f}%)")
    print(f"  平均召回率 (Recall):         {metrics['mean_recall']:.4f} ({metrics['mean_recall']*100:.2f}%)")
    print(f"  平均F1分数:                  {metrics['mean_f1']:.4f} ({metrics['mean_f1']*100:.2f}%)")
    print(f"  平衡准确率:                  {metrics['balanced_accuracy']:.4f} ({metrics['balanced_accuracy']*100:.2f}%)")
    
    # 二分类指标（背景 vs 前景）
    print(f"\n【二分类指标（背景 vs 所有前景类）】")
    print(f"  过检率 (False Alarm Rate):   {metrics['false_alarm_rate']:.4f} ({metrics['false_alarm_rate']*100:.2f}%)")
    print(f"    → 将背景错误预测为前景的比例")
    print(f"  漏检率 (Miss Rate):          {metrics['miss_rate']:.4f} ({metrics['miss_rate']*100:.2f}%)")
    print(f"    → 将前景错误预测为背景的比例")
    if 'binary_tp' in metrics:
        total_foreground = metrics['binary_tp'] + metrics['binary_fn']
        total_background = metrics['binary_fp'] + metrics['binary_tn']
        print(f"  前景像素数: {total_foreground:,} (TP={metrics['binary_tp']:,}, FN={metrics['binary_fn']:,})")
        print(f"  背景像素数: {total_background:,} (TN={metrics['binary_tn']:,}, FP={metrics['binary_fp']:,})")
    
    # 每类详细指标
    if show_per_class:
        print(f"\n【各类别指标】")
        print(f"{'类别':<15} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'TP':>8} {'FP':>8} {'FN':>8}")
        print("-" * 80)
        
        for i, name in enumerate(class_names):
            print(f"{name:<15} "
                  f"{metrics['iou_per_class'][i]:>7.4f} "
                  f"{metrics['precision_per_class'][i]:>10.4f} "
                  f"{metrics['recall_per_class'][i]:>8.4f} "
                  f"{metrics['f1_per_class'][i]:>8.4f} "
                  f"{int(metrics['tp'][i]):>8d} "
                  f"{int(metrics['fp'][i]):>8d} "
                  f"{int(metrics['fn'][i]):>8d}")
    
    print("="*80 + "\n")


class EnhancedFewShotDataset(Dataset):
    """增强版Few-shot数据集，包含数据增强"""
    
    def __init__(self, base_dataset, selected_indices, augment_factor=10):
        """
        Args:
            base_dataset: 基础数据集
            selected_indices: 选中的样本索引
            augment_factor: 增强倍数（每个原始样本生成多少个增强版本）
        """
        self.base_dataset = base_dataset
        self.selected_indices = selected_indices
        self.augment_factor = augment_factor
        
        # 定义数据增强管道
        self.img_transform = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(p=0.5),                    # 随机水平翻转
            T.RandomRotation(degrees=(-10, 10)),              # 小角度旋转
            T.ColorJitter(brightness=0.2, contrast=0.2,       # 颜色抖动
                         saturation=0.1, hue=0.05),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05), # 轻微平移和缩放
                          scale=(0.95, 1.05)),
            T.ToTensor()
        ])
        
        # 掩码增强（只做几何变换）
        self.mask_transform = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=(-10, 10)),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05),
                          scale=(0.95, 1.05)),
            T.ToTensor()
        ])
        
        # ImageNet标准化
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
    def __len__(self):
        return len(self.selected_indices) * self.augment_factor
    
    def __getitem__(self, idx):
        # 确定原始样本索引
        original_idx = self.selected_indices[idx // self.augment_factor]
        augment_idx = idx % self.augment_factor
        
        # 获取原始数据
        img_tensor, mask_tensor, meta = self.base_dataset[original_idx]
        
        # 转换为numpy格式用于增强
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        # mask已经是类别标签(0-4),不需要乘以255
        mask_np = mask_tensor.squeeze().numpy().astype(np.uint8)
        
        # 设置相同的随机种子确保图像和掩码变换一致
        seed = random.randint(0, 2**32-1)
        
        # 增强图像
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        img_aug = self.img_transform(img_np)
        
        # 增强掩码（使用相同种子）
        # 注意：mask是类别标签，需要特殊处理
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 对于类别mask,不能使用ToTensor()(会除以255)
        # 手动实现几何变换
        from PIL import Image as PILImage
        mask_pil = PILImage.fromarray(mask_np, mode='L')
        
        # 应用相同的几何变换
        if random.random() < 0.5:
            mask_pil = mask_pil.transpose(PILImage.FLIP_LEFT_RIGHT)
        
        # 随机旋转
        angle = random.uniform(-10, 10)
        mask_pil = mask_pil.rotate(angle, resample=PILImage.NEAREST, fillcolor=0)
        
        # 随机仿射变换
        width, height = mask_pil.size
        translate_x = random.uniform(-0.05, 0.05) * width
        translate_y = random.uniform(-0.05, 0.05) * height
        scale = random.uniform(0.95, 1.05)
        
        from PIL import Image as PILImage
        import PIL
        mask_pil = mask_pil.transform(
            mask_pil.size,
            PIL.Image.AFFINE,
            (1/scale, 0, translate_x, 0, 1/scale, translate_y),
            resample=PILImage.NEAREST,
            fillcolor=0
        )
        
        mask_aug = torch.from_numpy(np.array(mask_pil)).long()
        
        # 标准化图像
        img_normalized = self.normalize(img_aug)
        
        # 确保mask在有效范围内（0-4共5类）
        mask_final = torch.clamp(mask_aug, 0, 4)
        
        return img_normalized, mask_final, meta


class FewShotSampler:
    """Few-shot采样器"""
    
    def __init__(self, dataset, num_classes=5):
        self.dataset = dataset
        self.num_classes = num_classes
        self.class_indices = self._build_class_indices()
    
    def _build_class_indices(self):
        """构建每个类别的样本索引（按主要缺陷类别分组）"""
        # 改为字典，存储 {class_id: [(idx, pixel_count), ...]}
        class_pixel_counts = defaultdict(list)
        
        print("正在分析数据集中的类别分布（按主要缺陷类别分组）...")
        for idx in range(len(self.dataset)):
            _, mask, _ = self.dataset[idx]
            mask_np = mask.squeeze().numpy().astype(int)
            
            # 统计所有前景类别的像素数
            foreground_pixel_counts = {}
            for class_id in range(1, self.num_classes):  # 跳过背景类(0)
                pixel_count = np.sum(mask_np == class_id)
                if pixel_count > 0:
                    foreground_pixel_counts[class_id] = pixel_count
            
            # 如果有前景像素，将样本归类到主要缺陷类别（像素数最多的前景类）
            if foreground_pixel_counts:
                main_class = max(foreground_pixel_counts, key=foreground_pixel_counts.get)
                main_pixel_count = foreground_pixel_counts[main_class]
                class_pixel_counts[main_class].append((idx, main_pixel_count))
        
        # 按像素数降序排序（优先选择该类别像素多的样本）
        class_indices = {}
        for class_id in range(self.num_classes):
            if class_id in class_pixel_counts:
                # 按像素数降序排序
                sorted_samples = sorted(class_pixel_counts[class_id], 
                                      key=lambda x: x[1], reverse=True)
                class_indices[class_id] = [idx for idx, _ in sorted_samples]
            else:
                class_indices[class_id] = []
        
        # 输出类别分布（更详细）
        print("\n类别分布统计（按主要缺陷类别）:")
        for class_id in range(self.num_classes):
            count = len(class_indices[class_id])
            if class_id == 0:
                print(f"  类别 {class_id} (背景): {count} 个样本")
            else:
                print(f"  类别 {class_id} (缺陷{class_id}): {count} 个样本")
        
        return class_indices
    
    def sample_k_shot(self, k_shot, seed=42):
        """采样k-shot数据子集（每个缺陷类别采样k个样本）"""
        random.seed(seed)
        np.random.seed(seed)
        
        selected_indices = []
        
        print(f"\n正在采样 {k_shot}-shot 数据子集（每个缺陷类别选{k_shot}个主要样本）...")
        # 只对缺陷类别(1-4)进行采样，跳过背景类(0)
        for class_id in range(1, self.num_classes):
            class_samples = self.class_indices[class_id]
            
            if len(class_samples) >= k_shot:
                # 选择前k个样本（已按像素数降序排列）
                sampled = class_samples[:k_shot]
                selected_indices.extend(sampled)
                print(f"  缺陷类别 {class_id}: 选择了 {k_shot} 个样本（该类别像素数最多的）")
            else:
                # 不足k个样本时，全部选择
                selected_indices.extend(class_samples)
                print(f"  警告: 缺陷类别 {class_id} 只有 {len(class_samples)} 个样本，少于所需的 {k_shot} 个")
                print(f"  缺陷类别 {class_id}: 选择了 {len(class_samples)} 个样本")
        
        # 去重（因为有些样本可能包含多个类别）
        selected_indices = list(set(selected_indices))
        print(f"总共选择了 {len(selected_indices)} 个样本用于 {k_shot}-shot 训练（去重后）")
        return selected_indices


def create_enhanced_few_shot_dataset(data_dir, k_shot, seed=42, augment_factor=10, use_augmentation=True):
    """创建增强版few-shot数据集
    
    Args:
        data_dir: 数据集目录
        k_shot: few-shot数量
        seed: 随机种子
        augment_factor: 数据增强倍数
        use_augmentation: 是否使用数据增强（默认True）
    """
    # 创建完整数据集(使用所有20张图片,不划分训练/测试)
    full_dataset = Dataset3570Final(
        root=data_dir,
        mode="all",  # 使用全部数据,不划分
        target_size=(547, 1032),
        normalize=not use_augmentation,  # 如果不增强则直接标准化,否则在增强数据集中处理
        seed=seed
    )
    
    # 采样few-shot样本
    sampler = FewShotSampler(full_dataset)
    selected_indices = sampler.sample_k_shot(k_shot, seed)
    
    if use_augmentation:
        # 创建增强数据集
        enhanced_dataset = EnhancedFewShotDataset(
            base_dataset=full_dataset,
            selected_indices=selected_indices,
            augment_factor=augment_factor
        )
        print(f"增强后数据集大小: {len(enhanced_dataset)} 个样本 (原始 {len(selected_indices)} × {augment_factor})")
    else:
        # 不使用增强,直接使用选中的样本子集
        enhanced_dataset = Subset(full_dataset, selected_indices)
        print(f"数据集大小: {len(enhanced_dataset)} 个样本（未使用数据增强）")
    
    return enhanced_dataset, full_dataset


def train_enhanced_few_shot_model(
    dataset, 
    val_dataset,  # 新增：独立验证集
    k_shot, 
    epochs, 
    batch_size, 
    output_dir, 
    use_layers='all',
    use_hypergraph=False,
    fusion_strategy='sequential',
    use_class_weights=False,
    use_augmentation=True,
    use_amp=True,
    ce_weight=1.0,
    dice_weight=1.0,
    focal_weight=0.5,
    num_classes=5,
    val_interval=10,
    dataset_type='tn3k'
):
    """训练增强版few-shot模型
    
    Args:
        dataset: 训练数据集
        val_dataset: 验证数据集（独立于训练集，用于评估泛化能力）
        k_shot: few-shot数量
        epochs: 训练轮数
        batch_size: 批次大小
        output_dir: 输出目录
        use_layers: 使用的特征层 ('all' 或 '6_9')
        use_hypergraph: 是否使用超图GCN
        fusion_strategy: 模块融合策略
        use_class_weights: 是否使用类别权重
        use_augmentation: 是否使用数据增强
        use_amp: 是否使用混合精度训练
        ce_weight: CrossEntropy损失权重
        dice_weight: Dice损失权重
        focal_weight: Focal损失权重
        num_classes: 类别数 (TN3K=5, ViSA=2)
        val_interval: 验证间隔(每N个epoch验证一次)
        dataset_type: 数据集类型 ('tn3k' 或 'visa')
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"使用设备: {device}")
    logger.info(f"数据集类型: {dataset_type}")
    logger.info(f"类别数: {num_classes}")
    logger.info(f"训练样本数: {len(dataset)}")
    logger.info(f"批次大小: {batch_size}")
    logger.info(f"训练轮数: {epochs}")
    logger.info(f"验证间隔: 每{val_interval}个epoch")
    logger.info(f"数据增强: {'启用' if use_augmentation else '禁用'}")
    logger.info(f"混合精度训练: {'启用' if use_amp else '禁用'}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    # 创建模型
    repo_dir = './dinov3'
    dino_ckpt = './web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
    backbone = torch.hub.load(repo_dir, 'dinov3_vits16', source='local', weights=dino_ckpt)
    
    # 根据是否使用增强模块选择模型
    use_enhanced = use_hypergraph
    
    if use_enhanced:
        from dpt_enhanced import DPTEnhanced
        model = DPTEnhanced(
            encoder_size='small',
            nclass=num_classes,
            features=256,
            out_channels=[96, 192, 384, 768],
            use_bn=False,
            backbone=backbone,
            use_layers=use_layers,
            use_hypergraph=use_hypergraph,
            fusion_strategy=fusion_strategy
        ).to(device)
    else:
        from dpt import DPT
        model = DPT(
            encoder_size='small',
            nclass=num_classes,
            features=256,
            out_channels=[96, 192, 384, 768],
            use_bn=False,
            backbone=backbone,
            use_layers=use_layers
        ).to(device)
    
    if use_layers == 'all':
        logger.info(f"模型配置: 使用第3、6、9、12层特征（4层融合）")
    else:
        logger.info(f"模型配置: 使用第6、9层特征（2层融合）")
    
    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 混合精度训练
    scaler = GradScaler() if use_amp else None
    if use_amp:
        logger.info("启用混合精度训练（AMP）")
    
    # 计算类别权重（如果需要）
    class_weights_tensor = None
    if use_class_weights:
        logger.info("计算类别权重（处理类别不平衡）...")
        class_counts = np.zeros(num_classes)
        for _, mask, _ in dataset:
            mask_np = mask.squeeze().numpy().astype(int)
            for cls in range(num_classes):
                class_counts[cls] += np.sum(mask_np == cls)
        
        # 检查是否有类别像素数为0
        logger.info(f"原始类别像素统计:")
        for cls in range(num_classes):
            logger.info(f"  类别 {cls}: {class_counts[cls]:.0f} 像素")
        
        # 改进的权重计算：对于像素数为0的类别，给予中等权重
        total_pixels = np.sum(class_counts)
        class_weights = np.ones(num_classes)
        
        for cls in range(num_classes):
            if class_counts[cls] > 0:
                # 有像素的类别：使用逆频率权重
                class_weights[cls] = total_pixels / (num_classes * class_counts[cls])
            else:
                # 像素数为0的类别：给予平均权重的2倍（鼓励学习）
                if total_pixels > 0:
                    avg_count = total_pixels / num_classes
                    class_weights[cls] = total_pixels / (num_classes * avg_count) * 2
                else:
                    class_weights[cls] = 1.0
        
        # 归一化权重（避免过大或过小）
        class_weights = class_weights / class_weights.sum() * num_classes
        
        # 限制权重范围在[0.1, 10]之间
        class_weights = np.clip(class_weights, 0.1, 10.0)
        
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
        
        logger.info(f"最终类别权重:")
        for cls in range(num_classes):
            logger.info(f"  类别 {cls}: {class_counts[cls]:.0f} 像素 ({class_counts[cls]/total_pixels*100:.2f}%), 权重: {class_weights_tensor[cls]:.4f}")
    
    # 创建复杂损失函数
    criterion = MultiClassSegmentationLoss(
        num_classes=num_classes,
        ce_weight=ce_weight,
        dice_weight=dice_weight,
        focal_weight=focal_weight,
        class_weights=class_weights_tensor
    ).to(device)
    
    logger.info(f"损失函数配置: CE权重={ce_weight}, Dice权重={dice_weight}, Focal权重={focal_weight}")
    
    # 训练循环
    model.train()
    best_loss = float('inf')
    best_miou = 0.0
    
    logger.info(f"开始训练{'增强版' if use_augmentation else ''} {k_shot}-shot 模型...")
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_dice_loss = 0.0
        epoch_focal_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (images, masks, _) in enumerate(pbar):
            images = images.to(device)
            masks = masks.squeeze(1).long().to(device)  # 移除多余维度并转换为Long类型
            
            optimizer.zero_grad()
            
            if use_amp:
                with autocast(device_type='cuda'):
                    outputs = model(images)
                    # 调整输出尺寸到mask尺寸
                    if outputs.shape[-2:] != masks.shape[-2:]:
                        outputs = F.interpolate(outputs, size=masks.shape[-2:], 
                                              mode='bilinear', align_corners=False)
                    loss, loss_dict = criterion(outputs, masks)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                # 调整输出尺寸到mask尺寸
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = F.interpolate(outputs, size=masks.shape[-2:], 
                                          mode='bilinear', align_corners=False)
                loss, loss_dict = criterion(outputs, masks)
                
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss_dict['total_loss']
            epoch_ce_loss += loss_dict['ce_loss']
            epoch_dice_loss += loss_dict['dice_loss']
            epoch_focal_loss += loss_dict['focal_loss']
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f"{loss_dict['total_loss']:.4f}",
                'CE': f"{loss_dict['ce_loss']:.4f}",
                'Dice': f"{loss_dict['dice_loss']:.4f}",
                'Focal': f"{loss_dict['focal_loss']:.4f}"
            })
        
        scheduler.step()
        
        avg_loss = epoch_loss / num_batches
        avg_ce = epoch_ce_loss / num_batches
        avg_dice = epoch_dice_loss / num_batches
        avg_focal = epoch_focal_loss / num_batches
        
        # 每val_interval个epoch进行一次验证评估
        if (epoch + 1) % val_interval == 0 or epoch == epochs - 1:
            logger.info(f"\n进行验证评估（使用完整独立验证集）...")
            model.eval()
            
            if dataset_type == 'visa':
                # ViSA数据集: 二值分割，计算前景/背景的IoU, Dice, HD95
                total_fg_iou = 0.0
                total_fg_dice = 0.0
                total_fg_hd95 = 0.0
                total_bg_iou = 0.0
                total_bg_dice = 0.0
                total_bg_hd95 = 0.0
                n_samples = 0
                
                with torch.no_grad():
                    for idx in range(len(val_dataset)):
                        image, mask, _ = val_dataset[idx]
                        image = image.unsqueeze(0).to(device)
                        
                        if use_amp:
                            with autocast(device_type='cuda'):
                                output = model(image)
                                if output.shape[-2:] != mask.shape[-2:]:
                                    output = F.interpolate(output, size=mask.shape[-2:], 
                                                         mode='bilinear', align_corners=False)
                        else:
                            output = model(image)
                            if output.shape[-2:] != mask.shape[-2:]:
                                output = F.interpolate(output, size=mask.shape[-2:], 
                                                     mode='bilinear', align_corners=False)
                        
                        # 预测 (0=背景, 1=前景)
                        pred = torch.argmax(output, dim=1)
                        
                        # 计算前景指标
                        pred_fg = pred.squeeze().cpu()
                        gt_fg = mask.squeeze()
                        
                        # IoU
                        intersection_fg = ((pred_fg == 1) & (gt_fg == 1)).sum().float()
                        union_fg = ((pred_fg == 1) | (gt_fg == 1)).sum().float()
                        iou_fg = (intersection_fg / (union_fg + 1e-6)).item()
                        
                        # Dice
                        dice_fg = (2 * intersection_fg / (pred_fg.sum() + gt_fg.sum() + 1e-6)).item()
                        
                        # HD95 (前景)
                        hd95_fg = compute_hd95(pred_fg, gt_fg)
                        
                        # 计算背景指标
                        pred_bg = (pred == 0).squeeze().cpu()
                        gt_bg = (mask == 0).squeeze()
                        
                        # IoU
                        intersection_bg = ((pred_bg == 1) & (gt_bg == 1)).sum().float()
                        union_bg = ((pred_bg == 1) | (gt_bg == 1)).sum().float()
                        iou_bg = (intersection_bg / (union_bg + 1e-6)).item()
                        
                        # Dice
                        dice_bg = (2 * intersection_bg / (pred_bg.sum() + gt_bg.sum() + 1e-6)).item()
                        
                        # HD95 (背景)
                        hd95_bg = compute_hd95(pred_bg, gt_bg)
                        
                        total_fg_iou += iou_fg
                        total_fg_dice += dice_fg
                        total_fg_hd95 += hd95_fg
                        total_bg_iou += iou_bg
                        total_bg_dice += dice_bg
                        total_bg_hd95 += hd95_bg
                        n_samples += 1
                
                # 计算平均值
                avg_fg_iou = total_fg_iou / n_samples * 100
                avg_fg_dice = total_fg_dice / n_samples * 100
                avg_fg_hd95 = total_fg_hd95 / n_samples
                avg_bg_iou = total_bg_iou / n_samples * 100
                avg_bg_dice = total_bg_dice / n_samples * 100
                avg_bg_hd95 = total_bg_hd95 / n_samples
                
                # mIoU, mDice, mHD95
                val_miou = (avg_fg_iou + avg_bg_iou) / 2
                val_mdice = (avg_fg_dice + avg_bg_dice) / 2
                val_mhd95 = (avg_fg_hd95 + avg_bg_hd95) / 2
                
                logger.info(f"\n{'='*100}")
                logger.info(f"Epoch [{epoch+1}/{epochs}] 验证结果:")
                logger.info(f"{'-'*100}")
                logger.info(f"Train Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, Dice: {avg_dice:.4f}, Focal: {avg_focal:.4f})")
                logger.info(f"LR: {scheduler.get_last_lr()[0]:.6f}")
                logger.info(f"{'-'*100}")
                logger.info(f"前景   - IoU: {avg_fg_iou:.2f}%, Dice: {avg_fg_dice:.2f}%, HD95: {avg_fg_hd95:.2f}")
                logger.info(f"背景   - IoU: {avg_bg_iou:.2f}%, Dice: {avg_bg_dice:.2f}%, HD95: {avg_bg_hd95:.2f}")
                logger.info(f"{'-'*100}")
                logger.info(f"mIoU: {val_miou:.2f}%, mDice: {val_mdice:.2f}%, mHD95: {val_mhd95:.2f}")
                logger.info(f"{'='*100}\n")
                
            else:
                # TN3K数据集: 多类别分割
                all_val_preds = []
                all_val_gts = []
                
                with torch.no_grad():
                    for idx in range(len(val_dataset)):
                        image, mask, _ = val_dataset[idx]
                        image = image.unsqueeze(0).to(device)
                        mask_np = mask.squeeze().numpy().astype(int)
                        
                        if use_amp:
                            with autocast(device_type='cuda'):
                                output = model(image)
                                if output.shape[-2:] != mask.shape[-2:]:
                                    output = F.interpolate(output, size=mask.shape[-2:], 
                                                         mode='bilinear', align_corners=False)
                        else:
                            output = model(image)
                            if output.shape[-2:] != mask.shape[-2:]:
                                output = F.interpolate(output, size=mask.shape[-2:], 
                                                     mode='bilinear', align_corners=False)
                        
                        pred = torch.argmax(output, dim=1).cpu().numpy()[0]
                        all_val_preds.append(pred.flatten())
                        all_val_gts.append(mask_np.flatten())
                
                all_val_preds_flat = np.concatenate(all_val_preds)
                all_val_gts_flat = np.concatenate(all_val_gts)
                val_metrics_dict = calculate_segmentation_metrics(
                    all_val_preds_flat, 
                    all_val_gts_flat, 
                    num_classes=num_classes, 
                    ignore_background=True
                )
                val_miou = val_metrics_dict['mIoU']
                
                logger.info(f"Epoch [{epoch+1}/{epochs}]:")
                logger.info(f"  Train Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, Dice: {avg_dice:.4f}, Focal: {avg_focal:.4f})")
                logger.info(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
                logger.info(f"  Val mIoU: {val_miou:.4f}")
                logger.info(f"  Val Pixel Acc: {val_metrics_dict['pixel_accuracy']:.4f}")
                logger.info(f"  Val False Alarm: {val_metrics_dict['false_alarm_rate']:.4f}")
                logger.info(f"  Val Miss Rate: {val_metrics_dict['miss_rate']:.4f}")
            
            model.train()
            
            # 保存最佳模型（基于mIoU）
            if val_miou > best_miou:
                best_miou = val_miou
                model_save_path = os.path.join(output_dir, f'best_model_enhanced_{k_shot}shot.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'loss': avg_loss,
                    'val_miou': val_miou,
                    'k_shot': k_shot,
                    'num_classes': num_classes,
                    'dataset_type': dataset_type,
                    'model_config': {
                        'encoder_size': 'small',
                        'nclass': num_classes,
                        'features': 256,
                        'out_channels': [96, 192, 384, 768],
                        'use_bn': False,
                        'use_layers': use_layers,
                        'use_hypergraph': use_hypergraph,
                        'fusion_strategy': fusion_strategy
                    }
                }, model_save_path)
                logger.info(f"  ✓ 保存最佳模型 (Val mIoU: {val_miou:.2f}%) 到: {model_save_path}")
        else:
            logger.info(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} (CE: {avg_ce:.4f}, Dice: {avg_dice:.4f}, Focal: {avg_focal:.4f}), LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # 也保存最后一个epoch的模型
        if epoch == epochs - 1:
            last_model_path = os.path.join(output_dir, f'last_model_enhanced_{k_shot}shot.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'k_shot': k_shot,
                'num_classes': num_classes,
                'dataset_type': dataset_type,
                'model_config': {
                    'encoder_size': 'small',
                    'nclass': num_classes,
                    'features': 256,
                    'out_channels': [96, 192, 384, 768],
                    'use_bn': False,
                    'use_layers': use_layers,
                    'use_hypergraph': use_hypergraph,
                    'fusion_strategy': fusion_strategy
                }
            }, last_model_path)
            logger.info(f"  保存最终模型到: {last_model_path}")
    
    total_time = time.time() - start_time
    logger.info(f"\n训练完成！")
    logger.info(f"总训练时间: {total_time/3600:.2f}小时 ({total_time/60:.1f}分钟)")
    logger.info(f"最佳验证mIoU: {best_miou:.4f}")
    return model_save_path


def test_enhanced_few_shot_model(model_path, test_dataset, k_shot, exp_dir, num_samples=20, 
                                use_layers='all', use_hypergraph=False,
                                fusion_strategy='sequential',
                                num_classes=5, dataset_type='tn3k'):
    """测试增强版few-shot模型并生成可视化结果，同时计算评估指标
    
    Args:
        model_path: 模型路径
        test_dataset: 测试数据集
        k_shot: few-shot数量
        exp_dir: 实验目录
        num_samples: 测试样本数量，默认20
        use_layers: 使用的特征层
        use_hypergraph: 是否使用超图GCN
        fusion_strategy: 模块融合策略
        num_classes: 类别数
        dataset_type: 数据集类型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"\n测试增强版 {k_shot}-shot 模型...")
    logger.info(f"正在加载模型: {model_path}")
    
    # 检查是否使用增强模块
    use_enhanced = use_hypergraph
    
    # 创建模型
    repo_dir = './dinov3'
    dino_ckpt = './web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
    backbone = torch.hub.load(repo_dir, 'dinov3_vits16', source='local', weights=dino_ckpt)
    
    if use_enhanced:
        from dpt_enhanced import DPTEnhanced
        model = DPTEnhanced(
            encoder_size='small',
            nclass=5,
            features=256,
            out_channels=[96, 192, 384, 768],
            use_bn=False,
            backbone=backbone,
            use_layers=use_layers,
            use_hypergraph=use_hypergraph,
            fusion_strategy=fusion_strategy
        ).to(device)
    else:
        from dpt import DPT
        model = DPT(
            encoder_size='small',
            nclass=5,
            features=256,
            out_channels=[96, 192, 384, 768],
            use_bn=False,
            backbone=backbone,
            use_layers=use_layers
        ).to(device)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 创建结果目录
    results_dir = os.path.join(exp_dir, f'{k_shot}shot_enhanced_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 类别名称
    class_names = ['Background', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
    
    # 累积指标
    all_predictions = []
    all_ground_truths = []
    
    # 随机选择测试样本（确保可重复）
    random.seed(42)
    test_indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))
    
    print(f"\n开始测试 {len(test_indices)} 个样本...")
    
    with torch.no_grad():
        for i, idx in enumerate(test_indices):
            image, mask, meta = test_dataset[idx]
            image = image.unsqueeze(0).to(device)  # 添加batch维度
            
            # 模型预测（使用与验证相同的方式，不使用背景抑制）
            output = model(image)
            if output.shape[-2:] != mask.shape[-2:]:
                output = F.interpolate(output, size=mask.shape[-2:], 
                                     mode='bilinear', align_corners=False)
            pred = torch.argmax(output, dim=1).cpu().numpy()[0]
            
            # 获取原始图像路径用于显示
            original_image_path = None
            if 'image_path' in meta:
                original_image_path = meta['image_path']
            elif hasattr(meta, 'get'):
                original_image_path = meta.get('image_path')
            
            # 准备可视化数据
            if original_image_path and os.path.exists(original_image_path):
                # 直接读取原始图像，保持原始亮度
                import cv2
                image_bgr = cv2.imread(original_image_path)
                image_np = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                image_np = cv2.resize(image_np, (mask.shape[-1], mask.shape[-2]))
                image_np = image_np.astype(np.float32) / 255.0  # 只做简单的0-1归一化
            else:
                # 备用方案：反标准化预处理的图像
                image_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)
                # ImageNet反标准化
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image_np = image_np * std + mean  # 反标准化
                image_np = np.clip(image_np, 0, 1)
            
            mask_np = mask.squeeze().cpu().numpy()
            pred_np = pred
            
            # 收集预测和真实标签用于指标计算
            all_predictions.append(pred_np)
            all_ground_truths.append(mask_np)
            
            # 计算单张图像的指标
            sample_metrics = calculate_segmentation_metrics(pred_np, mask_np, num_classes=5, ignore_background=True)
            
            print(f"\n  样本 {i+1}/{len(test_indices)}:")
            print(f"    图像形状: {image_np.shape}")
            print(f"    预测mask形状: {pred_np.shape}")
            print(f"    真实mask形状: {mask_np.shape}")
            print(f"    mIoU: {sample_metrics['mIoU']:.4f}")
            print(f"    像素准确率: {sample_metrics['pixel_accuracy']:.4f}")
            print(f"    过检率: {sample_metrics['false_alarm_rate']:.4f}")
            print(f"    漏检率: {sample_metrics['miss_rate']:.4f}")
            
            # 保存结果
            result_path = os.path.join(results_dir, f'enhanced_test_sample_{i+1}.png')
            visualize_enhanced_prediction(image_np, pred_np, mask_np, result_path, k_shot)
            
            print(f"    结果保存到: {result_path}")
    
    # 计算整体指标
    print(f"\n{'='*80}")
    print(f"计算所有测试样本的整体指标...")
    print(f"{'='*80}")
    
    all_predictions_flat = np.concatenate([p.flatten() for p in all_predictions])
    all_ground_truths_flat = np.concatenate([g.flatten() for g in all_ground_truths])
    
    overall_metrics = calculate_segmentation_metrics(
        all_predictions_flat.reshape(-1, 1).squeeze(),
        all_ground_truths_flat.reshape(-1, 1).squeeze(),
        num_classes=5,
        ignore_background=True
    )
    
    # 打印详细指标
    print_metrics(overall_metrics, class_names=class_names, show_per_class=True)
    
    # 保存指标到文件
    metrics_file = os.path.join(results_dir, 'evaluation_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Few-shot模型评估指标 ({k_shot}-shot)\n")
        f.write("="*80 + "\n\n")
        f.write("【总体指标】\n")
        f.write(f"mIoU (平均交并比):          {overall_metrics['mIoU']:.4f} ({overall_metrics['mIoU']*100:.2f}%)\n")
        f.write(f"像素准确率 (Pixel Accuracy): {overall_metrics['pixel_accuracy']:.4f} ({overall_metrics['pixel_accuracy']*100:.2f}%)\n")
        f.write(f"平均精确率 (Precision):      {overall_metrics['mean_precision']:.4f} ({overall_metrics['mean_precision']*100:.2f}%)\n")
        f.write(f"平均召回率 (Recall):         {overall_metrics['mean_recall']:.4f} ({overall_metrics['mean_recall']*100:.2f}%)\n")
        f.write(f"平均F1分数:                  {overall_metrics['mean_f1']:.4f} ({overall_metrics['mean_f1']*100:.2f}%)\n")
        f.write(f"过检率 (False Alarm Rate):   {overall_metrics['false_alarm_rate']:.4f} ({overall_metrics['false_alarm_rate']*100:.2f}%)\n")
        f.write(f"漏检率 (Miss Rate):          {overall_metrics['miss_rate']:.4f} ({overall_metrics['miss_rate']*100:.2f}%)\n")
        f.write(f"平衡准确率:                  {overall_metrics['balanced_accuracy']:.4f} ({overall_metrics['balanced_accuracy']*100:.2f}%)\n\n")
        
        f.write("【各类别指标】\n")
        f.write(f"{'类别':<15} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'TP':>8} {'FP':>8} {'FN':>8}\n")
        f.write("-" * 80 + "\n")
        
        for i, name in enumerate(class_names):
            f.write(f"{name:<15} "
                  f"{overall_metrics['iou_per_class'][i]:>7.4f} "
                  f"{overall_metrics['precision_per_class'][i]:>10.4f} "
                  f"{overall_metrics['recall_per_class'][i]:>8.4f} "
                  f"{overall_metrics['f1_per_class'][i]:>8.4f} "
                  f"{int(overall_metrics['tp'][i]):>8d} "
                  f"{int(overall_metrics['fp'][i]):>8d} "
                  f"{int(overall_metrics['fn'][i]):>8d}\n")
    
    print(f"\n指标已保存到: {metrics_file}")
    
    return overall_metrics


def visualize_enhanced_prediction(image, pred_mask, gt_mask, save_path, k_shot):
    """可视化增强版预测结果"""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    # 创建自定义颜色映射（5个类别：背景+4种缺陷）
    # 类别0=背景(黑), 类别1=缺陷1(橙), 类别2=缺陷2(绿), 类别3=缺陷3(红), 类别4=缺陷4(紫)
    colors = ['black', 'orange', 'green', 'red', 'purple']
    cmap = mcolors.ListedColormap(colors)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原图 - 确保显示原始亮度
    axes[0].imshow(image, vmin=0, vmax=1)  # 明确指定显示范围
    axes[0].set_title('Original Image (原始亮度)', fontsize=12)
    axes[0].axis('off')
    
    # 预测mask（背景黑色，4种缺陷用不同颜色）
    axes[1].imshow(pred_mask, cmap=cmap, vmin=0, vmax=4)
    axes[1].set_title(f'Enhanced Prediction ({k_shot}-shot)', fontsize=12)
    axes[1].axis('off')
    
    # 真实mask（背景黑色，4种缺陷用不同颜色）
    axes[2].imshow(gt_mask, cmap=cmap, vmin=0, vmax=4)
    axes[2].set_title('Ground Truth', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='增强版Few-shot SegDINO训练 - 支持TN3K和ViSA数据集')
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, default='tn3k', 
                        choices=['tn3k', 'visa'],
                        help='数据集类型')
    parser.add_argument('--data_dir', type=str, default='./segdata/3570_20250903_far',
                        help='数据集目录 (TN3K或ViSA根目录)')
    parser.add_argument('--visa_category', type=str, default=None,
                        help='ViSA类别名称 (如candle, capsules等), None表示所有类别')
    parser.add_argument('--visa_csv', type=str, default='split_csv/2cls_fewshot.csv',
                        help='ViSA CSV文件路径')
    parser.add_argument('--include_normal', action='store_true',
                        help='ViSA: 是否包含正常样本（默认False，只用异常样本）')
    parser.add_argument('--output_dir', type=str, default='./runs/enhanced_few_shot_experiments',
                        help='输出目录')
    
    # Few-shot参数
    parser.add_argument('--k_shots', type=int, nargs='+', default=[1, 3, 5],
                        help='要测试的shot数量列表')
    parser.add_argument('--num_test_samples', type=int, default=20,
                        help='测试时使用的样本数量（默认20，增加可获得更可靠的统计）')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数（小样本学习建议100-200轮）')
    parser.add_argument('--val_interval', type=int, default=10,
                        help='验证间隔(每N个epoch验证一次)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--augment_factor', type=int, default=10,
                        help='数据增强倍数')
    parser.add_argument('--use_augmentation', action='store_true',
                        help='启用数据增强（默认不启用）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--mode', type=str, default='train_test',
                        choices=['train', 'test', 'train_test'],
                        help='运行模式：train（仅训练）、test（仅测试）、train_test（训练+测试）')
    parser.add_argument('--use_layers', type=str, default='all',
                        choices=['all', '6_9'],
                        help='使用的特征层：all=4层(第3,6,9,12层), 6_9=2层(第6,9层)')
    
    # 增强模块参数
    parser.add_argument('--use_hypergraph', action='store_true',
                        help='启用超图GCN模块')
    parser.add_argument('--fusion_strategy', type=str, default='sequential',
                        choices=['sequential', 'parallel'],
                        help='模块融合策略：sequential=顺序应用, parallel=并行融合')
    parser.add_argument('--use_class_weights', action='store_true',
                        help='启用类别权重（处理类别不平衡，默认不启用）')
    parser.add_argument('--use_amp', action='store_true',
                        help='启用混合精度训练（默认不启用）')
    
    # 损失函数权重参数
    parser.add_argument('--ce_weight', type=float, default=1.0,
                        help='CrossEntropy损失权重（默认1.0）')
    parser.add_argument('--dice_weight', type=float, default=1.0,
                        help='Dice损失权重（默认1.0）')
    parser.add_argument('--focal_weight', type=float, default=0.5,
                        help='Focal损失权重（默认0.5）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=" * 60)
    print(f"增强版Few-shot SegDINO训练 - {args.dataset.upper()}数据集")
    print("=" * 60)
    print(f"数据集类型: {args.dataset}")
    if args.dataset == 'visa':
        print(f"ViSA类别: {args.visa_category if args.visa_category else '所有类别'}")
        print(f"包含正常样本: {'是' if args.include_normal else '否'}")
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"运行模式: {args.mode}")
    print(f"K-shot设置: {args.k_shots}")
    print(f"测试样本数: {args.num_test_samples}")
    print(f"数据增强: {'启用' if args.use_augmentation else '禁用'}")
    if args.use_augmentation:
        print(f"数据增强倍数: {args.augment_factor}")
    print(f"训练轮数: {args.epochs}")
    print(f"验证间隔: 每{args.val_interval}个epoch")
    print(f"使用特征层: {args.use_layers} ({'4层融合' if args.use_layers == 'all' else '2层融合'})")
    print(f"类别权重: {'启用' if args.use_class_weights else '禁用'}")
    print(f"混合精度训练: {'启用' if args.use_amp else '禁用'}")
    print(f"损失函数权重: CE={args.ce_weight}, Dice={args.dice_weight}, Focal={args.focal_weight}")
    print(f"增强模块:")
    print(f"  - 超图GCN: {'启用' if args.use_hypergraph else '禁用'}")
    if args.use_hypergraph:
        print(f"  - 融合策略: {args.fusion_strategy}")
    print("=" * 60)
    
    # 创建测试数据集（根据数据集类型）
    if args.dataset == 'visa':
        # ViSA数据集: 二值分割 (0=背景, 1=前景)
        target_size = (518, 518)  # DINOv3标准尺寸
        test_dataset = ViSADataset(
            root=args.data_dir,
            csv_file=args.visa_csv,
            split='test',
            category=args.visa_category,
            target_size=target_size
        )
        num_classes = 2  # 二值分割
        logger.info(f"ViSA测试数据集大小: {len(test_dataset)} 个样本")
        logger.info(f"目标尺寸: {target_size}")
        logger.info(f"类别数: {num_classes} (二值分割: 0=背景, 1=前景)")
    else:
        # TN3K/3570数据集: 多类别分割
        target_size = (547, 1032)
        test_dataset = Dataset3570Final(
            root=args.data_dir,
            mode="test",
            target_size=target_size,
            normalize=True,
            seed=args.seed
        )
        num_classes = 5  # 多类别分割
        logger.info(f"测试数据集大小: {len(test_dataset)} 个样本")
        logger.info(f"目标尺寸: {target_size}")
        logger.info(f"类别数: {num_classes}")
    
    # 运行不同k-shot实验
    for k_shot in args.k_shots:
        logger.info(f"\n{'='*50}")
        logger.info(f"开始 {k_shot}-shot 增强训练实验")
        logger.info(f"{'='*50}")
        
        # 创建实验目录
        if args.dataset == 'visa' and args.visa_category:
            exp_dir = os.path.join(args.output_dir, args.dataset, args.visa_category, f'{k_shot}shot_enhanced')
        else:
            exp_dir = os.path.join(args.output_dir, f'{k_shot}shot_enhanced')
        os.makedirs(exp_dir, exist_ok=True)
        
        if args.mode in ['train', 'train_test']:
            # 创建训练数据集
            if args.dataset == 'visa':
                # ViSA数据集
                full_train_dataset = ViSADataset(
                    root=args.data_dir,
                    csv_file=args.visa_csv,
                    split='train',
                    category=args.visa_category,
                    target_size=target_size
                )
                
                # 采样k-shot
                selected_indices = sample_k_shot_visa(
                    full_train_dataset, 
                    k_shot, 
                    include_normal=args.include_normal,
                    verbose=True
                )
                
                # 创建子集
                train_dataset = Subset(full_train_dataset, selected_indices)
                logger.info(f"ViSA训练数据集: {len(train_dataset)} 个样本")
            else:
                # TN3K/3570数据集 - 使用原有的增强方法
                train_dataset, _ = create_enhanced_few_shot_dataset(
                    args.data_dir, k_shot, args.seed, args.augment_factor, args.use_augmentation
                )
            
            # 训练模型（传入测试集作为验证集，num_classes, val_interval）
            model_path = train_enhanced_few_shot_model(
                train_dataset,
                test_dataset,
                k_shot, 
                args.epochs, 
                args.batch_size, 
                exp_dir, 
                args.use_layers,
                args.use_hypergraph,
                args.fusion_strategy,
                args.use_class_weights,
                args.use_augmentation,
                args.use_amp,
                args.ce_weight,
                args.dice_weight,
                args.focal_weight,
                num_classes=num_classes,
                val_interval=args.val_interval,
                dataset_type=args.dataset
            )
        else:
            # 仅测试模式，查找已训练的模型
            model_path = os.path.join(exp_dir, f'best_model_enhanced_{k_shot}shot.pth')
            if not os.path.exists(model_path):
                print(f"警告：未找到模型文件 {model_path}，跳过 {k_shot}-shot 测试")
                continue
        
        if args.mode in ['test', 'train_test']:
            # 测试模型并生成可视化结果（传入增强模块参数）
            test_enhanced_few_shot_model(
                model_path, 
                test_dataset, 
                k_shot, 
                exp_dir,
                num_samples=args.num_test_samples,
                use_layers=args.use_layers,
                use_hypergraph=args.use_hypergraph,
                fusion_strategy=args.fusion_strategy,
                num_classes=num_classes,
                dataset_type=args.dataset
            )
        
        print(f"{k_shot}-shot 增强训练实验完成！")
        print(f"模型保存在: {model_path}")
    
    print(f"\n所有增强版few-shot实验完成！结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()