#!/usr/bin/env python3
"""
GLCM (Global-Local Calibration Module)
异常感知校准模块

改编自 AD-DINOv3: https://github.com/Kaisor-Yuan/AD-DINOv3
核心思想: 通过CLS token(全局基线)与patch tokens比对，识别并增强异常区域
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipAdapter(nn.Module):
    """轻量级适配器层"""
    def __init__(self, c_in, bottleneck=768):
        super(ClipAdapter, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(bottleneck, c_in, bias=False),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x):
        """
        Args:
            x: [B, N, C] 或 [B, C]
        Returns:
            bottleneck_features: [B, N, bottleneck] 或 [B, bottleneck]
            output_features: [B, N, C] 或 [B, C]
        """
        bottleneck = self.fc1(x)
        output = self.fc2(bottleneck)
        return bottleneck, output


class GLCMModule(nn.Module):
    """
    Global-Local Calibration Module (GLCM)
    异常感知校准模块
    
    功能:
    1. 使用CLS token(全局基线)与patch tokens比对
    2. 计算patch-cls相似度，低相似度 = 异常区域
    3. 增强异常区域的特征表
    
    工作流程:
    1. 提取全局特征作为"正常基线" (CLS token / 全局平均池化)
    2. 通过适配器层处理特征
    3. 计算patch-cls相似
    4. 取负相似度作为异常分数(不相似 → 高异常分数)
    5. 使用异常图增强异常区域特征
    
    免疫学对照:
    - CLS token = MHC-I呈递的自身抗原(正常基线)
    - Patch tokens = 组织采样
    - 低相似度 = 与自身不同 = 异常/外源
    - 特征增强 = 免疫细胞聚集
    """
    
    def __init__(self, feature_dim, bottleneck_dim=256, scale_factor=10.0):
        """
        Args:
            feature_dim: 特征维度 (如DINOv3的384)
            bottleneck_dim: 适配器瓶颈层维度
            scale_factor: 相似度缩放因
        """
        super(GLCMModule, self).__init__()
        
        self.feature_dim = feature_dim
        self.bottleneck_dim = bottleneck_dim
        self.scale_factor = scale_factor
        
        # CLS token适配
        self.cls_adapter = ClipAdapter(c_in=feature_dim, bottleneck=bottleneck_dim)
        
        # Patch tokens适配
        self.patch_adapter = ClipAdapter(c_in=feature_dim, bottleneck=bottleneck_dim)
        
        print(f"  GLCM模块: feature_dim={feature_dim}, bottleneck_dim={bottleneck_dim}, scale={scale_factor}")
    
    def forward(self, features):
        """
        Args:
            features: [B, N+1, C] - DINOv3特征 (包含1个CLS token + N个patch tokens)
                     或 [B, C, H, W] - 空间特征图
        
        Returns:
            enhanced_features: [B, C, H, W] - 异常感知增强后的特征
            anomaly_map: [B, 1, H, W] - 异常感知图(可选用于监督)
        """
        B = features.shape[0]
        
        # 判断输入格式
        if features.dim() == 4:  # [B, C, H, W]
            C, H, W = features.shape[1], features.shape[2], features.shape[3]
            # 转换为 [B, H*W, C]
            features_flat = features.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
            N = H * W
            
            # 提取全局特征作为CLS token (使用全局平均池化)
            cls_token = features.mean(dim=[2, 3])  # [B, C]
            patch_tokens = features_flat  # [B, H*W, C]
            
        elif features.dim() == 3:  # [B, N+1, C] - DINOv3原始格式
            # 分离CLS token和patch tokens
            cls_token = features[:, 0, :]  # [B, C]
            patch_tokens = features[:, 1:, :]  # [B, N, C]
            N = patch_tokens.shape[1]
            H = W = int(N ** 0.5)
        else:
            raise ValueError(f"不支持的特征维度: {features.shape}")
        
        # ===== 步骤1: 通过适配器处理=====
        # CLS token适配
        cls_bottleneck, cls_adapted = self.cls_adapter(cls_token)  # [B, bottleneck], [B, C]
        
        # Patch tokens适配
        patch_bottleneck, patch_adapted = self.patch_adapter(patch_tokens)  # [B, N, bottleneck], [B, N, C]
        
        # ===== 步骤2: 计算异常感知分数 =====
        # 归一化(类似MHC呈递的标准化处理
        cls_norm = F.normalize(cls_bottleneck, dim=-1)  # [B, bottleneck] - 正常基线
        patch_norm = F.normalize(patch_bottleneck, dim=-1)  # [B, N, bottleneck] - 局部采
        
        # 计算余弦相似度: [B, N, bottleneck] @ [B, bottleneck, 1] = [B, N, 1]
        # 相似度高 → 与全局一致 → 正常区域
        # 相似度低 → 与全局不同 → 异常区域
        similarity = self.scale_factor * torch.bmm(
            patch_norm, 
            cls_norm.unsqueeze(-1)
        )  # [B, N, 1]
        
        # ===== 步骤3: 生成异常感知图 =====
        # Reshape到空间维
        # 关键修改: 使用负相似度，使得与全局不相似的区域获得高异常分
        # 原理: 异常区域 = 与全局基线(正常模式)偏离的区
        anomaly_score = -similarity.squeeze(-1).view(B, 1, H, W)  # [B, 1, H, W]
        
        # Sigmoid激活得到异常概
        # 现在: 不相似 → anomaly_score高 → anomaly_map高
        anomaly_map = torch.sigmoid(anomaly_score)  # [B, 1, H, W]
        
        # ===== 步骤4: 特征增强 =====
        # 使用异常图加权原始特
        patch_adapted_spatial = patch_adapted.permute(0, 2, 1).view(B, -1, H, W)  # [B, C, H, W]
        
        # 异常区域增强: 与全局不相似的区域获得更多注意
        # (1 + anomaly_map) 使异常区域的特征被放
        enhanced_features = patch_adapted_spatial * (1 + anomaly_map)
        
        return enhanced_features, anomaly_map
    
    def get_anomaly_awareness_loss(self, anomaly_map, gt_mask):
        """
        计算异常感知损失（可用于监督GLCM
        
        Args:
            anomaly_map: [B, 1, H, W] - GLCM生成的异常图
            gt_mask: [B, 1, H, W] - 真实标签
        
        Returns:
            loss: scalar
        """
        # 确保尺寸匹配
        if anomaly_map.shape != gt_mask.shape:
            anomaly_map = F.interpolate(
                anomaly_map, 
                size=gt_mask.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # BCE loss
        bce_loss = F.binary_cross_entropy(anomaly_map, gt_mask)
        
        # Dice loss
        intersection = (anomaly_map * gt_mask).sum()
        union = anomaly_map.sum() + gt_mask.sum()
        dice_loss = 1 - (2 * intersection + 1e-6) / (union + 1e-6)
        
        return bce_loss + dice_loss


class MultiLayerGLCM(nn.Module):
    """
    多层GLCM模块
    在DINOv3的多个层后应用GLCM
    """
    
    def __init__(self, feature_dims, bottleneck_dim=256, num_layers=4):
        """
        Args:
            feature_dims: 各层特征维度列表，如[96, 192, 384, 768]
            bottleneck_dim: 适配器瓶颈维
            num_layers: 使用的层
        """
        super(MultiLayerGLCM, self).__init__()
        
        self.num_layers = num_layers
        
        # 为每一层创建独立的GLCM模块
        self.glcm_modules = nn.ModuleList([
            GLCMModule(
                feature_dim=feature_dims[i],
                bottleneck_dim=bottleneck_dim,
                scale_factor=10.0
            )
            for i in range(num_layers)
        ])
    
    def forward(self, features_list):
        """
        Args:
            features_list: list of [B, C_i, H_i, W_i] - 多层特征
        
        Returns:
            enhanced_features: list of [B, C_i, H_i, W_i] - 增强后的特征
            anomaly_maps: list of [B, 1, H_i, W_i] - 各层异常
        """
        enhanced_features = []
        anomaly_maps = []
        
        for i, features in enumerate(features_list):
            enhanced, anomaly_map = self.glcm_modules[i](features)
            enhanced_features.append(enhanced)
            anomaly_maps.append(anomaly_map)
        
        return enhanced_features, anomaly_maps


if __name__ == '__main__':
    """测试GLCM模块"""
    print("="*60)
    print("测试GLCM模块")
    print("="*60)
    
    # 测试单层GLCM
    print("\n1. 测试单层GLCM (空间特征图输入)")
    glcm = GLCMModule(feature_dim=384, bottleneck_dim=256)
    
    # 模拟输入: [B, C, H, W]
    x = torch.randn(2, 384, 32, 32)
    print(f"输入特征: {x.shape}")
    
    enhanced, anomaly_map = glcm(x)
    print(f"增强特征: {enhanced.shape}")
    print(f"异常图: {anomaly_map.shape}")
    
    # 测试DINOv3格式输入
    print("\n2. 测试单层GLCM (DINOv3 token格式)")
    x_tokens = torch.randn(2, 1025, 384)  # [B, 1+32*32, C]
    print(f"输入特征: {x_tokens.shape}")
    
    enhanced, anomaly_map = glcm(x_tokens)
    print(f"增强特征: {enhanced.shape}")
    print(f"异常图: {anomaly_map.shape}")
    
    # 测试多层GLCM
    print("\n3. 测试多层GLCM")
    multi_glcm = MultiLayerGLCM(
        feature_dims=[96, 192, 384, 768],
        bottleneck_dim=256,
        num_layers=4
    )
    
    # 模拟4层输
    features_list = [
        torch.randn(2, 96, 32, 32),
        torch.randn(2, 192, 32, 32),
        torch.randn(2, 384, 32, 32),
        torch.randn(2, 768, 32, 32),
    ]
    
    enhanced_list, anomaly_maps = multi_glcm(features_list)
    
    print("\n增强后的特征:")
    for i, enhanced in enumerate(enhanced_list):
        print(f"  层{i+1}: {enhanced.shape}")
    
    print("\n异常图:")
    for i, anomaly_map in enumerate(anomaly_maps):
        print(f"  层{i+1}: {anomaly_map.shape}")
    
    # 测试损失计算
    print("\n4. 测试异常感知损失")
    gt_mask = torch.randint(0, 2, (2, 1, 32, 32)).float()
    loss = glcm.get_anomaly_awareness_loss(anomaly_map, gt_mask)
    print(f"GLCM损失: {loss.item():.4f}")
    
    print("\n"+"="*60)
    print("✅ GLCM模块测试完成")
    print("="*60)
