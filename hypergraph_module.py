#!/usr/bin/env python3
"""
超图GCN模块
用于建模多节点之间的高阶关系
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HypergraphConv(nn.Module):
    """超图卷积层"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.theta = nn.Linear(in_channels, out_channels)
        self.phi = nn.Linear(out_channels, out_channels)  # 修复：输入应该是out_channels
        
    def forward(self, x, H):
        """
        Args:
            x: [B, N, C] 节点特征
            H: [N, E] 超边关联矩阵 (N个节点, E条超边)
        Returns:
            output: [B, N, C'] 增强后的节点特征
        """
        # 节点到超边的聚合
        x_theta = self.theta(x)  # [B, N, C']
        
        # 计算超边特征 - 归一化后聚合
        H_normalized = F.normalize(H.float(), p=1, dim=0)
        edge_features = torch.matmul(H_normalized.t(), x_theta)  # [B, E, C']
        
        # 超边到节点的传播
        x_phi = self.phi(edge_features)  # [B, E, C']
        output = torch.matmul(H_normalized, x_phi)  # [B, N, C']
        
        return output


class HypergraphFeatureEnhancer(nn.Module):
    """超图特征增强模块"""
    def __init__(self, in_channels, hidden_channels=256, num_layers=2, k_neighbors=8):
        super().__init__()
        self.in_channels = in_channels
        self.k_neighbors = k_neighbors
        
        # 超图卷积层
        self.hyper_convs = nn.ModuleList([
            HypergraphConv(
                in_channels if i == 0 else hidden_channels,
                hidden_channels
            ) for i in range(num_layers)
        ])
        
        # 输出投影
        self.out_proj = nn.Conv2d(hidden_channels, in_channels, 1)
        self.norm = nn.BatchNorm2d(in_channels)
        
    def build_hypergraph(self, features):
        """
        构建超图结构
        Args:
            features: [B, C, H, W]
        Returns:
            H: [N, E] 超边关联矩阵
            feat_flat: [B, N, C] 节点特征
        """
        B, C, H, W = features.shape
        N = H * W
        
        # 将特征reshape为节点
        feat_flat = features.flatten(2).permute(0, 2, 1)  # [B, N, C]
        
        # 计算节点间相似度
        feat_norm = F.normalize(feat_flat, p=2, dim=-1)
        similarity = torch.matmul(feat_norm, feat_norm.transpose(1, 2))  # [B, N, N]
        
        # 为每个节点找k个最近邻，构成超边
        _, topk_indices = torch.topk(similarity[0], self.k_neighbors, dim=-1)  # [N, k]
        
        # 构建超边矩阵
        H = torch.zeros(N, N, device=features.device)
        for i in range(N):
            H[topk_indices[i], i] = 1.0
            
        return H, feat_flat
    
    def forward(self, features):
        """
        Args:
            features: [B, C, H, W]
        Returns:
            enhanced_features: [B, C, H, W]
        """
        B, C, H, W = features.shape
        
        # 构建超图
        hypergraph, feat_nodes = self.build_hypergraph(features)
        
        # 超图卷积
        x = feat_nodes
        for i, conv in enumerate(self.hyper_convs):
            x_new = F.relu(conv(x, hypergraph))
            # 只在第一层之后做残差连接（因为维度已经变化）
            if i > 0:
                x = x_new + x  # 残差连接（hidden_channels + hidden_channels）
            else:
                x = x_new  # 第一层不做残差（因为维度从in_channels变为hidden_channels）
        
        # Reshape回空间维度
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        x = self.out_proj(x)
        x = self.norm(x)
        
        return features + x  # 最终残差连接
