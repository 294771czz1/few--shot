#!/usr/bin/env python3
"""
增强版DPT模型
融合GLCM、超图GCN、Internal Adapter模块
支持通过参数选择使用哪些模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import _make_scratch

# 导入增强模块
try:
    from glcm_module import GLCMModule
    GLCM_AVAILABLE = True
except ImportError:
    GLCM_AVAILABLE = False
    print("Warning: glcm_module not available")

try:
    from internal_adapter import create_internal_adapters
    INTERNAL_ADAPTER_AVAILABLE = True
except ImportError:
    INTERNAL_ADAPTER_AVAILABLE = False
    print("Warning: internal_adapter not available")

try:
    from hypergraph_module import HypergraphFeatureEnhancer
    HYPERGRAPH_AVAILABLE = True
except ImportError:
    HYPERGRAPH_AVAILABLE = False
    print("Warning: hypergraph_module not available")




class DPTHeadEnhanced(nn.Module):
    """增强版DPT解码头（集成GLCM + 超图GCN + Internal Adapter"""
    def __init__(
        self, 
        nclass,
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512],
        num_layers=2,
        # 增强模块开
        use_glcm=False,
        use_hypergraph=False,
        use_internal_adapter=False,
        fusion_strategy='sequential',
        # Internal Adapter配置
        internal_adapter_config=None
    ):
        super(DPTHeadEnhanced, self).__init__()
        self.num_layers = num_layers
        self.use_glcm = use_glcm and GLCM_AVAILABLE
        self.use_hypergraph = use_hypergraph and HYPERGRAPH_AVAILABLE
        self.use_internal_adapter = use_internal_adapter and INTERNAL_ADAPTER_AVAILABLE
        self.fusion_strategy = fusion_strategy
        
        # Internal Adapter配置
        self.internal_adapter_injector = None
        if internal_adapter_config:
            self.internal_adapter_config = internal_adapter_config
        else:
            # 默认配置
            self.internal_adapter_config = {
                'adapter_type': 'uniform',
                'target_layers': [2, 5, 8],
                'hidden_dim': 384,
                'bottleneck_dim': 64
            }
        
        # 投影
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        # === 模块1: GLCM (Global-Local Calibration Module) ===
        if self.use_glcm:
            print("  启用GLCM模块（异常感知校准）")
            # GLCM在投影后应用，所以使用out_channels的维
            self.glcm_modules = nn.ModuleList([
                GLCMModule(
                    feature_dim=out_channels[i],
                    bottleneck_dim=256,
                    scale_factor=10.0
                )
                for i in range(num_layers)
            ])
        
        # === 模块2: 超图GCN ===
        if self.use_hypergraph:
            print("  启用超图GCN模块")
            self.hypergraph_enhancers = nn.ModuleList([
                HypergraphFeatureEnhancer(out_channels[i], features)
                for i in range(num_layers)
            ])
        
        # 特征精炼
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        self.scratch.stem_transpose = None
        
        # 输出卷积
        self.scratch.output_conv = nn.Conv2d(features*num_layers, nclass, kernel_size=1, stride=1, padding=0)
    
    def apply_enhancements(self, x, layer_idx):
        """
        应用增强模块（按照时序处理）
        
        时序步骤:
        1. GLCM: 异常感知校准（引导关注异常区域）
        2. 超图GCN: 高阶空间关系建模
        """
        anomaly_map = None  # 用于存储GLCM生成的异常图（可选用于监督）
        
        if self.fusion_strategy == 'sequential':
            # 顺序应用：GLCM → 超图GCN
            
            # 步骤1: GLCM异常感知校准
            if self.use_glcm:
                x, anomaly_map = self.glcm_modules[layer_idx](x)
            
            # 步骤2: 超图GCN
            if self.use_hypergraph:
                x = self.hypergraph_enhancers[layer_idx](x)
        
        elif self.fusion_strategy == 'parallel':
            # 并行应用后融
            outputs = [x]
            
            if self.use_glcm:
                x_glcm, anomaly_map = self.glcm_modules[layer_idx](x)
                outputs.append(x_glcm)
            
            if self.use_hypergraph:
                outputs.append(self.hypergraph_enhancers[layer_idx](x))
            
            x = sum(outputs) / len(outputs)  # 平均融合
        
        return x, anomaly_map
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        anomaly_maps = []  # 收集各层的异常图
        
        for i, x in enumerate(out_features):
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            
            # 应用增强模块（包括GLCM
            x, anomaly_map = self.apply_enhancements(x, i)
            if anomaly_map is not None:
                anomaly_maps.append(anomaly_map)
            
            out.append(x)
        
        # 根据层数进行特征融合
        if self.num_layers == 2:
            # 2层融
            layer_1, layer_2 = out
            layer_1_rn = self.scratch.layer1_rn(layer_1)
            layer_2_rn = self.scratch.layer2_rn(layer_2)
            
            target_hw = layer_1_rn.shape[-2:]
            layer_2_up = F.interpolate(layer_2_rn, size=target_hw, mode="bilinear", align_corners=True)
            fused = torch.cat([layer_1_rn, layer_2_up], dim=1)
        else:
            # 4层融
            layer_1, layer_2, layer_3, layer_4 = out
            layer_1_rn = self.scratch.layer1_rn(layer_1)
            layer_2_rn = self.scratch.layer2_rn(layer_2)
            layer_3_rn = self.scratch.layer3_rn(layer_3)
            layer_4_rn = self.scratch.layer4_rn(layer_4)
            
            target_hw = layer_1_rn.shape[-2:]  
            layer_2_up = F.interpolate(layer_2_rn, size=target_hw, mode="bilinear", align_corners=True)
            layer_3_up = F.interpolate(layer_3_rn, size=target_hw, mode="bilinear", align_corners=True)
            layer_4_up = F.interpolate(layer_4_rn, size=target_hw, mode="bilinear", align_corners=True)
            fused = torch.cat([layer_1_rn, layer_2_up, layer_3_up, layer_4_up], dim=1)
        
        # 主分割输
        out = self.scratch.output_conv(fused)
        
        # ===== 可选：融合GLCM异常图到最终输出 =====
        if self.use_glcm and anomaly_maps:
            # 将多层anomaly_map融合
            # 先上采样到与out相同的尺
            anomaly_maps_upsampled = []
            for amap in anomaly_maps:
                if amap.shape[-2:] != out.shape[-2:]:
                    amap_up = F.interpolate(amap, size=out.shape[-2:], 
                                          mode='bilinear', align_corners=False)
                else:
                    amap_up = amap
                anomaly_maps_upsampled.append(amap_up)
            
            # 平均融合多层异常
            anomaly_map_fused = torch.mean(torch.stack(anomaly_maps_upsampled, dim=0), dim=0)
            
            # 返回主输出、异常图列表、融合后的异常图
            return out, anomaly_maps, anomaly_map_fused
        else:
            return out, None, None


class DPTEnhanced(nn.Module):
    """增强版DPT模型（集成GLCM + 超图GCN + Internal Adapter"""
    def __init__(
        self, 
        encoder_size='base', 
        nclass=2,
        features=128, 
        out_channels=[96, 192, 384, 768], 
        use_bn=False,
        backbone=None,
        use_layers='all',
        # 增强模块参数
        use_glcm=False,
        use_hypergraph=False,
        use_internal_adapter=False,
        fusion_strategy='sequential',
        # Internal Adapter配置
        internal_adapter_config=None
    ):
        super(DPTEnhanced, self).__init__()
        
        self.use_glcm = use_glcm
        self.use_hypergraph = use_hypergraph
        self.use_internal_adapter = use_internal_adapter
        
        # 打印模块使用情况
        print(f"创建增强版DPT模型:")
        print(f"  - 特征层配置 {use_layers}")
        print(f"  - Internal Adapter (骨干网络适配): {'启用' if use_internal_adapter else '禁用'}")
        print(f"  - GLCM (异常感知): {'启用' if use_glcm else '禁用'}")
        print(f"  - 超图GCN: {'启用' if use_hypergraph else '禁用'}")
        print(f"  - 融合策略: {fusion_strategy}")
        
        # 根据use_layers参数设置使用的层
        if use_layers == '6_9':
            # 只使用第6层和第9层（索引5和8）
            self.intermediate_layer_idx = {
                'small': [5, 8],
                'base': [5, 8], 
            }
            self.num_layers = 2
            # 只使用前2个out_channels
            out_channels = out_channels[:2] if len(out_channels) >= 2 else out_channels
        else:
            # 使用全部4层（第3、6、9、12层，索引2、5、8、11）
            self.intermediate_layer_idx = {
                'small': [2, 5, 8, 11],
                'base': [2, 5, 8, 11], 
            }
            self.num_layers = 4
        
        self.encoder_size = encoder_size
        self.backbone = backbone
        
        # === 注入Internal Adapters到DINOv3骨干网络 ===
        self.internal_adapter_injector = None
        if self.use_internal_adapter and INTERNAL_ADAPTER_AVAILABLE:
            print("\n  🔴 注入Internal Adapters到DINOv3骨干网络...")
            
            # 设置默认配置
            if internal_adapter_config is None:
                internal_adapter_config = {
                    'adapter_type': 'uniform',
                    'target_layers': [2, 5, 8],  # Block 3, 6, 9
                    'hidden_dim': backbone.embed_dim,  # 使用backbone的embed_dim
                    'bottleneck_dim': 64
                }
            
            # 创建并注入adapters
            self.internal_adapter_injector = create_internal_adapters(
                dino_model=backbone,
                **internal_adapter_config
            )
            print("  ✅ Internal Adapters注入完成\n")
        
        self.head = DPTHeadEnhanced(
            nclass, 
            self.backbone.embed_dim, 
            features, 
            use_bn, 
            out_channels=out_channels, 
            num_layers=self.num_layers,
            use_glcm=use_glcm,
            use_hypergraph=use_hypergraph,
            fusion_strategy=fusion_strategy
        )
    
    def get_trainable_adapter_params(self):
        """获取Internal Adapter的可训练参数"""
        if self.internal_adapter_injector is not None:
            return self.internal_adapter_injector.get_trainable_parameters()
        return []
    
    def lock_backbone(self):
        """冻结骨干网络（但不冻结Internal Adapters"""
        for p in self.backbone.parameters():
            p.requires_grad = False
        
        # 如果有Internal Adapters，确保它们是可训练的
        if self.internal_adapter_injector is not None:
            for p in self.get_trainable_adapter_params():
                p.requires_grad = True
            print("骨干网络已冻结，但Internal Adapters保持可训练")
    
    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16
        features = self.backbone.get_intermediate_layers(
            x, n = self.intermediate_layer_idx[self.encoder_size]
        )
        out, anomaly_maps, anomaly_map_fused = self.head(features, patch_h, patch_w)
        out = F.interpolate(out, (patch_h * 16, patch_w * 16), mode='bilinear', align_corners=True)
        
        # 如果有融合的anomaly_map，也上采样到输入尺寸
        if anomaly_map_fused is not None:
            anomaly_map_fused = F.interpolate(
                anomaly_map_fused, 
                (patch_h * 16, patch_w * 16), 
                mode='bilinear', 
                align_corners=True
            )
        
        # 返回主输出、各层异常图、融合后的异常图
        return out, anomaly_maps, anomaly_map_fused
