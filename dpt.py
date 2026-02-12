import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import _make_scratch

class DPTHead(nn.Module):
    def __init__(
        self, 
        nclass,
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512],  # 修改为2层
        num_layers=2,  # 添加层数参数
    ):
        super(DPTHead, self).__init__()
        self.num_layers = num_layers
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        self.scratch.stem_transpose = None
        # 修改输出卷积通道数，适配2层融合
        self.scratch.output_conv = nn.Conv2d(features*num_layers, nclass, kernel_size=1, stride=1, padding=0)  
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            out.append(x)
        
        # 根据层数动态处理
        if self.num_layers == 2:
            # 只使用2层特征
            layer_1, layer_2 = out
            layer_1_rn = self.scratch.layer1_rn(layer_1)
            layer_2_rn = self.scratch.layer2_rn(layer_2)
            target_hw = layer_1_rn.shape[-2:]
            layer_2_up = F.interpolate(layer_2_rn, size=target_hw, mode="bilinear", align_corners=True)
            fused = torch.cat([layer_1_rn, layer_2_up], dim=1)
        else:
            # 使用4层特征（原始逻辑）
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
        
        out = self.scratch.output_conv(fused)
        return out

class DPT(nn.Module):
    def __init__(
        self, 
        encoder_size='base', 
        nclass=2,
        features=128, 
        out_channels=[96, 192, 384, 768], 
        use_bn=False,
        backbone = None,
        use_layers='all'  # 'all' 或 '6_9' 选择使用的层，默认使用全部4层
    ):
        super(DPT, self).__init__()
        
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
        self.head = DPTHead(nclass, self.backbone.embed_dim, features, use_bn, 
                           out_channels=out_channels, num_layers=self.num_layers)
        
    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
    
    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16
        features = self.backbone.get_intermediate_layers(
            x, n = self.intermediate_layer_idx[self.encoder_size]
        )
        out = self.head(features, patch_h, patch_w)
        out = F.interpolate(out, (patch_h * 16, patch_w * 16), mode='bilinear', align_corners=True)
        return out
