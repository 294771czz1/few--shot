"""
Internal Adapter Module for DINOv3
在冻结的DINOv3 Transformer内部注入轻量级适配器，使其适应特定数据集
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerAdapter(nn.Module):
    """
    轻量级Adapter，插入到Transformer Block后
    使用bottleneck结构减少参数量
    """
    def __init__(self, hidden_dim=384, bottleneck_dim=64, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        
        # Down projection
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        
        # Activation
        self.activation = nn.GELU()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Up projection
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim, bias=False)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 初始化为接近恒等映射，避免初期干扰预训练特征
        nn.init.zeros_(self.up_proj.weight)
        nn.init.normal_(self.down_proj.weight, std=0.01)
    
    def forward(self, x):
        """
        Args:
            x: [B, N, hidden_dim] Transformer输出
        Returns:
            [B, N, hidden_dim] 适配后的特征
        """
        residual = x
        
        # Adapter pathway
        x = self.down_proj(x)          # [B, N, bottleneck_dim]
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)            # [B, N, hidden_dim]
        
        # Residual connection
        x = residual + x
        
        # Layer normalization
        x = self.layer_norm(x)
        
        return x


class InternalAdapterInjector(nn.Module):
    """
    将Adapter注入到DINOv3的指定层
    使用forward hook机制，不修改原模型结构
    继承nn.Module以便正确处理设备转移
    """
    def __init__(self, dino_model, target_layers=[2, 5, 8], hidden_dim=384, bottleneck_dim=64):
        """
        Args:
            dino_model: DINOv3模型
            target_layers: 要注入adapter的层索引 (0-based)
            hidden_dim: DINOv3的hidden dimension
            bottleneck_dim: Adapter的bottleneck dimension
        """
        super().__init__()
        self.dino_model = dino_model
        self.target_layers = target_layers
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        
        # 创建adapter模块
        self.adapters = nn.ModuleList([
            TransformerAdapter(hidden_dim, bottleneck_dim)
            for _ in target_layers
        ])
        
        # 存储hook handles
        self.hooks = []
        
        # 注入adapters
        self._inject_adapters()
    
    def _inject_adapters(self):
        """使用forward hook注入adapters"""
        for idx, layer_idx in enumerate(self.target_layers):
            adapter = self.adapters[idx]
            
            # 创建hook函数
            def make_hook(adapter_module):
                def hook(module, input, output):
                    # output是Transformer Block的输出: [B, N, hidden_dim]
                    # N = num_patches + 1 (包含CLS token)
                    return adapter_module(output)
                return hook
            
            # 注册hook到指定的Transformer Block
            handle = self.dino_model.blocks[layer_idx].register_forward_hook(
                make_hook(adapter)
            )
            self.hooks.append(handle)
        
        print(f"✅ Internal Adapters注入成功:")
        print(f"   - 目标层: {self.target_layers}")
        print(f"   - Hidden dim: {self.hidden_dim}")
        print(f"   - Bottleneck dim: {self.bottleneck_dim}")
        print(f"   - 每个Adapter参数量: {self._count_adapter_params():,}")
        print(f"   - 总Adapter参数量: {self._count_adapter_params() * len(self.adapters):,}")
    
    def remove_adapters(self):
        """移除所有adapter hooks"""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        print("✅ Internal Adapters已移除")
    
    def _count_adapter_params(self):
        """计算单个adapter的参数量"""
        if len(self.adapters) > 0:
            return sum(p.numel() for p in self.adapters[0].parameters())
        return 0
    
    def get_trainable_parameters(self):
        """获取所有adapter的可训练参数"""
        params = []
        for adapter in self.adapters:
            params.extend(list(adapter.parameters()))
        return params


class MultiScaleInternalAdapter(nn.Module):
    """
    多尺度内部Adapter
    在不同深度的层使用不同的bottleneck维度
    """
    def __init__(self, dino_model, layer_configs):
        """
        Args:
            dino_model: DINOv3模型
            layer_configs: list of dict, 每个dict包含:
                {
                    'layer_idx': int,
                    'hidden_dim': int,
                    'bottleneck_dim': int
                }
        """
        super().__init__()
        self.dino_model = dino_model
        self.layer_configs = layer_configs
        
        # 创建不同配置的adapters
        self.adapters = nn.ModuleList([
            TransformerAdapter(
                hidden_dim=config['hidden_dim'],
                bottleneck_dim=config['bottleneck_dim']
            )
            for config in layer_configs
        ])
        
        self.hooks = []
        self._inject_adapters()
    
    def _inject_adapters(self):
        """注入多尺度adapters"""
        for idx, config in enumerate(self.layer_configs):
            layer_idx = config['layer_idx']
            adapter = self.adapters[idx]
            
            def make_hook(adapter_module):
                def hook(module, input, output):
                    return adapter_module(output)
                return hook
            
            handle = self.dino_model.blocks[layer_idx].register_forward_hook(
                make_hook(adapter)
            )
            self.hooks.append(handle)
        
        print(f"✅ Multi-Scale Internal Adapters注入成功:")
        for config in self.layer_configs:
            print(f"   - Layer {config['layer_idx']}: "
                  f"hidden={config['hidden_dim']}, "
                  f"bottleneck={config['bottleneck_dim']}")
    
    def remove_adapters(self):
        """移除所有adapters"""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []


def create_internal_adapters(dino_model, adapter_type='uniform', **kwargs):
    """
    工厂函数：创建Internal Adapters
    
    Args:
        dino_model: DINOv3模型
        adapter_type: 'uniform' 或 'multi_scale'
        **kwargs: adapter配置参数
    
    Returns:
        InternalAdapterInjector 或 MultiScaleInternalAdapter
    """
    if adapter_type == 'uniform':
        target_layers = kwargs.get('target_layers', [2, 5, 8])
        hidden_dim = kwargs.get('hidden_dim', 384)
        bottleneck_dim = kwargs.get('bottleneck_dim', 64)
        
        return InternalAdapterInjector(
            dino_model=dino_model,
            target_layers=target_layers,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim
        )
    
    elif adapter_type == 'multi_scale':
        layer_configs = kwargs.get('layer_configs', [
            {'layer_idx': 2, 'hidden_dim': 384, 'bottleneck_dim': 48},
            {'layer_idx': 5, 'hidden_dim': 384, 'bottleneck_dim': 64},
            {'layer_idx': 8, 'hidden_dim': 384, 'bottleneck_dim': 80},
        ])
        
        return MultiScaleInternalAdapter(
            dino_model=dino_model,
            layer_configs=layer_configs
        )
    
    else:
        raise ValueError(f"Unknown adapter_type: {adapter_type}")


if __name__ == '__main__':
    # 测试代码
    print("测试 Internal Adapter Module")
    
    # 模拟DINOv3模型
    class MockDinoBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(384, 384)
        
        def forward(self, x):
            return self.linear(x)
    
    class MockDinoModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([MockDinoBlock() for _ in range(12)])
        
        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x
    
    # 创建模型
    dino = MockDinoModel()
    
    # 测试1: Uniform Adapters
    print("\n" + "="*60)
    print("测试1: Uniform Internal Adapters")
    print("="*60)
    injector = create_internal_adapters(
        dino_model=dino,
        adapter_type='uniform',
        target_layers=[2, 5, 8],
        hidden_dim=384,
        bottleneck_dim=64
    )
    
    # 测试前向传播
    x = torch.randn(2, 197, 384)  # [B, N, C]
    with torch.no_grad():
        out = dino(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    
    # 获取可训练参数
    trainable_params = injector.get_trainable_parameters()
    print(f"可训练参数数量: {sum(p.numel() for p in trainable_params):,}")
    
    # 移除adapters
    injector.remove_adapters()
    
    # 测试2: Multi-Scale Adapters
    print("\n" + "="*60)
    print("测试2: Multi-Scale Internal Adapters")
    print("="*60)
    multi_injector = create_internal_adapters(
        dino_model=dino,
        adapter_type='multi_scale',
        layer_configs=[
            {'layer_idx': 2, 'hidden_dim': 384, 'bottleneck_dim': 48},
            {'layer_idx': 5, 'hidden_dim': 384, 'bottleneck_dim': 64},
            {'layer_idx': 8, 'hidden_dim': 384, 'bottleneck_dim': 80},
        ]
    )
    
    with torch.no_grad():
        out = dino(x)
    print(f"输出形状: {out.shape}")
    
    print("\n✅ 所有测试通过!")
