# 计算各模块参数量
hidden_dim = 384
bottleneck_dim = 64
num_adapters = 3

# Internal Adapter: down_proj + up_proj + LayerNorm (无bias的Linear)
adapter_params = (hidden_dim * bottleneck_dim) * 2 + (hidden_dim * 2)
total_internal_adapter = adapter_params * num_adapters
print(f'Internal Adapter: {total_internal_adapter:,}')

# GLCM: 4�? 每层2个ClipAdapter (cls + patch), 每个有fc1(C->256)+fc2(256->C)
bottleneck = 256
out_channels = [96, 192, 384, 768]
GLCM_params = sum(c * bottleneck * 4 for c in out_channels)  # 4 = 2 adapters * 2 layers each
print(f'GLCM (GLCM): {GLCM_params:,}')

# Hypergraph: 4�?
hidden_channels = 256
hypergraph_params = 0
for c in out_channels:
    # 2层HypergraphConv, 每层有theta+phi (Linear有bias)
    conv1 = (c * hidden_channels + hidden_channels) + (hidden_channels * hidden_channels + hidden_channels)
    conv2 = (hidden_channels * hidden_channels + hidden_channels) * 2
    out_proj = hidden_channels * c + c
    norm = c * 2
    hypergraph_params += conv1 + conv2 + out_proj + norm
print(f'Hypergraph: {hypergraph_params:,}')

# Decoder
in_ch = 384
features = 256
projects = sum(in_ch * c + c for c in out_channels)
scratch = sum(c * features + features for c in out_channels)
output = features * 4 * 1 + 1
decoder = projects + scratch + output
print(f'Decoder: {decoder:,}')

# 总计
total = total_internal_adapter + GLCM_params + hypergraph_params + decoder
backbone = 22_000_000
print(f'\nTotal trainable: {total:,}')
print(f'Backbone (frozen): {backbone:,}')
print(f'\nInternal Adapter ratio: {total_internal_adapter/backbone*100:.2f}%')
print(f'Total trainable ratio: {total/backbone*100:.2f}%')
