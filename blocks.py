import torch.nn as nn

def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()
    
    # 根据输入层数动态创建
    num_layers = len(in_shape)
    
    if num_layers >= 1:
        scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    if num_layers >= 2:
        scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    if num_layers >= 3:
        scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    if num_layers >= 4:
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    
    return scratch