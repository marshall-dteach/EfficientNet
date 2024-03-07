import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups=64):
        super(SpatialGroupEnhance, self).__init__()
        self.groups = groups

    def forward(self, x):
        # x: input features with shape [N, C, H, W]
        N, C, H, W = x.size()
        
        # Step 1: Split channels into groups
        x = x.view(N, self.groups, C // self.groups, H, W)
        
        # Step 2: Spatial Global Pooling
        x_global = torch.mean(x, dim=(3, 4), keepdim=True)
        
        # Step 3: Calculate channel-wise statistics
        x_std = x.std(dim=(3, 4), keepdim=True) + 1e-5
        x_mean = torch.mean(x, dim=(3, 4), keepdim=True)
        
        # Step 4: Scale and normalize
        x_scaled = (x - x_mean) / x_std
        
        # Step 5: Squeeze and Excitation
        x_squeeze = torch.mean(x_scaled, dim=2, keepdim=True)
        x_excite = torch.sigmoid(x_squeeze)
        
        # Step 6: Scaling input features
        x = x * x_excite
        
        # Step 7: Merge groups
        x = x.view(N, C, H, W)
        
        return x

# Example of using SGE module
if __name__ == "__main__":
    # Assume input feature maps have the shape [N, C, H, W] = [2, 512, 14, 14]
    input_features = torch.randn(2, 64, 64, 64)
    sge = SpatialGroupEnhance(groups=64)
    output_features = sge(input_features)
    print("Output shape:", output_features.shape)
