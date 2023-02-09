import einops
import numpy as np

a = np.random.random((10, 28, 28))
b, h, w = a.shape
patch_h = 4
patch_w = 4
step_h = 2
step_w = 2

idx_h_center = np.arange(step_h, h - step_h, step_h)
idx_w_center = np.arange(step_w, h - step_w, step_w)

idx_h = einops.rearrange(idx_h_center, 'h -> h 1 1 1')
idx_h = idx_h + einops.rearrange(np.arange(patch_h), 'h -> h 1') - patch_h // 2
idx_w = einops.rearrange(idx_w_center, 'w -> 1 w 1 1')
idx_w = idx_w + einops.rearrange(np.arange(patch_w), 'w -> 1 w') - patch_w // 2

b = a[:, idx_h, idx_w]
print(b.shape)

