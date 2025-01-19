import torch
from torchvision.io.image import read_image
import matplotlib.pyplot as plt
import numpy as np
import os 
from torchvision.transforms.functional import to_pil_image
a = read_image('C:/Users/mouhd/OneDrive/Images/eiffel.jpg')
a = a.float() / 255.0
a = torch.mean(a, dim=0, keepdim=True)
U, sigma, V = torch.svd(a, some=True)
sigma_diag = torch.zeros(sigma.shape[1],sigma.shape[1])
sigma_diag.diagonal(offset=0)[:] = sigma.squeeze()
# Print shapes
print("U shape:", U.shape)
print("sigma shape:", sigma_diag.shape)
print("V shape:", V.shape)
V = V.squeeze()
U = U.squeeze()
compressed = U[:,:90] @ sigma_diag[:90,:90] @ V[:,:90].T
image = to_pil_image(compressed).save('compressed.jpg',format='jpeg',quality=55)
print(f'Original Image Size: {os.path.getsize("C:/Users/mouhd/OneDrive/Images/eiffel.jpg")} bytes')
print(f'Compressed Image Size: {os.path.getsize("compressed.jpg")} bytes')
plt.imshow(compressed.squeeze().numpy(), cmap='gray')
plt.show()