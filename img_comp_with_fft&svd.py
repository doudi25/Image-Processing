import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
import os
# The class below is used to process images using the Fast Fourier Transform and Singular Value Decomposition
# you put your image's path and then you can use the FFT or SVD method to compress the image 
class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = read_image(image_path).float() / 255.0
        if self.image.shape[0] == 3: # if the image is RGB, convert to grayscale
            self.image = torch.mean(self.image, dim=0, keepdim=True)
    def add_noise(self) :
        noise = torch.randn_like(self.image) * 0.08
        self.image += noise
        self.image = self.image.clamp(0, 1)
        plt.imshow(self.image.squeeze().numpy(), cmap="gray")
        plt.title("Noisy Image")
        plt.show()
    def FFT(self):
        fft = torch.fft.fft2(self.image.squeeze())
        fft_shifted = torch.fft.fftshift(fft)
        magnitude = torch.abs(fft_shifted) # must use real values
        threshold = torch.quantile(magnitude, 0.92)
        mask = magnitude > threshold
        filtered_fft = fft_shifted * mask
        fft_inverse_shifted = torch.fft.ifftshift(filtered_fft)
        filtered_image = torch.fft.ifft2(fft_inverse_shifted).real
        self.filtered_image = filtered_image.clamp(0, 1)
        plt.imshow(self.filtered_image.squeeze().numpy(), cmap="gray")
        plt.title("Filtered Image")
        plt.show()
        self.save_filtered_image()
        print(f'Original Image Size: {os.path.getsize(self.image_path)} bytes')
        print(f'Filtered Image Size: {os.path.getsize("filtered_image.jpg")} bytes')
    def SVD(self):
        U , sigma, V = torch.svd(self.image, some=True)
        sigma_diag = torch.zeros(sigma.shape[1],sigma.shape[1])
        sigma_diag.diagonal(offset=0)[:] = sigma.squeeze()
        # Print shapes
        print("U shape:", U.shape)
        print("sigma shape:", sigma_diag.shape)
        print("V shape:", V.shape)
        V = V.squeeze()
        U = U.squeeze()
        self.compressed_image = U[:,:90] @ sigma_diag[:90,:90] @ V[:,:90].T
        plt.imshow(self.compressed_image.squeeze().numpy(), cmap='gray')
        plt.show()
        self.save_compressed_image()
        print(f'Original Image Size: {os.path.getsize(self.image_path)} bytes')
        print(f'Compressed Image Size: {os.path.getsize("compressed_image.jpg")} bytes')
    def save_filtered_image(self, path='filtered_image.jpg',quality=55):
        if self.filtered_image is not None:
            save_image = to_pil_image(self.filtered_image).save(path,format='jpeg',quality=quality)
            print(f'Image saved at {path}')
    def save_compressed_image(self, path='compressed_image.jpg',quality=55):
        if self.compressed_image is not None:
            save_image = to_pil_image(self.compressed_image).save(path,format='jpeg',quality=quality)
            print(f'Image saved at {path}')

processor = ImageProcessor(r'C:\Users\mouhd\OneDrive\Images\eiffel.jpg') # replace the path with your path
processor.SVD()
"""
if you want to use the FFT method , you call the noise first then the FFT method
processor.add_noise()
processor.FFT() 
"""