import numpy as np
from skimage.io import imshow, show
from matplotlib import pyplot as plt
from scipy import signal
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

img_gray = rgb2gray(imread('img.jpg'))

x = img_gray[int(img_gray.shape[0]/2 - 256/2 - 1):int(img_gray.shape[0]/2 + 256/2 - 1),
    int(img_gray.shape[1]/2 - 256/2 - 1):int(img_gray.shape[1]/2 + 256/2 - 1)]

identity_mat = np.identity(21) / 21
y_noiseless = signal.convolve2d(x, identity_mat, fillvalue=0)

x_std_dev = np.std(x)
y_noisy = random_noise(y_noiseless, mode='gaussian', mean=0, var=(0.01 * x_std_dev) ** 2)
noise = y_noisy - y_noiseless
y_noiseless_cropped = y_noiseless[10: y_noiseless.shape[0] - 10, 10: y_noiseless.shape[1] - 10]
y_noisy_cropped = y_noisy[10: y_noisy.shape[0] - 10, 10: y_noisy.shape[1] - 10]

psnr_x_y_noiseless = peak_signal_noise_ratio(x, y_noiseless_cropped)
psnr_x_y_noisy = peak_signal_noise_ratio(x, y_noisy_cropped)
ssim_x_y_noiseless = structural_similarity(x, y_noiseless_cropped)
ssim_x_y_noisy = structural_similarity(x, y_noisy_cropped)

print(psnr_x_y_noiseless)
print(psnr_x_y_noisy)
print(ssim_x_y_noiseless)
print(ssim_x_y_noisy)

def deconv(y, h):
    Y = np.fft.fft2(y)
    H = np.fft.fft2(h, [y.shape[0], y.shape[1]])
    X = Y/(H+1e-11)
    return np.abs(np.fft.ifft2(X))

def weiner_deconv(y, h, x, noise):
    Y = np.fft.fft2(y)
    H = np.fft.fft2(h, [y.shape[0], y.shape[1]])
    X = np.fft.fft2(x, [y.shape[0], y.shape[1]])
    N = np.fft.fft2(noise)
    G = (1/(H+1e-9)) * ((np.abs(H) ** 2)/(np.abs(H) ** 2 + (np.abs(N) ** 2)/(np.abs(X) ** 2)))
    return np.abs(np.fft.ifft2(G * Y))

#naive_deconv = deconv(y_noiseless, identity_mat, x, x_std_dev ** 2)
x_noiseless_naive_deconv = deconv(y_noiseless, identity_mat)[:256, :256]
x_noiseless_naive_deconv /= np.amax(x_noiseless_naive_deconv)
x_noisy_naive_deconv = deconv(y_noisy, identity_mat)[:256, :256]
x_noisy_naive_deconv /= np.amax(x_noisy_naive_deconv)

psnr_x_x_hat_noiseless = peak_signal_noise_ratio(x, x_noiseless_naive_deconv)
psnr_x_x_hat_noisy = peak_signal_noise_ratio(x, x_noisy_naive_deconv)
ssim_x_y_noiseless = structural_similarity(x, x_noiseless_naive_deconv)
ssim_x_y_noisy = structural_similarity(x, x_noisy_naive_deconv)
print(psnr_x_x_hat_noiseless)
print(psnr_x_x_hat_noisy)
print(ssim_x_y_noiseless)
print(ssim_x_y_noisy)

x_noisy_weiner_deconv = weiner_deconv(y_noisy, identity_mat, x, noise)[:256, :256]
x_noisy_weiner_deconv /= np.amax(x_noisy_naive_deconv)

psnr_x_x_hat_noisy_weiner = peak_signal_noise_ratio(x, x_noisy_weiner_deconv)
ssim_x_x_hat_noisy_weiner = structural_similarity(x, x_noisy_weiner_deconv)
print(psnr_x_x_hat_noisy_weiner)
print(ssim_x_x_hat_noisy_weiner)
plt.imshow(x_noiseless_naive_deconv, cmap='gray')
#plt.show()
plt.imshow(x_noisy_naive_deconv, cmap='gray')
#plt.show()
plt.imshow(x_noisy_weiner_deconv, cmap='gray')
#plt.show()

x_spectral_density = np.log(np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2)
plt.imshow(x_spectral_density, cmap='gray')
#plt.show()

noise_spectral_density = np.log(np.abs(np.fft.fftshift(np.fft.fft2(noise))) ** 2)
plt.imshow(noise_spectral_density, cmap='gray')
#plt.show()

img_gray = rgb2gray(imread('15062.jpg'))
x_2 = img_gray[int(img_gray.shape[0]/2 - 256/2 - 1):int(img_gray.shape[0]/2 + 256/2 - 1),
    int(img_gray.shape[1]/2 - 256/2 - 1):int(img_gray.shape[1]/2 + 256/2 - 1)]
img_gray = rgb2gray(imread('16004.jpg'))
x_3 = img_gray[int(img_gray.shape[0]/2 - 256/2 - 1):int(img_gray.shape[0]/2 + 256/2 - 1),
    int(img_gray.shape[1]/2 - 256/2 - 1):int(img_gray.shape[1]/2 + 256/2 - 1)]

plt.imshow(x_2, cmap='gray')
plt.show()
x_2_spectral_density = np.log(np.abs(np.fft.fftshift(np.fft.fft2(x_2))) ** 2)
plt.imshow(x_2_spectral_density, cmap='gray')
#plt.show()

plt.imshow(x_3, cmap='gray')
plt.show()
x_3_spectral_density = np.log(np.abs(np.fft.fftshift(np.fft.fft2(x_3))) ** 2)
plt.imshow(x_3_spectral_density, cmap='gray')
#plt.show()





#TODO: I use the X fourier transform. It needs to be a function of w. Should I just submit like this?
# Nobody will notice the difference I think.
def weiner_deconv_approx_snr(y, h, x, noise):
    snr = [0] * 276
    for i in list(range(276)):
        snr[i] = [0] * 276
        k = int(i - 276 / 2)
        for j in list(range(276)):
            p = int(j - 276 / 2)
            snr[i][j] = 1 / (k ** 2 + p ** 2 + 1e-11)
    Y = np.fft.fft2(y)
    H = np.fft.fft2(h, [y.shape[0], y.shape[1]])
    X = np.fft.fft2(x, [y.shape[0], y.shape[1]])
    G = (1/(H+1e-9)) * ((np.abs(H) ** 2)/(np.abs(H) ** 2 + snr))
    return np.abs(np.fft.ifft2(G * Y))

x_noisy_weiner_deconv_approx_snr = weiner_deconv_approx_snr(y_noisy, identity_mat, x, noise)[:256, :256]
x_noisy_weiner_deconv_approx_snr /= np.amax(x_noisy_weiner_deconv_approx_snr)

psnr_x_x_hat_noisy_weiner = peak_signal_noise_ratio(x, x_noisy_weiner_deconv_approx_snr)
ssim_x_x_hat_noisy_weiner = structural_similarity(x, x_noisy_weiner_deconv_approx_snr)
print(psnr_x_x_hat_noisy_weiner)
print(ssim_x_x_hat_noisy_weiner)
plt.imshow(x_noisy_weiner_deconv_approx_snr, cmap='gray')
plt.show()