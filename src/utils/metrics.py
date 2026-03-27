import cv2
import numpy as np

def calculate_single_psnr(o_img: np.ndarray, r_img: np.ndarray) -> float:
    """Upscale ảnh resize về kích thước gốc và tính PSNR"""
    h, w = o_img.shape[:2]
    r_upscaled = cv2.resize(r_img, (w, h), interpolation=cv2.INTER_CUBIC)
    psnr_val = cv2.PSNR(o_img, r_upscaled)
    return psnr_val

def calculate_single_ssim(o_img: np.ndarray, r_img: np.ndarray) -> float:
    """Tính SSIM giữa ảnh gốc và ảnh resize dựa trên Grayscale, sử dụng mean, variance, covariance"""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    h, w = o_img.shape[:2]
    r_upscaled = cv2.resize(r_img, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Đưa về dải màu xám đen để đo SSIM chính xác
    o_gray = cv2.cvtColor(o_img, cv2.COLOR_RGB2GRAY).astype(np.float64)
    r_gray = cv2.cvtColor(r_upscaled, cv2.COLOR_RGB2GRAY).astype(np.float64)
    
    mu1, mu2 = np.mean(o_gray), np.mean(r_gray)
    var1, var2 = np.var(o_gray), np.var(r_gray)
    cov12 = np.mean((o_gray - mu1) * (r_gray - mu2))
    
    ssim_val = ((2 * mu1 * mu2 + C1) * (2 * cov12 + C2)) / ((mu1**2 + mu2**2 + C1) * (var1 + var2 + C2))
    return ssim_val