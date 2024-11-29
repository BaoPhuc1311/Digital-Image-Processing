import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio  # Để đọc và lưu ảnh

def read_image(file_path):
    """Đọc ảnh từ file dưới dạng ma trận số nguyên 8-bit."""
    image = imageio.imread(file_path)
    if image.max() <= 1.0:  # Ảnh chuẩn hóa [0, 1]
        image = (image * 255).astype(np.uint8)
    return image

def save_image(image, file_path):
    """Lưu ma trận ảnh thành file."""
    imageio.imwrite(file_path, image.astype(np.uint8))

def apply_convolution(image, kernel, padding=0, stride=1):
    """Hàm áp dụng tích chập với kernel, padding, và stride."""
    h, w = image.shape
    kernel_size = kernel.shape[0]
    
    # Thêm padding vào ảnh
    padded_image = np.pad(image, pad_width=padding, mode='constant', constant_values=0)
    new_h = (padded_image.shape[0] - kernel_size) // stride + 1
    new_w = (padded_image.shape[1] - kernel_size) // stride + 1
    output = np.zeros((new_h, new_w), dtype=np.float32)

    # Tích chập
    for y in range(new_h):
        for x in range(new_w):
            region = padded_image[y * stride:y * stride + kernel_size, x * stride:x * stride + kernel_size]
            output[y, x] = np.sum(region * kernel)

    return np.clip(output, 0, 255).astype(np.uint8)

def apply_median_filter(image, kernel_size=3):
    """Hàm lọc trung vị."""
    h, w = image.shape
    pad = kernel_size // 2
    padded_image = np.pad(image, pad, mode='constant', constant_values=0)
    output = np.zeros_like(image)

    for y in range(h):
        for x in range(w):
            region = padded_image[y:y + kernel_size, x:x + kernel_size]
            output[y, x] = np.median(region)
    
    return output

def Bai2(file_path):
    # Đọc ảnh và chuyển sang ảnh xám
    image = read_image(file_path)
    gray_image = np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    # Kernel 3x3, padding = 1 (I1)
    kernel_3x3 = np.ones((3, 3), dtype=np.float32) / 9
    I1 = apply_convolution(gray_image, kernel_3x3, padding=1)

    # Kernel 5x5, padding = 2 (I2)
    kernel_5x5 = np.ones((5, 5), dtype=np.float32) / 25
    I2 = apply_convolution(gray_image, kernel_5x5, padding=2)

    # Kernel 7x7, padding = 3, stride = 2 (I3)
    kernel_7x7 = np.ones((7, 7), dtype=np.float32) / 49
    I3 = apply_convolution(gray_image, kernel_7x7, padding=3, stride=2)

    # Lọc trung vị ảnh I3 với lân cận 3x3 (I4)
    I4 = apply_median_filter(I3, kernel_size=3)

    # Hiển thị kết quả
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 3, 1)
    plt.title("Ảnh gốc")
    plt.imshow(gray_image, cmap='gray')

    plt.subplot(2, 3, 2)
    plt.title("I1 - Kernel 3x3")
    plt.imshow(I1, cmap='gray')

    plt.subplot(2, 3, 3)
    plt.title("I2 - Kernel 5x5")
    plt.imshow(I2, cmap='gray')

    plt.subplot(2, 3, 4)
    plt.title("I3 - Kernel 7x7, stride 2")
    plt.imshow(I3, cmap='gray')

    plt.subplot(2, 3, 5)
    plt.title("I4 - Lọc trung vị 3x3")
    plt.imshow(I4, cmap='gray')

    plt.tight_layout()
    plt.show()

    # Lưu kết quả
    save_image(I1, 'I1_kernel_3x3.jpg')
    save_image(I2, 'I2_kernel_5x5.jpg')
    save_image(I3, 'I3_kernel_7x7_stride_2.jpg')
    save_image(I4, 'I4_median_filtered.jpg')
