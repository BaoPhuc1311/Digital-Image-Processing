import cv2
import numpy as np
import matplotlib.pyplot as plt

def Bai2(file_path):
    # Đọc ảnh và chuyển sang ảnh xám
    image = cv2.imread(file_path)  # Thay bằng đường dẫn ảnh của bạn
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Hàm áp dụng tích chập với kernel, padding, và stride
    def apply_convolution(image, kernel, padding=0, stride=1):
        padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
        kernel_size = kernel.shape[0]
        output_shape = (
            (padded_image.shape[0] - kernel_size) // stride + 1,
            (padded_image.shape[1] - kernel_size) // stride + 1,
        )
        output = np.zeros(output_shape, dtype=np.float32)
        
        for y in range(0, output.shape[0], stride):
            for x in range(0, output.shape[1], stride):
                region = padded_image[y * stride:y * stride + kernel_size, x * stride:x * stride + kernel_size]
                output[y, x] = np.sum(region * kernel)
        
        return np.clip(output, 0, 255).astype(np.uint8)

    # Kernel 3x3, padding = 1 (I1)
    kernel_3x3 = np.ones((3, 3), dtype=np.float32) / 9  # Kernel trung bình
    I1 = apply_convolution(gray_image, kernel_3x3, padding=1)

    # Kernel 5x5, padding = 2 (I2)
    kernel_5x5 = np.ones((5, 5), dtype=np.float32) / 25  # Kernel trung bình
    I2 = apply_convolution(gray_image, kernel_5x5, padding=2)

    # Kernel 7x7, padding = 3, stride = 2 (I3)
    kernel_7x7 = np.ones((7, 7), dtype=np.float32) / 49  # Kernel trung bình
    I3 = apply_convolution(gray_image, kernel_7x7, padding=3, stride=2)

    # Lọc trung vị ảnh I3 với lân cận 3x3 (I4)
    I4 = cv2.medianBlur(I3, 3)

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

    # Lưu kết quả nếu cần
    cv2.imwrite('I1_kernel_3x3.jpg', I1)
    cv2.imwrite('I2_kernel_5x5.jpg', I2)
    cv2.imwrite('I3_kernel_7x7_stride_2.jpg', I3)
    cv2.imwrite('I4_median_filtered.jpg', I4)
