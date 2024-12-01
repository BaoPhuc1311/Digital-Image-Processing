import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np  # Để đọc và lưu ảnh

def read_image(file_path):
    """Đọc ảnh từ file dưới dạng ma trận số nguyên 8-bit."""
    image = imageio.imread(file_path)
    if image.max() <= 1.0:  # Ảnh chuẩn hóa [0, 1]
        image = (image * 255).astype(int)
    return image

def save_image(image, file_path):
    """Lưu ma trận ảnh thành file."""
    # Chuyển đổi kiểu dữ liệu thành uint8 và đảm bảo giá trị trong phạm vi [0, 255]
    image = np.clip(image, 0, 255).astype(np.uint8)
    imageio.imwrite(file_path, image)

def apply_convolution(image, kernel, padding=0, stride=1):
    """Hàm áp dụng tích chập với kernel, padding, và stride."""
    h = len(image)
    w = len(image[0])
    kernel_size = len(kernel)

    # Thêm padding vào ảnh
    padded_image = [[0] * (w + 2 * padding) for _ in range(h + 2 * padding)]
    for i in range(h):
        for j in range(w):
            padded_image[i + padding][j + padding] = image[i][j]

    new_h = (h + 2 * padding - kernel_size) // stride + 1
    new_w = (w + 2 * padding - kernel_size) // stride + 1
    output = [[0] * new_w for _ in range(new_h)]

    # Tích chập
    for y in range(new_h):
        for x in range(new_w):
            region = [padded_image[y * stride + i][x * stride + j] for i in range(kernel_size) for j in range(kernel_size)]
            output[y][x] = sum(region[k] * kernel[k // kernel_size][k % kernel_size] for k in range(len(region)))

    # Giới hạn giá trị ảnh trong phạm vi [0, 255]
    return [[min(max(int(val), 0), 255) for val in row] for row in output]

def apply_median_filter(image, kernel_size=3):
    """Hàm lọc trung vị."""
    h = len(image)
    w = len(image[0])
    pad = kernel_size // 2
    padded_image = [[0] * (w + 2 * pad) for _ in range(h + 2 * pad)]
    for i in range(h):
        for j in range(w):
            padded_image[i + pad][j + pad] = image[i][j]

    output = [[0] * w for _ in range(h)]

    for y in range(h):
        for x in range(w):
            region = [padded_image[y + i][x + j] for i in range(kernel_size) for j in range(kernel_size)]
            output[y][x] = sorted(region)[len(region) // 2]

    return output

def Bai2(file_path):
    # Đọc ảnh và chuyển sang ảnh xám
    image = read_image(file_path)
    gray_image = [
        [int(image[i][j][0] * 0.299 + image[i][j][1] * 0.587 + image[i][j][2] * 0.114) for j in range(len(image[i]))]
        for i in range(len(image))
    ]

    # Kernel 3x3, padding = 1 (I1)
    kernel_3x3 = [[1 / 9] * 3 for _ in range(3)]
    I1 = apply_convolution(gray_image, kernel_3x3, padding=1)

    # Kernel 5x5, padding = 2 (I2)
    kernel_5x5 = [[1 / 25] * 5 for _ in range(5)]
    I2 = apply_convolution(gray_image, kernel_5x5, padding=2)

    # Kernel 7x7, padding = 3, stride = 2 (I3)
    kernel_7x7 = [[1 / 49] * 7 for _ in range(7)]
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


Bai2('picture.jpg')
