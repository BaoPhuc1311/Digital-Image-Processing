import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio  # Sử dụng imageio.v2 để giữ hành vi cũ

def read_image(file_path):
    """Đọc ảnh từ file dưới dạng ma trận số nguyên 8-bit."""
    return imageio.imread(file_path)


def calculate_histogram(image):
    """Tính histogram của ảnh (số lượng pixel ở mỗi mức cường độ)."""
    hist = [0] * 256
    for row in image:
        for pixel in row:
            hist[pixel] += 1
    return hist

def calculate_cdf(hist):
    """Tính hàm phân bố tích lũy (CDF) từ histogram."""
    cdf = [0] * len(hist)
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i] = cdf[i - 1] + hist[i]
    return cdf

def normalize_cdf(cdf, total_pixels):
    cdf_min = min(cdf)
    cdf_normalized = [(c - cdf_min) * 255 // (total_pixels - cdf_min) for c in cdf]
    return cdf_normalized

def apply_histogram_equalization(image, cdf_normalized):
    equalized_image = np.zeros_like(image, dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            equalized_image[i, j] = cdf_normalized[image[i, j]]
    return equalized_image

def scale_intensity(image, a, b):
    # Tìm giá trị pixel nhỏ nhất và lớn nhất
    min_val = min(min(row) for row in image)
    max_val = max(max(row) for row in image)
    
    if max_val == min_val:
        # Tránh chia cho 0, trả về hình ảnh có cùng cường độ trong khoảng [a, b]
        return [[a for _ in row] for row in image]

    # Thu hẹp giá trị pixel vào khoảng [a, b]
    scaled_image = []
    for row in image:
        scaled_row = []
        for pixel in row:
            scaled_value = ((pixel - min_val) * (b - a) / (max_val - min_val) + a)
            scaled_value = max(0, min(255, int(scaled_value)))  # Giới hạn trong khoảng [0, 255]
            scaled_row.append(scaled_value)
        scaled_image.append(scaled_row)
    
    return scaled_image


def Bai1(file_path):
    # Đọc ảnh màu và chuyển sang ảnh xám
    image = read_image(file_path)
    gray_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            gray_image[i, j] = int(0.299 * image[i, j, 0] +
                                   0.587 * image[i, j, 1] +
                                   0.114 * image[i, j, 2])

    # Tính histogram ban đầu
    hist = calculate_histogram(gray_image)

    # Tính CDF từ histogram và chuẩn hóa
    cdf = calculate_cdf(hist)
    cdf_normalized = normalize_cdf(cdf, gray_image.size)

    # Ánh xạ cường độ pixel theo CDF
    equalized_image = apply_histogram_equalization(gray_image, cdf_normalized)

    # Thu hẹp cường độ vào khoảng [50, 100]
    narrowed_image = scale_intensity(equalized_image, 50, 100)

    # Vẽ các histogram
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.bar(range(256), hist, color='black')
    plt.title('Histogram ban đầu (H1)')
    plt.xlabel('Pixel intensity')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 2)
    plt.bar(range(256), calculate_histogram(equalized_image), color='black')
    plt.title('Histogram cân bằng (H2)')
    plt.xlabel('Pixel intensity')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 3)
    plt.bar(range(256), calculate_histogram(narrowed_image), color='black')
    plt.title('Histogram thu hẹp (50,100)')
    plt.xlabel('Pixel intensity')
    plt.ylabel('Frequency')

    # Hiển thị kết quả
    plt.tight_layout()
    plt.show()


