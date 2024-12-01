import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio  # Sử dụng imageio.v2 để giữ hành vi cũ

def read_image(file_path):
    return imageio.imread(file_path)


def calculate_histogram(image):
    hist = [0] * 256
    for row in image:
        for pixel in row:
            hist[pixel] += 1
    return hist

def calculate_cdf(hist):
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
            # Chuyển pixel sang kiểu float để tránh tràn số
            pixel = float(pixel)
            min_val = float(min_val)
            max_val = float(max_val)
            scaled_value = ((pixel - min_val) * (b - a) / (max_val - min_val) + a)
            # Đảm bảo giá trị nằm trong khoảng [0, 255]
            scaled_value = max(0, min(255, int(scaled_value)))
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
    plt.figure(figsize=(16, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Ảnh gốc')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(equalized_image, cmap='gray')
    plt.title('Ảnh sau cân bằng histogram (H2)')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(narrowed_image, cmap='gray')
    plt.title('Ảnh sau thu hẹp cường độ (50, 100)')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.bar(range(256), hist, color='black')
    plt.title('Histogram ban đầu (H1)')
    plt.xlabel('Pixel intensity')
    plt.ylabel('Frequency')

    plt.subplot(2, 3, 5)
    plt.bar(range(256), calculate_histogram(equalized_image), color='black')
    plt.title('Histogram cân bằng (H2)')
    plt.xlabel('Pixel intensity')
    plt.ylabel('Frequency')

    plt.subplot(2, 3, 6)
    plt.bar(range(256), calculate_histogram(narrowed_image), color='black')
    plt.title('Histogram thu hẹp (50, 100)')
    plt.xlabel('Pixel intensity')
    plt.ylabel('Frequency')

    # Hiển thị kết quả
    plt.tight_layout()
    plt.show()


Bai1('picture.jpg')
