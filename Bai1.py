import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio  # Sử dụng imageio.v2 để giữ hành vi cũ

def read_image(file_path):
    """Đọc ảnh từ file dưới dạng ma trận số nguyên 8-bit."""
    return imageio.imread(file_path)

def save_image(image, file_path):
    """Lưu ma trận ảnh thành file."""
    imageio.imwrite(file_path, image)

def Bai1(file_path):
    # Đọc ảnh màu và chuyển sang ảnh xám
    image = read_image(file_path)  # Đọc ảnh vào ma trận
    gray_image = np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

    # Vẽ histogram ban đầu (H1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.hist(gray_image.ravel(), bins=256, range=(0, 256), color='black')
    plt.title('Histogram ban đầu (H1)')
    plt.xlabel('Pixel intensity')
    plt.ylabel('Frequency')

    # Histogram cân bằng (H2)
    # Tính histogram và CDF
    hist, bins = np.histogram(gray_image.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()  # Hàm phân bố tích lũy
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())  # Chuẩn hóa CDF
    equalized_image = cdf_normalized[gray_image]  # Ánh xạ cường độ từ CDF

    plt.subplot(1, 3, 2)
    plt.hist(equalized_image.ravel(), bins=256, range=(0, 256), color='black')
    plt.title('Histogram cân bằng (H2)')
    plt.xlabel('Pixel intensity')
    plt.ylabel('Frequency')

    # Hiệu chỉnh thu hẹp H2 trong khoảng (50, 100)
    # Công thức chuyển đổi cường độ
    a, b = 50, 100
    narrowed_image = ((equalized_image - equalized_image.min()) * (b - a) /
                      (equalized_image.max() - equalized_image.min()) + a).astype(np.uint8)

    plt.subplot(1, 3, 3)
    plt.hist(narrowed_image.ravel(), bins=256, range=(0, 256), color='black')
    plt.title('Histogram thu hẹp (50,100)')
    plt.xlabel('Pixel intensity')
    plt.ylabel('Frequency')

    # Hiển thị kết quả
    plt.tight_layout()
    plt.show()

    # Lưu các ảnh nếu cần thiết
    save_image(gray_image, 'gray_image.jpg')
    save_image(equalized_image.astype(np.uint8), 'equalized_image.jpg')
    save_image(narrowed_image, 'narrowed_image.jpg')
