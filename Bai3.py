import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import imageio.v2 as imageio

def read_image(file_path):
    """Đọc ảnh từ file dưới dạng ma trận số nguyên 8-bit."""
    image = imageio.imread(file_path)
    if image.max() <= 1.0:  # Nếu ảnh được chuẩn hóa trong khoảng [0, 1]
        image = (image * 255).astype(np.uint8)
    return image

def convert_to_grayscale(image):
    """Chuyển đổi ảnh màu sang ảnh xám."""
    return np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

def equalize_histogram(hist):
    """Cân bằng histogram."""
    cdf = hist.cumsum()  # Tính CDF
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    return np.clip(cdf_normalized, 0, 255).astype(np.uint8)

def pad_image(image, radius, padding=0):
    """Thêm padding cho ảnh."""
    if padding == 1:
        return np.pad(image, pad_width=radius, mode='constant', constant_values=0)
    return image

def compute_lbp_and_histogram(image, radius, padding):
    """Tính LBP và histogram."""
    # Số điểm lân cận (P)
    P = 8 * radius

    # Thêm padding nếu cần
    padded_image = pad_image(image, radius, padding)

    # Tính LBP
    lbp_image = local_binary_pattern(padded_image, P, radius, method='default')

    # Tính histogram
    hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist_normalized = hist / hist.sum()  # Chuẩn hóa histogram

    return lbp_image, hist, hist_normalized

def save_image(figure, file_name):
    """Lưu hình ảnh từ plt.figure."""
    figure.savefig(file_name, dpi=300)

def Bai3(file_path):
    # Đọc ảnh và chuyển sang ảnh xám
    image = read_image(file_path)
    gray_image = convert_to_grayscale(image)

    # Thiết lập thông số
    radii = [1, 2, 3]
    paddings = [0, 1]

    # LBP và histogram cho từng trường hợp
    results = {}
    for radius in radii:
        for padding in paddings:
            lbp_image, hist, hist_normalized = compute_lbp_and_histogram(gray_image, radius, padding)
            results[(radius, padding)] = (lbp_image, hist, hist_normalized)

    # Hiển thị LBP và histogram
    lbp_figure = plt.figure(figsize=(16, 12))
    for i, ((radius, padding), (lbp_image, hist, hist_normalized)) in enumerate(results.items()):
        plt.subplot(len(radii), len(paddings) * 2, i * 2 + 1)
        plt.title(f"LBP R={radius}, Padding={padding}")
        plt.imshow(lbp_image, cmap='gray')
        plt.axis('off')

        plt.subplot(len(radii), len(paddings) * 2, i * 2 + 2)
        plt.title(f"Histogram R={radius}, Padding={padding}")
        plt.bar(range(len(hist)), hist_normalized, color='black')
        plt.xlabel("LBP Patterns")
        plt.ylabel("Frequency")
    save_image(lbp_figure, "lbp_histograms.png")

    # Cân bằng histogram
    eq_hist_figure = plt.figure(figsize=(16, 12))
    for i, ((radius, padding), (lbp_image, hist, hist_normalized)) in enumerate(results.items()):
        equalized_hist = equalize_histogram((hist_normalized * 255).astype(np.uint8))

        plt.subplot(len(radii), len(paddings), i + 1)
        plt.title(f"Cân bằng Histogram R={radius}, Padding={padding}")
        plt.plot(equalized_hist, color='black')
        plt.xlabel("LBP Patterns")
        plt.ylabel("Frequency")
    save_image(eq_hist_figure, "equalized_histograms.png")

    plt.tight_layout()
    plt.show()
