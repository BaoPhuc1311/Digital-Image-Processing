import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

def Bai3(file_path):
    # Đọc ảnh và chuyển sang ảnh xám
    image = cv2.imread('image1.png')  # Thay bằng đường dẫn ảnh của bạn
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Hàm tính và hiển thị LBP
    def compute_lbp_and_histogram(image, radius, padding):
        if padding == 1:
            # Padding với giá trị 0
            padded_image = cv2.copyMakeBorder(image, radius, radius, radius, radius, cv2.BORDER_CONSTANT, value=0)
        else:
            padded_image = image

        # Số điểm lân cận (P)
        P = 8 * radius
        # Tính LBP
        lbp_image = local_binary_pattern(padded_image, P, radius, method='default')

        # Tính histogram
        hist, bins = np.histogram(lbp_image.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
        hist_normalized = hist / hist.sum()  # Chuẩn hóa histogram

        return lbp_image, hist, hist_normalized

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
    plt.figure(figsize=(16, 12))
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

    # Cân bằng histogram
    plt.figure(figsize=(16, 12))
    for i, ((radius, padding), (lbp_image, hist, hist_normalized)) in enumerate(results.items()):
        equalized_hist = cv2.equalizeHist((hist_normalized * 255).astype(np.uint8))

        plt.subplot(len(radii), len(paddings), i + 1)
        plt.title(f"Cân bằng Histogram R={radius}, Padding={padding}")
        plt.plot(equalized_hist, color='black')
        plt.xlabel("LBP Patterns")
        plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()
