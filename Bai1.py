import cv2
import numpy as np
import matplotlib.pyplot as plt

def Bai1(file_path):
    # Đọc ảnh màu và chuyển sang ảnh xám
    image = cv2.imread(file_path)  # Đổi 'image.jpg' thành đường dẫn ảnh của bạn
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Vẽ histogram ban đầu (H1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.hist(gray_image.ravel(), bins=256, range=(0, 256), color='black')
    plt.title('Histogram ban đầu (H1)')
    plt.xlabel('Pixel intensity')
    plt.ylabel('Frequency')

    # Histogram cân bằng (H2)
    equalized_image = cv2.equalizeHist(gray_image)
    plt.subplot(1, 3, 2)
    plt.hist(equalized_image.ravel(), bins=256, range=(0, 256), color='black')
    plt.title('Histogram cân bằng (H2)')
    plt.xlabel('Pixel intensity')
    plt.ylabel('Frequency')

    # Hiệu chỉnh thu hẹp H2 trong khoảng (50, 100)
    # Chuyển đổi cường độ pixel từ khoảng [0, 255] sang [50, 100]
    narrowed_image = cv2.normalize(equalized_image, None, 50, 100, cv2.NORM_MINMAX)
    plt.subplot(1, 3, 3)
    plt.hist(narrowed_image.ravel(), bins=256, range=(0, 256), color='black')
    plt.title('Histogram thu hẹp (50,100)')
    plt.xlabel('Pixel intensity')
    plt.ylabel('Frequency')

    # Hiển thị kết quả
    plt.tight_layout()
    plt.show()

    # Lưu các ảnh nếu cần thiết
    cv2.imwrite('gray_image.jpg', gray_image)
    cv2.imwrite('equalized_image.jpg', equalized_image)
    cv2.imwrite('narrowed_image.jpg', narrowed_image)
