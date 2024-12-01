import math
import matplotlib.pyplot as plt
import imageio.v2 as imageio


def read_image(file_path):
    """Đọc ảnh từ file dưới dạng ma trận số nguyên 8-bit."""
    image = imageio.imread(file_path)
    if image.max() <= 1.0:  # Nếu ảnh được chuẩn hóa trong khoảng [0, 1]
        image = [[int(p * 255) for p in row] for row in image]
    return image


def convert_to_grayscale(image):
    """Chuyển đổi ảnh màu sang ảnh xám."""
    gray_image = []
    for row in image:
        gray_row = []
        for pixel in row:
            r, g, b = pixel[:3]
            gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
            gray_row.append(gray_value)
        gray_image.append(gray_row)
    return gray_image


def pad_image(image, padding=0):
    """Thêm padding vào ảnh."""
    h, w = len(image), len(image[0])
    padded_image = [[0] * (w + 2 * padding) for _ in range(h + 2 * padding)]
    for i in range(h):
        for j in range(w):
            padded_image[i + padding][j + padding] = image[i][j]
    return padded_image


def compute_angles(P):
    """Tính toán các góc thủ công."""
    angles = [2 * math.pi * i / P for i in range(P)]
    return angles


def bilinear_interpolate(image, x, y):
    """Nội suy bilinear thủ công."""
    x0 = int(x)
    x1 = min(x0 + 1, len(image[0]) - 1)
    y0 = int(y)
    y1 = min(y0 + 1, len(image) - 1)

    a = x - x0
    b = y - y0

    value = (1 - a) * (1 - b) * image[y0][x0] + a * (1 - b) * image[y0][x1] + \
            (1 - a) * b * image[y1][x0] + a * b * image[y1][x1]
    return value


def local_binary_pattern(image, P, radius):
    """Tính toán Local Binary Pattern (LBP)."""
    height, width = len(image), len(image[0])
    lbp_image = [[0 for _ in range(width)] for _ in range(height)]
    angles = compute_angles(P)
    dx = [radius * math.cos(angle) for angle in angles]
    dy = [-radius * math.sin(angle) for angle in angles]

    for x in range(radius, width - radius):
        for y in range(radius, height - radius):
            center = image[y][x]
            lbp_value = 0
            for i in range(P):
                nx, ny = x + dx[i], y + dy[i]
                neighbor = bilinear_interpolate(image, nx, ny)
                if neighbor >= center:
                    lbp_value |= (1 << i)
            lbp_image[y][x] = lbp_value
    return lbp_image


def compute_lbp_and_histogram(image, radius, padding):
    """Tính toán LBP và histogram."""
    padded_image = pad_image(image, padding)
    P = 8 * radius
    lbp_image = local_binary_pattern(padded_image, P, radius)

    hist = [0] * (P + 2)
    for row in lbp_image:
        for value in row:
            if 0 <= value < len(hist):
                hist[value] += 1

    total = sum(hist)
    hist_normalized = [h / total for h in hist]
    return lbp_image, hist, hist_normalized


def equalize_histogram(hist):
    """Cân bằng histogram."""
    cdf = [sum(hist[:i + 1]) for i in range(len(hist))]  # Tính CDF
    cdf_min = min(cdf)
    cdf_max = max(cdf)
    cdf_normalized = [(value - cdf_min) * 255 // (cdf_max - cdf_min) for value in cdf]
    return [min(max(value, 0), 255) for value in cdf_normalized]  # Giới hạn giá trị trong [0, 255]


def plot_histogram_combined_per_radius(ax, hist, P, radius):
    """Vẽ histogram dạng kết hợp cho mỗi radius trên cùng một trục."""
    combined_hist = []
    group_size = P // 8  # Mỗi nhóm 8-bit
    for i in range(8):
        combined_hist += hist[i * group_size:(i + 1) * group_size]

    # Vẽ histogram dạng kết hợp trên trục `ax`
    ax.bar(range(len(combined_hist)), combined_hist, color='gray')
    ax.set_title(f"Histogram Combined (R={radius})")
    ax.set_xlabel("LBP Patterns")
    ax.set_ylabel("Frequency")


def Bai3(file_path):
    # Đọc ảnh và chuyển sang ảnh xám
    image = read_image(file_path)
    gray_image = convert_to_grayscale(image)

    # Thiết lập thông số
    radii = [1, 2, 3]
    paddings = [0, 1]

    # Hiển thị kết quả từng trường hợp
    for radius in radii:
        P = 8 * radius
        for padding in paddings:
            lbp_image, hist, hist_normalized = compute_lbp_and_histogram(gray_image, radius, padding)
            equalized_hist = equalize_histogram(hist)

            # Tạo hình và các trục
            fig, axes = plt.subplots(1, 4, figsize=(24, 6))

            # Hiển thị LBP
            axes[0].imshow(lbp_image, cmap='gray')
            axes[0].set_title(f"LBP (R={radius}, Padding={padding})")
            axes[0].axis('off')

            # Hiển thị Histogram rời rạc
            axes[1].bar(range(len(hist)), hist_normalized, color='black')
            axes[1].set_title(f"Histogram (R={radius}, Padding={padding})")
            axes[1].set_xlabel("LBP Patterns")
            axes[1].set_ylabel("Frequency")
            if radius in [2, 3]:
                plot_histogram_combined_per_radius(axes[2], hist, P, radius)
            # Hiển thị cân bằng Histogram
            axes[3].bar(range(len(equalized_hist)), equalized_hist, color='black')
            axes[3].set_title(f"Cân bằng Histogram (R={radius}, Padding={padding})")
            axes[3].set_xlabel("LBP Patterns")
            axes[3].set_ylabel("Frequency")
            # Hiển thị Histogram Combined nếu R=2 hoặc R=3


            # Sắp xếp hiển thị
            plt.tight_layout()
            plt.show()
        # Vẽ histogram dạng kết hợp cho các trường hợp R=2 và R=3



# Gọi hàm với ảnh đầu vào
Bai3('picture.jpg')
