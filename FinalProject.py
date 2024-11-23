import Bai1
import Bai2
import Bai3

from tkinter import Tk, Label, Button, filedialog

# Hàm chọn tệp
def browse_file():
    global file_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")])
    if file_path:
        lbl_status.config(text=f"Đã chọn: {file_path}")
    else:
        lbl_status.config(text="Chưa chọn tệp")

# Gọi hàm Bai1 với file_path
def run_bai1():
    if file_path:
        Bai1.Bai1(file_path)
    else:
        lbl_status.config(text="Vui lòng chọn tệp trước khi chạy Bài 1!")

# Gọi hàm Bai2 với file_path
def run_bai2():
    if file_path:
        Bai2.Bai2(file_path)
    else:
        lbl_status.config(text="Vui lòng chọn tệp trước khi chạy Bài 2!")

# Gọi hàm Bai3 với file_path
def run_bai3():
    if file_path:
        Bai3.Bai3(file_path)
    else:
        lbl_status.config(text="Vui lòng chọn tệp trước khi chạy Bài 3!")

# Giao diện Tkinter
file_path = None

root = Tk()
root.title("Chương trình xử lý ảnh")

# Nhãn và nút chọn tệp
lbl_status = Label(root, text="Chưa chọn tệp", fg="red")
lbl_status.pack()

btn_browse = Button(root, text="Chọn tệp ảnh", command=browse_file)
btn_browse.pack()

# Nút gọi các bài tập
btn_bai1 = Button(root, text="Bài 1", command=run_bai1)
btn_bai1.pack()

btn_bai2 = Button(root, text="Bài 2", command=run_bai2)
btn_bai2.pack()

btn_bai3 = Button(root, text="Bài 3", command=run_bai3)
btn_bai3.pack()

# Khởi chạy giao diện
root.mainloop()
