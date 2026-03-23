from image import ImageDataset
from core import Preprocessing
from config import SUPPORT_RESIZE
import numpy as np
# Task 21: Thực hiện resize ảnh và tính độ đo SSIM và PSNR
# Hàm SSIM() và hàm PSNR() nhận đầu vào là 2 mảng ảnh trước và sau resize, trả về MẢNG chỉ số khác nhau giữa 2 ảnh cùng index
# Hàm run() giữ nguyên cấu trúc
# Hàm visitImageDataset() xử lý trên toàn bộ dữ liệu và tính SSIM trung bình và tính PSNR trung bình trên toàn bộ dataset
# Hàm fit() dùng để kiểm tra dữ liệu và cập nhật metadata của step
# Hàm transform() chỉ dùng để resize toàn bộ dataset truyền vào và trả về bộ dataset mới với size mới (Các ảnh giữ nguyên index)
# Log() thì in ra các thông tin cần thiết như các bước cũ, nhớ in chỉ số trung bình PSNR và trung bình SSIM
# Có thể thêm các hàm con để hỗ trợ tính các tham số như muy,var trong công thức PSNR, để trong utils
# define các property
# Thực hiện vẽ đường cong tại relationship.py

# Chú ý: Có thể tách thành nhiều hàm nhỏ trong VisitImageDataset()
class ImageResize(Preprocessing):
    def __init__(self, transform_size: int):
        if transform_size not in SUPPORT_RESIZE:
            raise ValueError("Not Support This Size")
        self._size = transform_size
        return
    

    @property
    def avg_ssim():
        return
    
    @property
    def avg_psnr():
        return
    
    def fit(self, arr: np.ndarray):
        return
    
    def transform(self, arr: np.ndarray):
        return
    
    def fit_transform(self, arr: np.ndarray):
        self.fit(arr)
        return self.transform(arr)
    
    def log(self):
        return
    
    def run(self, obj: ImageDataset):
        if isinstance(obj, ImageDataset):
            self.visitImageDataset(obj)
        return
    
    def visitImageDataset(self, obj: ImageDataset):
        pass
    
    def SSIM():
        return
    
    def PSNR():
        return