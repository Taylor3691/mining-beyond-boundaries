from image import ImageDataset
from core import Preprocessing

# Task 21: Thực hiện resize ảnh và tính độ đo SSIM và PSNR
# Hàm SSIM() và hàm PSNR() nhận đầu vào là 2 ảnh và kết quả trả về là chỉ số tính toán được
# Hàm run() giữ nguyên cấu trúc
# Hàm visitImageDataset() xử lý trên toàn bộ dữ liệu và tính SSIM trung bình và tính PSNR trung bình trên toàn bộ dataset
# Hàm addSize() dùng để thêm 1 size vào trong kế hoạch khảo sát
# Hàm plot là hàm visualize trên toàn bộ dataset với các kích thước nhận vào 1 list các chỉ sô SSIM và vẽ line chart 
#(hoặc đường cong gì đó thầy bảo)


class ImageResize(Preprocessing):
    def __init__(self, transform_size: int):
        self._listTransform = [32,64,128]
        return
    
    def log(self):
        return super().log()
    
    def run(self, obj: ImageDataset):
        if isinstance(obj, ImageDataset):
            self.visitImageDataset(obj)
        return
    
    def SSIM():
        return
    
    def PSNR():
        return
    
    def addSize():
        return

    def visitImageDataset():
        pass

    def plot():
        pass