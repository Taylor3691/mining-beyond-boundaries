from core import DistributionTesting
from image import ImageDataset

class KolmogorovSmirnovTesting(DistributionTesting):
    def __init__(self):
        # Khởi tạo các metadata cần thiết
        return

    def visitImageDataset(self, obj: ImageDataset):
        # Logic chính thực hiện 
        # Chuẩn bị dữ liệu ở đây hoặc define thêm hàm cbi
        #  dữ liệu
        # Dữ liệu ở đây là dạng ảnh, nên cần làm flattern nếu cần
        # Mặc dataset đã có 2 biến là self._orgin_images và _processed_images
        # là 2 mảng các ảnh, ông xử lý trên các mảng này
        # Nếu không có 2 attribute này thì rasie Error  
        return
    
    def run(self, obj: ImageDataset):
        if isinstance(obj, ImageDataset):
            self.visitImageDataset(obj)
        return 
    
    def log(self):
        # Thực hiện trình bày được các thông tin sau
        # 1. Tên step
        # 2. Tên kiểm định 
        # 3. Giả thuyết H0
        # 4: Kết quả trả về có p-value
        # 5: Kết luận bác bỏ hoặc chấp nhận H0
        # 6: Kết luận và nhắc lại giả thuyết H0 nếu chấp nhận ngược lại
        return
    
    def test(self):
        # Nhận vào 2 mảng dữ liệu phẳng là 2 phân phối
        # Thực hiện gọi thư viện spciy để kiểm định
        # Nếu cần thiết thì có thể tạo thêm các metadata khác trả về 
        return