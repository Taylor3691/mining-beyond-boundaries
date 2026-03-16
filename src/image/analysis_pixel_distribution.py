from core import Visualization, Object
from image import ImageDataset

class Distribution(Visualization):
    pass

class HistogramDistribution(Distribution):
    def run(self, obj: Object):
        return
    
    def visitImageDataset(obj: ImageDataset):
        # Task 7, Task 8
        # Ở mục này, nhận vào một đối tượng là folder Image 
        # Làm 2 việc
        # 1. Thực hiện lấy các thông tin cần thiết từ data qua các hàm
        # 2. Xây dựng hàm histogram nhận các thông tin trên và vẽ trên file distributuion.py (Khai báo thủ thục bình thường)
        return

    def log(self):
        # Thực hiện in ra màn hình các thông tin xử lý trong giai đoạn này bao gồm
        # 1. Tên bước xử lý (VD: Analysis Histogram Distribution)
        # 2. Tập dữ liệu dữ lý
        # 3. Trạng thái (Success/Failed)
        # 4. In ra thông tin kết quả trả về khi tính toán, linh hoạt trong việc chỉnh sửa hàm
        return
    
class KDEDistribution(Visualization):
    def run():
        return
    
    def visitImageDataset(obj: ImageDataset):
        # Thực hiện như Histogram
        return
    
    def log(self):
        # Thực hiện như Histogram
        return