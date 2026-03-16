from core import Visualization, Object
from image import ImageDataset

class ConstrastAndBrightness(Visualization):
    def run():
        return
    
    def visitImageDataset(obj: ImageDataset):
        # Task 15, Task 16
        # Ở chỗ này thì tính 2 giá trị độ sáng theo công thức và visualize nó
        # Hàm visualize code riêng và hàm này nhận thông tin 2 giá trị (hoặc các giá trị khác) vừa tính để tạo box plot
        return
    
    def log(self):
        # Thực hiện in ra màn hình các thông tin xử lý trong giai đoạn này bao gồm
        # 1. Tên bước xử lý (VD: Analysis Histogram Distribution)
        # 2. Tập dữ liệu dữ lý
        # 3. Trạng thái (Success/Failed)
        # 4. In ra thông tin kết quả trả về khi tính toán, linh hoạt trong việc chỉnh sửa hàm
        return