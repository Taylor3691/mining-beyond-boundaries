from core import Visualization, Object
from image import ImageDataset

class Imbalance(Visualization):
    def run():
        return
    
    def visitImageDataset(obj: ImageDataset):
        # Task 11
        # Ở task này thực hiện logic để kiểm tra sự cân bằng số lượng giữa các class
        # Vẽ bar chart hoặc các loại chart phù hợp để thấy được sự chênh lệch giữa các class
        # Hàm nhận tham số có thể định nghĩa thủ tục tại comparision.py hoặc relationship.py tùy chart
        # Mong muốn khi vẽ thì highlight được 2 class lớn nhất và bé nhất
        return
    
    def log(self):
        # Thực hiện in ra màn hình các thông tin xử lý trong giai đoạn này bao gồm
        # 1. Tên bước xử lý (VD: Analysis Histogram Distribution)
        # 2. Tập dữ liệu dữ lý
        # 3. Trạng thái (Success/Failed)
        # 4. In ra thông tin kết quả trả về khi tính toán, linh hoạt trong việc chỉnh sửa hàm
        return