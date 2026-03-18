import sys
import os
from collections import Counter

# Đảm bảo Python nhận diện được thư mục gốc để import visualization và core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import Visualization, Object
from image import ImageDataset
from visualization import plot_class_distribution 

class Imbalance(Visualization):
    def __init__(self):
        # Khởi tạo các biến để lưu trữ thông tin log
        self.step_name = "Analysis Imbalance Class Distribution"
        self.dataset_path = "Unknown"
        self.status = "Pending"
        self.error_message = ""
        self.total_samples = 0
        self.num_classes = 0
        self.results = [] # Lưu danh sách thống kê của từng lớp

    def run(self, obj: ImageDataset):
        # Kiểm tra đối tượng truyền vào có phải là ImageDataset không
        if isinstance(obj, ImageDataset):
            self.visitImageDataset(obj)
        return
    
    def visitImageDataset(self, obj: ImageDataset):
        # Task 11
        # Ở task này thực hiện logic để kiểm tra sự cân bằng số lượng giữa các class
        # Vẽ bar chart hoặc các loại chart phù hợp để thấy được sự chênh lệch giữa các class
        # Hàm nhận tham số có thể định nghĩa thủ tục tại comparision.py hoặc relationship.py tùy chart
        # Mong muốn khi vẽ thì highlight được 2 class lớn nhất và bé nhất


        # Lưu lại đường dẫn tập dữ liệu
        self.dataset_path = obj.folder_path if obj.folder_path else "Unknown Path"
        
        try:
            # Nạp dữ liệu vào biến obj 
            obj.load()

            # Lấy dữ liệu từ biến obj
            imgs, labels = obj.images
            file_names = obj._file_names
            self.total_samples = len(labels)
            
            # kiểm tra tập dữ liệu có trống không 
            if self.total_samples == 0:
                self.status = "Failed"
                self.error_message = "Dataset is empty!"
                return

            # Trích xuất tên lớp từ tên file (Ví dụ như 'dog_0.jpeg' -> 'dog') để lấy ra các thông tin cho mỗi lớp 
            idx_to_name = {}
            for label, fname in zip(labels, file_names):
                if label not in idx_to_name:
                    fname_str = str(fname)
                    class_name = fname_str.split('_')[0] if '_' in fname_str else f"Class_{label}"
                    idx_to_name[label] = class_name

            self.num_classes = len(idx_to_name)
            counts = Counter(labels)

            chart_class_names = []
            chart_counts = []

            # Lưu dữ liệu vào self.results
            for idx in sorted(idx_to_name.keys()):
                name = idx_to_name[idx]
                count = counts.get(idx, 0)
                ratio = (count / self.total_samples) * 100 if self.total_samples > 0 else 0
                
                chart_class_names.append(name)
                chart_counts.append(count)
                
                short_samples = [str(f) for l, f in zip(labels, file_names) if l == idx][:3]
                samples_str = ", ".join(short_samples) if short_samples else "No samples"
                
                # Đưa data vào danh sách kết quả (kiểu dữ liệu là Dictionary)
                self.results.append({
                    "id": idx,
                    "name": name,
                    "count": count,
                    "ratio": ratio,
                    "samples": samples_str
                })

            # Vẽ biểu đồ
            plot_class_distribution(chart_class_names, chart_counts)
            
            # Đánh dấu trạng thái thành công
            self.status = "Success"

        except Exception as e:
            # Bắt lỗi nếu có bất kỳ sự cố nào xảy ra trong quá trình tính toán
            self.status = "Failed"
            self.error_message = str(e)
            
        return
    
    def log(self):
        # Thực hiện in ra màn hình các thông tin xử lý trong giai đoạn này bao gồm
        # 1. Tên bước xử lý (VD: Analysis Histogram Distribution)
        # 2. Tập dữ liệu dữ lý
        # 3. Trạng thái (Success/Failed)
        # 4. In ra thông tin kết quả trả về khi tính toán, linh hoạt trong việc chỉnh sửa hàm


        print("=" * 85)
        print(f"1. Processing Step : {self.step_name}")
        print(f"2. Target Dataset  : {self.dataset_path}")
        print(f"3. Status          : {self.status}")
        print("=" * 85)
        
        if self.status == "Success":
            print(f"4. Result Output:")
            print(f"   - Total samples : {self.total_samples}")
            print(f"   - Total classes : {self.num_classes}")
            print("-" * 85)
            # print(f"{'ID':<4} | {'Class Name':<15} | {'Count':<10} | {'Ratio (%)':<10} | {'Samples (File Names)'}")
            print(f"{'ID':<4} | {'Class Name':<15} | {'Count':<10} | {'Ratio (%)':<10}")
            print("-" * 85)
            
            # Duyệt qua danh sách kết quả đã được tính toán ở hàm visit và in ra màn hình console 
            for res in self.results:
                # print(f"{res['id']:<4} | {res['name']:<15} | {res['count']:<10} | {res['ratio']:>8.2f}% | {res['samples']}")
                print(f"{res['id']:<4} | {res['name']:<15} | {res['count']:<10} | {res['ratio']:>8.2f}%")
            print("-" * 85)
            
        elif self.status == "Failed":
            print(f"4. Error Details   : {self.error_message}")
            print("-" * 85)


if __name__ == "__main__":
    path = "./data/small" 
    my_dataset = ImageDataset(path=path)
    imbalance_service = Imbalance()
    
    # Chạy hàm tính toán (sẽ không in gì ra console, chỉ bật biểu đồ)
    # Sau khi tắt biểu đồ được bật ra thì chương trình sẽ thực hiện các dòng lệnh tiếp theo 
    my_dataset.accept(imbalance_service)
    
    # Gọi hàm log để in toàn bộ thông tin
    imbalance_service.log()