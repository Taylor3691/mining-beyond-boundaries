from collections import Counter
from core.service_base import Visualization
import os
from image.dataset import ImageDataset
from config import PATH_FOLDER_IMAGE_TEST 
from visualization import plot_class_distribution 

class Imbalance(Visualization):
    def __init__(self):
        """
        Khởi tạo lớp phân tích sự mất cân bằng giữa các lớp dữ liệu.
        
        Input:
            Không có.
        
        Output:
            None.
        """
        self._step_name = "Analysis Class Imbalance"
        self._dataset_info = "Unknown Dataset"
        self._status = "Pending"
        self._result_data = {}      
        self._error_message = ""   

    def visitImageDataset(self, dataset: ImageDataset):
        """
        Thực hiện đếm số lượng mẫu trong từng lớp của tập dữ liệu.
        
        Input:
            dataset: Đối tượng dữ liệu đầu vào cần được xử lý.
        
        Output:
            None.
        """
        # Cập nhật thông tin tập dữ liệu đang xử lý
        folder = dataset.folder_path if dataset.folder_path else "Empty Path"
        self._dataset_info = f"ImageDataset (Path: {folder})"
        
        try:
            _, labels = dataset.images
            class_idx = dataset.class_idx

            if not labels:
                raise ValueError("Dataset hiện không có nhãn (labels) nào để phân tích!")

            counter = Counter(labels)
            idx_to_name = {v: k for k, v in class_idx.items()}
            
            # Tính toán kết quả
            class_counts = {idx_to_name.get(k, f"Unknown_{k}"): v for k, v in counter.items()}
            self._result_data = dict(sorted(class_counts.items(), key=lambda item: item[1], reverse=True))
            
            # Nếu chạy đến đây không có lỗi -> Thành công
            self._status = "Success"
            
        except Exception as e:
            # Nếu có bất kỳ lỗi nào xảy ra -> Thất bại
            self._status = "Failed"
            self._error_message = str(e)
            self._result_data = {}

        return

    def run(self, dataset: ImageDataset):
        """
        Chạy quy trình phân tích và vẽ biểu đồ phân phối lớp.
        
        Input:
            dataset: Đối tượng dữ liệu đầu vào cần được xử lý.
        
        Output:
            Giá trị trả về của hàm.
        """
        # Chạy logic xử lý
        self.visitImageDataset(dataset)
        
        # Gọi hàm log để in ra màn hình
        self.log()
        
        # Chỉ vẽ biểu đồ nếu xử lý thành công
        if self._status == "Success":
            plot_class_distribution(self._result_data.keys(), self._result_data.values())
        return self._result_data

    def log(self):
        """
        Thực hiện in ra màn hình các thông tin xử lý trong giai đoạn này bao gồm
        
        Input:
            Không có.
        
        Output:
            None.
        """
        print("\n" + "-"*50)
        print(f"1. Tên bước xử lý : {self._step_name}")
        print(f"2. Tập dữ liệu    : {self._dataset_info}")
        print(f"3. Trạng thái     : {self._status}")
        print(f"4. Kết quả        : ", end="")
        
        if self._status == "Success":
            print("Thống kê chi tiết bên dưới")
            total_images = sum(self._result_data.values())
            for cls_name, count in self._result_data.items():
                percentage = (count / total_images) * 100 if total_images > 0 else 0
                print(f"   + {cls_name:<12}: {count:<6} ảnh ({percentage:.2f}%)")
        else:
            print("Không có kết quả")
            print(f"\nChi tiết lỗi: {self._error_message}")
            
        print("-"*50 + "\n")

def main():
    """
    Thực thi xử lý trong hàm main.
    
    Input:
        Không có.
    
    Output:
        None.
    """
    try:
        dataset = ImageDataset(path=PATH_FOLDER_IMAGE_TEST) 
    except Exception as e:
        print(f"Lỗi khi khởi tạo Dataset: {e}")
        return

    imbalance_service = Imbalance()
    dataset.accept(imbalance_service)

if __name__ == "__main__":
    main()