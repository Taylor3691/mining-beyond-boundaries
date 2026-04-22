import os
import shutil
from core import Preprocessing
from image.dataset import ImageDataset

class RemoveDuplication(Preprocessing):
    """
    A service to remove duplicate images from an ImageDataset based on a
    pre-determined set of indices. It moves those duplicate files to a separate directory.
    """
    def __init__(self, indices_to_remove: set, output_dir: str = None):
        """
        Khởi tạo lớp thực hiện loại bỏ ảnh trùng lặp.
        
        Input:
            indices_to_remove: Tập chỉ số ảnh trùng lặp cần loại bỏ khỏi dataset.
            output_dir: Đường dẫn tệp hoặc thư mục liên quan đến dữ liệu.
        
        Output:
            None.
        """
        if not isinstance(indices_to_remove, set):
            # Ép kiểu an toàn nếu lỡ truyền list
            indices_to_remove = set(indices_to_remove)
            
        self._indices_to_remove = indices_to_remove
        self._output_dir = output_dir
        self._status = "Not Run"
        self._removed_count = 0
        self._moved_count = 0

    def fit(self, X, y=None):
        """
        Hàm fit tương thích với kiến trúc pipeline.
        
        Input:
            X: Dữ liệu đặc trưng hoặc nhãn dùng cho huấn luyện/đánh giá.
            y: Dữ liệu đặc trưng hoặc nhãn dùng cho huấn luyện/đánh giá.
        
        Output:
            Giá trị trả về của hàm.
        """
        return self

    def transform(self, X):
        """
        Hàm biến đổi (không thực hiện gì trong RAM cho X thô).
        
        Input:
            X: Dữ liệu đặc trưng hoặc nhãn dùng cho huấn luyện/đánh giá.
        
        Output:
            Giá trị trả về của hàm.
        """
        return X

    def fit_transform(self, X, y=None):
        """
        Thực hiện fit và transform.
        
        Input:
            X: Dữ liệu đặc trưng hoặc nhãn dùng cho huấn luyện/đánh giá.
            y: Dữ liệu đặc trưng hoặc nhãn dùng cho huấn luyện/đánh giá.
        
        Output:
            Giá trị trả về của hàm.
        """
        return self.transform(X)

    def visitImageDataset(self, obj: ImageDataset):
        """
        Cổng kết nối để thực thi trên ImageDataset.
        
        Input:
            obj: Đối tượng dữ liệu đầu vào cần được xử lý.
        
        Output:
            Giá trị trả về của hàm.
        """
        self.run(obj)
        return self

    def run(self, obj: ImageDataset):
        """
        Thực hiện di chuyển file vật lý và cập nhật lại danh sách của Dataset.
        
        Input:
            obj: Đối tượng dữ liệu đầu vào cần được xử lý.
        
        Output:
            None.
        """
        initial_count = len(obj.image_paths)
        if initial_count == 0 or not self._indices_to_remove:
            print("Dataset is empty or no duplicates to remove.")
            self._status = "Success"
            self.log(initial_count, initial_count)
            return

        try:
            # 1. Di chuyển file vật lý
            if self._output_dir and len(self._indices_to_remove) > 0:
                os.makedirs(self._output_dir, exist_ok=True)
                for idx in self._indices_to_remove:
                    source_path = obj._paths[idx]
                    file_name = obj._file_names[idx]
                    
                    # Giữ nguyên cấu trúc class
                    class_name = os.path.basename(os.path.dirname(source_path))
                    dest_folder = os.path.join(self._output_dir, class_name)
                    os.makedirs(dest_folder, exist_ok=True)
                    
                    destination_path = os.path.join(dest_folder, file_name)
                    
                    try:
                        shutil.move(source_path, destination_path)
                        self._moved_count += 1
                    except Exception as copy_err:
                        print(f"Warning: Failed to move {file_name}: {copy_err}")

            # 2. Loại bỏ thông tin khỏi RAM (Cập nhật các list của Dataset)
            indices_to_keep = [i for i in range(initial_count) if i not in self._indices_to_remove]
            
            # Chỉ slice mảng ảnh nếu nó đã được load lên RAM toàn bộ
            if hasattr(obj, '_images') and obj._images:
                obj._images = [obj._images[i] for i in indices_to_keep]
                
            obj._labels = [obj._labels[i] for i in indices_to_keep]
            obj._paths = [obj._paths[i] for i in indices_to_keep]
            obj._file_names = [obj._file_names[i] for i in indices_to_keep]
            
            new_size = len(obj._paths)
            obj._size = new_size
            
            if hasattr(obj, '_shape') and isinstance(obj._shape, tuple) and len(obj._shape) == 4:
                obj._shape = (new_size, obj._shape[1], obj._shape[2], obj._shape[3])
            elif hasattr(obj, '_shape') and len(obj._shape) == 0:
                # Trường hợp shape đang là (0,0,0,0) do init
                pass
            
            self._removed_count = initial_count - new_size
            self._status = "Success"
            
        except Exception as e:
            self._status = f"Failed: {e}"
            print(f"An error occurred during removal process: {e}")
        
        self.log(initial_count, obj._size)
            
    def log(self, initial_count, final_count):
        """
        In thông tin tóm tắt về số lượng ảnh đã xóa và trạng thái sau khi xử lý.
        
        Input:
            initial_count: Số lượng mẫu hoặc số lần lặp dùng trong xử lý/đánh giá.
            final_count: Số lượng mẫu hoặc số lần lặp dùng trong xử lý/đánh giá.
        
        Output:
            None.
        """
        print("\n--- Remove Duplication Preprocessing Log ---")
        print(f"1. Processing Step: Preprocessing - Remove Duplicates")
        print(f"2. Status: {self._status}")
        if self._status == "Success":
            print(f"\t- Initial Image Count: {initial_count}")
            if self._output_dir:
                print(f"\t- Files Moved to '{self._output_dir}': {self._moved_count}")
            print(f"\t- Dataset Trackers Updated (Removed from Lists): {self._removed_count}")
            print(f"\t- Final Image Count: {final_count}")
        print("------------------------------------------\n")