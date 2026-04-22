import sys
import os
import cv2
import numpy as np
import concurrent.futures 

from config import PATH_FOLDER_IMAGE_TEST 
from core.service_base import Preprocessing
from image.dataset import ImageDataset
from config import SUPPORT_RESIZE
from utils.metrics import calculate_single_psnr, calculate_single_ssim
from visualization.relationship import plot_ssim_curve

class ImageResize(Preprocessing):
    def __init__(self, transform_size: int):
        """
        Khởi tạo lớp thay đổi kích thước ảnh và đánh giá chất lượng.
        
        Input:
            transform_size: Kích thước hoặc hình dạng mục tiêu trong quá trình xử lý.
        
        Output:
            None.
        """
        if transform_size not in SUPPORT_RESIZE:
            raise ValueError(f"Not Support This Size. Supported sizes: {SUPPORT_RESIZE}")
        
        self._size = transform_size
        self._avg_ssim = 0.0
        self._avg_psnr = 0.0
        
        self._step_name = "Image Resizing and Quality Evaluation"
        self._dataset_path = "Unknown"
        self._status = "Pending"
        self._error_message = ""
        self._metadata = {
            "processed_count": 0,
            "target_size": (self._size, self._size)
        }
        
        self._max_workers = min(16, (os.cpu_count() or 1) + 4)

    @property
    def avg_ssim(self):
        """
        Thực thi xử lý trong hàm avg_ssim.
        
        Input:
            Không có.
        
        Output:
            Giá trị trả về của hàm.
        """
        return self._avg_ssim
    
    @property
    def avg_psnr(self):
        """
        Thực thi xử lý trong hàm avg_psnr.
        
        Input:
            Không có.
        
        Output:
            Giá trị trả về của hàm.
        """
        return self._avg_psnr

    def fit(self, arr: list):
        """
        Cập nhật thông tin hình dạng ảnh gốc và đếm số lượng ảnh đã xử lý.
        
        Input:
            arr: Mảng dữ liệu đầu vào cần xử lý.
        
        Output:
            None.
        """
        if not arr:
            raise ValueError("Input array is empty. Cannot fit data.")
        
        # Chỉ cập nhật original_shape 1 lần từ batch đầu tiên
        if "original_shape" not in self._metadata:
            self._metadata["original_shape"] = arr[0].shape
            
        self._metadata["processed_count"] += len(arr)
        return

    def transform(self, arr: list) -> list:
        """
        Thực hiện thay đổi kích thước cho một danh sách ảnh sử dụng đa luồng.
        
        Input:
            arr: Mảng dữ liệu đầu vào cần xử lý.
        
        Output:
            Giá trị trả về của hàm.
        """
        def _resize_single(img):
            """
            Thực thi xử lý trong hàm _resize_single.
            
            Input:
                img: Ảnh đầu vào dạng mảng NumPy.
            
            Output:
                Giá trị trả về của hàm.
            """
            return cv2.resize(img, (self._size, self._size), interpolation=cv2.INTER_AREA)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            resized_list = list(executor.map(_resize_single, arr))
            
        return resized_list

    def fit_transform(self, arr: list) -> list:
        """
        Vừa cập nhật metadata vừa thực hiện resize ảnh.
        
        Input:
            arr: Mảng dữ liệu đầu vào cần xử lý.
        
        Output:
            Giá trị trả về của hàm.
        """
        self.fit(arr)
        return self.transform(arr)

    def PSNR(self, original_imgs: list, resized_imgs: list) -> np.ndarray:
        """
        Tính toán chỉ số PSNR (Peak Signal-to-Noise Ratio) để đánh giá độ thất thoát chất lượng.
        
        Input:
            original_imgs: Danh sách ảnh gốc trước khi resize.
            resized_imgs: Danh sách ảnh sau khi resize để đo mức suy hao chất lượng.
        
        Output:
            Giá trị trả về của hàm.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            psnr_list = list(executor.map(calculate_single_psnr, original_imgs, resized_imgs))
        return np.array(psnr_list)

    def SSIM(self, original_imgs: list, resized_imgs: list) -> np.ndarray:
        """
        Tính toán chỉ số SSIM (Structural Similarity Index) để đánh giá độ tương đồng cấu trúc.
        
        Input:
            original_imgs: Danh sách ảnh gốc trước khi resize.
            resized_imgs: Danh sách ảnh sau khi resize để đo mức tương đồng cấu trúc.
        
        Output:
            Giá trị trả về của hàm.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            ssim_list = list(executor.map(calculate_single_ssim, original_imgs, resized_imgs))
        return np.array(ssim_list)

    def visitImageDataset(self, obj: ImageDataset):
        """
        Xử lý quy trình resize và tính toán chất lượng cho đối tượng ImageDataset theo batch.
        
        Input:
            obj: Đối tượng dữ liệu đầu vào cần được xử lý.
        
        Output:
            None.
        """
        self._dataset_path = obj.folder_path if obj.folder_path else "Unknown Path"
        
        try:
            total_psnr_scores = []
            total_ssim_scores = []
            all_resized_images = []
            
            # Lặp qua từng batch dữ liệu thay vì load toàn bộ
            for batch_imgs, batch_indices in obj.load():
                if not batch_imgs:
                    continue

                # 1. Resize batch
                resized_batch = self.fit_transform(batch_imgs)
                
                # 2. Tính Metrics cho batch hiện tại
                batch_psnr = self.PSNR(batch_imgs, resized_batch)
                batch_ssim = self.SSIM(batch_imgs, resized_batch)
                
                # 3. Gom kết quả
                total_psnr_scores.extend(batch_psnr)
                total_ssim_scores.extend(batch_ssim)
                
                # Gom ảnh đã resize (Lưu ý: Lưu toàn bộ ảnh vào RAM)
                all_resized_images.extend(resized_batch)

            # Tính trung bình toàn bộ Dataset
            self._avg_psnr = float(np.mean(total_psnr_scores))
            self._avg_ssim = float(np.mean(total_ssim_scores))

            # Trả về bộ dataset mới với size mới (giữ nguyên index ban đầu)
            obj.images = all_resized_images
            
            # Update lại size trong metadata của dataset
            obj._image_size = (self._size, self._size) 
            
            self._status = "Success"
            
        except Exception as e:
            self._status = "Failed"
            self._error_message = str(e)

    def run(self, obj: ImageDataset):
        """
        Thực thi quá trình tiền xử lý resize.
        
        Input:
            obj: Đối tượng dữ liệu đầu vào cần được xử lý.
        
        Output:
            None.
        """
        self.visitImageDataset(obj)
        return

    def log(self):
        """
        In log kết quả về kích thước mục tiêu, số lượng xử lý và chất lượng ảnh (PSNR, SSIM).
        
        Input:
            Không có.
        
        Output:
            None.
        """
        print("\n" + "="*50)
        print(f"1. Processing Step : {self._step_name}")
        print(f"2. Target Dataset  : {self._dataset_path}")
        print(f"3. Target Size     : {self._size}x{self._size}")
        print(f"4. Status          : {self._status}")
        
        if self._status == "Success":
            print(f"5. Result Output:")
            print(f"   - Images processed    : {self._metadata.get('processed_count', 0)}")
            print(f"   - Original Shape (1st): {self._metadata.get('original_shape', 'N/A')}")
            print(f"   - Average PSNR        : {self._avg_psnr:.2f} dB")
            print(f"   - Average SSIM        : {self._avg_ssim:.4f}")
        else:
            print(f"5. Error Details   : {self._error_message}")
        print("="*50 + "\n")


# Hamf main dung de testttttt
if __name__ == "__main__":
    original_dataset = ImageDataset(path=PATH_FOLDER_IMAGE_TEST)
    
    if len(original_dataset.image_paths) == 0:
        print(f"Error: Dataset is empty. Please check your data folder at: {PATH_FOLDER_IMAGE_TEST}")
        sys.exit()
        
    # In ra số lượng dựa vào độ dài danh sách image_paths
    print(f"Successfully connected to dataset ({len(original_dataset.image_paths)} images)\n")

    target_sizes = [128, 64, 32]
    results_summary = {}

    for size in target_sizes:
        print(f">>> Testing Resize Pipeline for: {size}x{size}")
        
        # Clone thông tin mảng rỗng và metadata để chuẩn bị cho batch loader
        working_dataset = original_dataset.clone()
        
        resize_service = ImageResize(transform_size=size)
        
        # Chạy pipeline - generator obj.load() sẽ tự động kích hoạt bên trong hàm visitImageDataset
        working_dataset.accept(resize_service)
        
        if resize_service._status == "Success":
            results_summary[size] = {
                "psnr": resize_service.avg_psnr,
                "ssim": resize_service.avg_ssim
            }
            resize_service.log()
        else:
            print(f"Warning: Resize to {size} failed. Check the logs above.")
            results_summary[size] = {"psnr": 0.0, "ssim": 0.0}

    # Vẽ biểu đồ SSIM Curve
    plot_sizes = list(results_summary.keys())
    plot_ssims = [results_summary[s]["ssim"] for s in plot_sizes]
    
    try:
        plot_ssim_curve(sizes=plot_sizes, ssim_scores=plot_ssims)
    except Exception as e:
        print(f"Lỗi khi vẽ biểu đồ: {e}")