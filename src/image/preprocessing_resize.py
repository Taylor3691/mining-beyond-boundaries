import sys
import os
import cv2
import numpy as np

# Đảm bảo Python nhận diện được thư mục gốc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import Preprocessing
from image import ImageDataset
from config import SUPPORT_RESIZE
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

class ImageResize(Preprocessing):
    """
    Khởi tạo các biến để lưu trữ thông tin log
    """
    def __init__(self, transform_size: int):
        if transform_size not in SUPPORT_RESIZE:
            raise ValueError(f"Not Support This Size. Supported sizes: {SUPPORT_RESIZE}")
        
        self._size = transform_size
        self._avg_ssim = 0.0
        self._avg_psnr = 0.0
        
        self._step_name = "Image Resizing and Quality Evaluation"
        self._dataset_path = "Unknown"
        self._status = "Pending"
        self._error_message = ""
        self._metadata = {}

    @property
    def avg_ssim(self):
        return self._avg_ssim
    
    @property
    def avg_psnr(self):
        return self._avg_psnr

    def fit(self, arr: np.ndarray):
        """Checks data rồi update metadata cho step"""
        if arr is None or len(arr) == 0:
            raise ValueError("Input array is empty. Cannot fit data.")
        
        self._metadata["original_count"] = len(arr)
        self._metadata["target_size"] = (self._size, self._size)
        self._metadata["original_shape"] = arr[0].shape
        return

    def transform(self, arr: list) -> list:
        """Resize toàn bộ dataset and trả về list chứa data của ảnh đã resized"""
        resized_arr = []
        for img in arr:
            resized = cv2.resize(img, (self._size, self._size), interpolation=cv2.INTER_AREA)
            resized_arr.append(resized)
        return resized_arr

    def fit_transform(self, arr: list) -> list:
        """Fit data để lấy metadata, sau đó transforms"""
        self.fit(arr)
        return self.transform(arr)

    def PSNR(self, original_imgs: list, resized_imgs: list) -> list:
        """Tính chỉ số PSNR giữa ảnh gốc và ảnh đã resized"""
        psnr_list = []
        for img1, img2 in zip(original_imgs, resized_imgs):
            # Upscale ảnh đã resized về lại bản gốc để so sánh 1-1 
            img2_restored = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_CUBIC)
            
            i1 = img1.astype(np.float64)
            i2 = img2_restored.astype(np.float64)
            
            mse = np.mean((i1 - i2) ** 2)
            if mse == 0:
                psnr_list.append(100.0)
            else:
                psnr = 20 * np.log10(255.0) - 10 * np.log10(mse)
                psnr_list.append(psnr)
                
        return psnr_list

    def SSIM(self, original_imgs: list, resized_imgs: list) -> list:
        """Tính chỉ số SSIM"""
        ssim_list = []
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        for img1, img2 in zip(original_imgs, resized_imgs):
            img2_restored = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_CUBIC)
            
            i1 = img1.astype(np.float64)
            i2 = img2_restored.astype(np.float64)
            
            mu1 = np.mean(i1)
            mu2 = np.mean(i2)
            var1 = np.var(i1)
            var2 = np.var(i2)
            cov12 = np.mean((i1 - mu1) * (i2 - mu2))
            
            ssim = ((2 * mu1 * mu2 + C1) * (2 * cov12 + C2)) / ((mu1**2 + mu2**2 + C1) * (var1 + var2 + C2))
            ssim_list.append(ssim)
            
        return ssim_list

    def visitImageDataset(self, obj: ImageDataset):
        """
        Processes the entire dataset, calculates average SSIM and average PSNR, 
        and updates the object.
        """
        self._dataset_path = obj.folder_path if obj.folder_path else "Unknown Path"
        
        try:
            # Đảm bảo data đã load 
            if obj._size == 0:
                obj.load()
                
            original_images, labels = obj.images
            
            if len(original_images) == 0:
                raise ValueError("Dataset contains no images to resize.")

            # Tiến hành resize bằng cách fit và transform 
            resized_images = self.fit_transform(original_images)

            # Tính các chỉ số PSNR và SSIM  
            psnr_array = self.PSNR(original_images, resized_images)
            ssim_array = self.SSIM(original_images, resized_images)
            
            self._avg_psnr = np.mean(psnr_array)
            self._avg_ssim = np.mean(ssim_array)

            # Cập nhật ImageDataset object
            obj.images = list(resized_images)
            obj._image_size = self._size  # Cập nhật metadata 
            
            self._status = "Success"
            
        except Exception as e:
            self._status = "Failed"
            self._error_message = str(e)

    def run(self, obj: ImageDataset):
        if isinstance(obj, ImageDataset):
            self.visitImageDataset(obj)
        return

    def log(self):
        """
        # Thực hiện in ra màn hình các thông tin xử lý trong giai đoạn này bao gồm
        # 1. Tên bước xử lý 
        # 2. Tập dữ liệu dữ lý
        # 3. Kích thước dữ liệu 
        # 4. Trạng thái (Success/Failed)
        # 5. In ra thông tin kết quả trả về khi tính toán
        """
        print(f"1. Processing Step : {self._step_name}")
        print(f"2. Target Dataset  : {self._dataset_path}")
        print(f"3. Target Size     : {self._size}x{self._size}")
        print(f"4. Status          : {self._status}")
        
        if self._status == "Success":
            print(f"5. Result Output:")
            print(f"   - Images processed    : {self._metadata.get('original_count', 0)}")
            print(f"   - Original Shape (1st): {self._metadata.get('original_shape', 'N/A')}")
            print(f"   - Average PSNR        : {self._avg_psnr:.2f} dB")
            print(f"   - Average SSIM        : {self._avg_ssim:.4f}")
            print("-" * 85)
        elif self._status == "Failed":
            print(f"5. Error Details   : {self._error_message}")
            print("-" * 85)

        print('\n')

# Hàm main này dùng để test kết qủa 
if __name__ == "__main__":
    #  Load bộ dataset vào RAM
    path = "./data/small" 
    original_dataset = ImageDataset(path=path)
    original_dataset.load()
    
    if original_dataset._size == 0:
        print("Error: Dataset is empty. Please check your data folder")
        sys.exit()
        
    print(f"Successfully loaded {original_dataset._size} original images\n")

    # Khởi tạo các size ảnh 
    target_sizes = [128, 64, 32]
    
    # Lưu tất cả kết quả vào biến results_summary 
    # results_summary = {}

    # Duyệt qua từng size 
    for size in target_sizes:
        print(f">>> Testing Resize Pipeline for: {size}x{size}")
        
        # Clone dataset cho mỗi size để đảm bảo mỗi size đều được giảm kích thước từ size của dataset gốc 
        working_dataset = original_dataset.clone()
        
        resize_service = ImageResize(transform_size=size)
        
        working_dataset.accept(resize_service)
        
        # Lưu bộ kết quả 
        if resize_service._status == "Success":
            #results_summary[size] = {
            #    "psnr": resize_service.avg_psnr,
            #    "ssim": resize_service.avg_ssim
            #}
            resize_service.log()
        else:
            print(f"Warning: Resize to {size} failed. Check the logs above.")
            #results_summary[size] = {"psnr": 0.0, "ssim": 0.0}

    # In ra tất cả kết quả thu đc từ biến results_summary
    #print(f"{'Target Size':<15} | {'Avg PSNR (dB)':<15} | {'Avg SSIM':<10}")
    #print("-" * 55)
    
    #for size in target_sizes:
    #    res = results_summary.get(size, {})
    #    psnr = res.get("psnr", 0.0)
    #    ssim = res.get("ssim", 0.0)
    #    print(f"{size}x{size:<12} | {psnr:<15.2f} | {ssim:<10.4f}")