import cv2
import gc
import numpy as np
from core.service_base import Visualization
from core.data_base import Object
from image.dataset import ImageDataset
from visualization.distribution import plot_histogram, plot_kde

# Task 7

class Distribution(Visualization):
    def __init__(self):
        self.step_name = "Base Analysis"
        self.dataset_name = "Unknown"
        self.status = "Pending"

    def log(self):
        print(f"Bước xử lý : {self.step_name}")
        print(f"Tập dữ liệu: {self.dataset_name}")
        print(f"Trạng thái : {self.status}")

    # Thêm tham số target_size để ép size ảnh gọn nhẹ ngay từ lúc nạp
    def _extract_pixels(self, obj: ImageDataset, max_samples=200000, target_size=(64, 64)):
        """ Task 7: Lấy thông tin pixel, giữ cấu trúc 3 kênh màu và áp dụng Sampling """
        print(f"[PROCESS] Đang trích xuất pixel theo Batch (Resize về {target_size})...")
        pixels_list = []
        total_images = len(obj.image_paths)
        processed = 0

        # Thay vì lấy obj.images (có thể gây tràn RAM), ta dùng Generator load từng batch
        for batch_images, _ in obj.load():
            for img in batch_images:
                # Ép cứng size để đồng nhất ma trận và cứu RAM
                img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                pixels_list.append(img_resized.reshape(-1, 3))
            
            processed += len(batch_images)
            print(f" -> Tiến độ: {processed} / {total_images} ảnh...", end='\r')
            
            # Giải phóng RAM ngay lập tức
            del batch_images
            gc.collect()
            
        print("\n") # Xuống dòng cho đẹp log

        if len(pixels_list) == 0:
            raise ValueError("Dataset rỗng hoặc không nạp được ảnh.")
            
        # Biến đổi mảng list thành 1 mảng numpy có shape (Tổng pixel, 3)
        pixels = np.concatenate(pixels_list, axis=0)

        # Áp dụng Sampling nếu số lượng pixel vượt quá max_samples
        if max_samples is not None and len(pixels) > max_samples:
            idx = np.random.choice(len(pixels), min(max_samples, len(pixels)), replace=False)
            pixels = pixels[idx]

        return pixels


class HistogramDistribution(Distribution):
    def __init__(self):
        super().__init__()
        self.step_name = "Histogram Distribution"

    def run(self, obj: Object):
        if isinstance(obj, ImageDataset):
            self.visitImageDataset(obj)
        else:
            self.status = "Failed"
            self.log()

    def visitImageDataset(self, obj: ImageDataset):
        self.dataset_name = obj.folder_path or "ImageDataset"
        try:
            pixels = self._extract_pixels(obj)
            
            plot_histogram(pixels, title_suffix=f"[{self.dataset_name}]")
            self.status = "Success"
        except Exception as e:
            self.status = f"Failed ({str(e)})"
        finally:
            self.log()


class KDEDistribution(Distribution):
    def __init__(self):
        super().__init__()
        self.step_name = "KDE Distribution"

    def run(self, obj: Object):
        if isinstance(obj, ImageDataset):
            self.visitImageDataset(obj)
        else:
            self.status = "Failed"
            self.log()

    def visitImageDataset(self, obj: ImageDataset):
        self.dataset_name = obj.folder_path or "ImageDataset"
        try:
            # max_samples=5000000 như ông set ban đầu để không vẽ quá lâu
            pixels = self._extract_pixels(obj, max_samples=5000000)
            
            plot_kde(pixels, title_suffix=f"[{self.dataset_name}]")
            self.status = "Success"
        except Exception as e:
            self.status = f"Failed ({str(e)})"
        finally:
            self.log()

class PixelDataExtractor(Distribution):
    """
    Dịch vụ chuyên trích xuất dữ liệu ảnh theo Batch để không làm tràn RAM.
    Chỉ gom dữ liệu, không vẽ biểu đồ.
    """
    def __init__(self, target_size=(64, 64)):
        super().__init__()
        self.step_name = "Pixel Data Extraction"
        self.target_size = target_size
        self.images = None
        self.labels = None
        self.pixel_data_all = None

    def visitImageDataset(self, obj: ImageDataset):
        self.dataset_name = obj.folder_path or "ImageDataset"
        print(f"[PROCESS] Đang trích xuất dữ liệu pixel theo Batch (Resize {self.target_size})...")
        images_temp = []
        labels_temp = []
        total_images = len(obj.image_paths)
        processed = 0

        try:
            # Quét bằng Generator an toàn cho RAM
            for batch_images, batch_indices in obj.load():
                for img, path_idx in zip(batch_images, batch_indices):
                    img_resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
                    images_temp.append(img_resized)
                    labels_temp.append(obj._labels[path_idx])

                processed += len(batch_images)
                print(f" -> Đã trích xuất: {processed} / {total_images} ảnh...", end='\r')
                
                del batch_images
                gc.collect()

            print("\n[INFO] Đang chuyển đổi sang định dạng Numpy Array...")
            self.images = np.array(images_temp)
            self.labels = np.array(labels_temp)
            self.pixel_data_all = self.images.reshape(-1, 3)
            self.status = "Success"
            
            print(f"[SUCCESS] Hoàn tất! Tổng số lượng pixel thu được: {self.pixel_data_all.shape[0]:,} pixels")
            
        except Exception as e:
            self.status = f"Failed ({str(e)})"
            print(f"\n[ERROR] {self.status}")
            
        finally:
            self.log() # Tự động gọi hàm log của class cha Distribution
            
        return self

    def run(self, obj: Object):
        if isinstance(obj, ImageDataset):
            self.visitImageDataset(obj)
        else:
            self.status = "Failed (Not an ImageDataset)"
            self.log()