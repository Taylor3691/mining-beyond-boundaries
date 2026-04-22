import cv2
import gc
import numpy as np
from core.service_base import Visualization
from core.data_base import Object
from image.dataset import ImageDataset
from visualization.distribution import plot_histogram, plot_kde


class Distribution(Visualization):
    def __init__(self):
        """
        Khởi tạo lớp phân tích cơ bản.
        """
        self.step_name = "Base Analysis"
        self.dataset_name = "Unknown"
        self.status = "Pending"

    def log(self):
        """
        In thông tin trạng thái hiện tại của quá trình phân tích.
        """
        print(f"Bước xử lý : {self.step_name}")
        print(f"Tập dữ liệu: {self.dataset_name}")
        print(f"Trạng thái : {self.status}")

    # Thêm tham số target_size để ép size ảnh gọn nhẹ ngay từ lúc nạp
    def _extract_pixels(self, obj: ImageDataset, max_samples=200000, target_size=(64, 64)):
        """
        Trích xuất dữ liệu pixel từ tập dữ liệu ảnh sau khi resize.

        Input:
            obj: Đối tượng ImageDataset chứa danh sách đường dẫn ảnh.
            max_samples: Số lượng pixel tối đa lấy ra để phân tích (mặc định 200,000).
            target_size: Kích thước ảnh sau khi resize (mặc định 64x64).

        Output:
            Mảng numpy chứa dữ liệu pixel (số lượng mẫu, 3 kênh màu).
        """
        print(f"[PROCESS] Đang trích xuất pixel theo Batch (Resize về {target_size})...")
        pixels_list = []
        total_images = len(obj.image_paths)
        processed = 0

        # Load theo batch để tránh tràn bộ nhớ
        for batch_images, _ in obj.load():
            for img in batch_images:
                # Resize để đồng nhất ma trận và tiết kiệm RAM
                img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                pixels_list.append(img_resized.reshape(-1, 3))
            
            processed += len(batch_images)
            print(f" -> Tiến độ: {processed} / {total_images} ảnh...", end='\r')
            
            # Giải phóng RAM ngay lập tức
            del batch_images
            gc.collect()
            
        print("\n")

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
        """
        Khởi tạo lớp phân tích phân phối thông qua biểu đồ Histogram.
        """
        super().__init__()
        self.step_name = "Histogram Distribution"

    def run(self, obj: Object):
        """
        Thực thi quá trình phân tích Histogram cho đối tượng dữ liệu.

        Input:
            obj: Đối tượng cần phân tích (thường là ImageDataset).
        """
        if isinstance(obj, ImageDataset):
            self.visitImageDataset(obj)
        else:
            self.status = "Failed"
            self.log()

    def visitImageDataset(self, obj: ImageDataset):
        """
        Xử lý cụ thể cho tập dữ liệu ImageDataset để vẽ Histogram.

        Input:
            obj: Đối tượng ImageDataset cụ thể.
        """
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
        """
        Khởi tạo lớp phân tích phân phối thông qua biểu đồ mật độ KDE.
        """
        super().__init__()
        self.step_name = "KDE Distribution"

    def run(self, obj: Object):
        """
        Thực thi quá trình phân tích KDE cho đối tượng dữ liệu.

        Input:
            obj: Đối tượng cần phân tích.
        """
        if isinstance(obj, ImageDataset):
            self.visitImageDataset(obj)
        else:
            self.status = "Failed"
            self.log()

    def visitImageDataset(self, obj: ImageDataset):
        """
        Xử lý cụ thể cho tập dữ liệu ImageDataset để vẽ KDE.

        Input:
            obj: Đối tượng ImageDataset cụ thể.
        """
        self.dataset_name = obj.folder_path or "ImageDataset"
        try:
            # Giới hạn số lượng mẫu để tối ưu thời gian vẽ
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
        """
        Khởi tạo lớp trích xuất dữ liệu pixel.

        Input:
            target_size: Kích thước ảnh để resize trước khi trích xuất.
        """
        super().__init__()
        self.step_name = "Pixel Data Extraction"
        self.target_size = target_size
        self.images = None
        self.labels = None
        self.pixel_data_all = None

    def visitImageDataset(self, obj: ImageDataset):
        """
        Thực hiện quét batch và trích xuất dữ liệu pixel, nhãn từ dataset.

        Input:
            obj: Đối tượng ImageDataset.

        Output:
            Trả về chính đối tượng PixelDataExtractor sau khi đã nạp dữ liệu.
        """
        self.dataset_name = obj.folder_path or "ImageDataset"
        print(f"[PROCESS] Đang trích xuất dữ liệu pixel theo Batch (Resize {self.target_size})...")
        images_temp = []
        labels_temp = []
        total_images = len(obj.image_paths)
        processed = 0

        try:
            # Sử dụng Generator để xử lý dữ liệu lớn
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
            self.log()
            
        return self

    def run(self, obj: Object):
        """
        Chạy quy trình trích xuất dữ liệu.

        Input:
            obj: Đối tượng cần trích xuất.
        """
        if isinstance(obj, ImageDataset):
            self.visitImageDataset(obj)
        else:
            self.status = "Failed (Not an ImageDataset)"
            self.log()