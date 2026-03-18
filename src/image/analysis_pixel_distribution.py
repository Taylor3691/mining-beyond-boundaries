import numpy as np
from core.service_base import Visualization
from core.data_base import Object
from image.dataset import ImageDataset
from visualization.distribution import plot_histogram, plot_kde

class Distribution(Visualization):
    def __init__(self):
        self.step_name = "Unknown Distribution"
        self.dataset_name = "Unknown Dataset"
        self.status = "Pending"

    def log(self):
        print(f"Bước xử lý: {self.step_name}")
        print(f"Tập dữ liệu: {self.dataset_name}")
        print(f"Trạng thái: {self.status}")

    def _extract_pixels(self, obj: ImageDataset):
        """ Task 7: Lấy thông tin pixel với 3 kênh màu """
        images, _ = obj.images
        if len(images) == 0:
            raise ValueError("Dataset rỗng.")
            
        # Biến đổi mảng list(N, H, W, 3) thành 1 mảng numpy có shape: (Tổng pixel, 3)
        return np.concatenate([img.reshape(-1, 3) for img in images])


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
            # 1. Trích xuất mảng pixel (Tổng số pixel, 3)
            pixels = self._extract_pixels(obj)
            
            # 2. Gọi hàm thủ tục vẽ Histogram
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
            # 1. Trích xuất mảng pixel (Tổng số pixel, 3)
            pixels = self._extract_pixels(obj)
            
            # 2. Gọi hàm thủ tục vẽ đường cong KDE
            plot_kde(pixels, title_suffix=f"[{self.dataset_name}]")
            
            self.status = "Success"
        except Exception as e:
            self.status = f"Failed ({str(e)})"
        finally:
            self.log()