import numpy as np
from core.service_base import Visualization
from core.data_base import Object
from image.dataset import ImageDataset
from visualization.distribution import plot_histogram, plot_kde

class Distribution(Visualization):
    def __init__(self):
        self.step_name = "Base Analysis"
        self.dataset_name = "Unknown"
        self.status = "Pending"

    def log(self):
        print(f"Bước xử lý : {self.step_name}")
        print(f"Tập dữ liệu: {self.dataset_name}")
        print(f"Trạng thái : {self.status}")
        print("="*50)

    def _extract_pixels(self, obj: ImageDataset, max_samples=200000):
        """ Task 7: Lấy thông tin pixel, giữ cấu trúc 3 kênh màu và áp dụng Sampling """
        images, _ = obj.images
        if len(images) == 0:
            raise ValueError("Dataset rỗng.")
            
        # Biến đổi mảng list(N, H, W, 3) thành 1 mảng numpy 2D (Tổng pixel, 3)
        pixels = np.concatenate([img.reshape(-1, 3) for img in images])

        if max_samples is not None and len(pixels) > max_samples:
            idx = np.random.choice(len(pixels), min(max_samples, len(pixels)), replace=False)
            pixels = pixels[idx]

        return pixels


class HistogramDistribution(Distribution):
    def __init__(self):
        super().__init__()
        self.step_name = "Analysis Histogram Distribution"

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
        self.step_name = "Analysis KDE Distribution"

    def run(self, obj: Object):
        if isinstance(obj, ImageDataset):
            self.visitImageDataset(obj)
        else:
            self.status = "Failed"
            self.log()

    def visitImageDataset(self, obj: ImageDataset):
        self.dataset_name = obj.folder_path or "ImageDataset"
        try:
            pixels = self._extract_pixels(obj, max_samples=5000000)
            
            plot_kde(pixels, title_suffix=f"[{self.dataset_name}]")
            self.status = "Success"
        except Exception as e:
            self.status = f"Failed ({str(e)})"
        finally:
            self.log()