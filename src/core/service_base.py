from abc import ABC, abstractmethod
from core import Object

class Pipeline(ABC):
    @abstractmethod
    def run(self):
        """
        Thực thi quy trình xử lý chính.

        Input:
            Không có.

        Output:
            None.
        """
        pass
    
    @abstractmethod
    def log(self):
        """
        In thông tin trạng thái và kết quả xử lý ra màn hình.

        Input:
            Không có.

        Output:
            None (in ra màn hình).
        """
        pass
    
class SubPipeline(Pipeline):
    def __init__(self, dataset: Object):
        """
        Khởi tạo SubPipeline chứa danh sách các Service cần chạy tuần tự.

        Input:
            dataset: Đối tượng dữ liệu (Object) để áp dụng các Service.

        Output:
            None.
        """
        self._list = []
        self._children = []
        self._dataset = dataset
        return
    
    def addService(self, sub: Pipeline):
        """
        Thêm một Service vào danh sách pipeline.

        Input:
            sub: Đối tượng Pipeline (Service) cần thêm.

        Output:
            None.
        """
        self._list.append(sub)
        return

    def run(self):
        """
        Thực thi tuần tự tất cả các Service trong pipeline trên dataset.

        Input:
            Không có.

        Output:
            None.
        """
        for service in self._list:
            self._dataset.accept(service)
        return

class Service(Pipeline):
    """
    @abstractmethod
    def visitTabbleDataset():
        pass
    """

    @abstractmethod
    def visitImageDataset():
        """
        Xử lý cụ thể cho đối tượng ImageDataset.

        Input:
            Không có (tham số do lớp con định nghĩa).

        Output:
            None.
        """
        pass

class Visualization(Service):
    pass

class Preprocessing(Service):
    @abstractmethod
    def fit():
        """
        Tính toán và lưu trữ các tham số thống kê từ dữ liệu.

        Input:
            Không có (tham số do lớp con định nghĩa).

        Output:
            None.
        """
        pass
    
    @abstractmethod
    def transform():
        """
        Áp dụng phép biến đổi lên dữ liệu dựa trên các tham số đã tính.

        Input:
            Không có (tham số do lớp con định nghĩa).

        Output:
            Dữ liệu sau khi biến đổi.
        """
        pass

    @abstractmethod
    def fit_transform():
        """
        Kết hợp fit và transform trong một bước duy nhất.

        Input:
            Không có (tham số do lớp con định nghĩa).

        Output:
            Dữ liệu sau khi biến đổi.
        """
        pass

class Testing(Service):    
    @abstractmethod
    def test():
        """
        Thực hiện kiểm định thống kê trên dữ liệu.

        Input:
            Không có (tham số do lớp con định nghĩa).

        Output:
            Kết quả kiểm định.
        """
        pass