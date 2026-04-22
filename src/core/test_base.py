from core import Testing

class DistributionTesting(Testing):
    """Lớp cơ sở cho các kiểm định phân phối dữ liệu."""

    def __init__(self):
        """
        Khởi tạo đối tượng kiểm định phân phối.

        Input:
            Không có.

        Output:
            None.
        """
        super().__init__()

    def log(self):
        """
        In kết quả kiểm định ra màn hình.

        Input:
            Không có.

        Output:
            None (in ra màn hình).
        """
        return super().log()
    
    def visitImageDataset():
        """
        Xử lý kiểm định cho đối tượng ImageDataset.

        Input:
            Không có (tham số do lớp con định nghĩa).

        Output:
            None.
        """
        return
        
    def visitTableDataset(self, obj):
        """
        Xử lý kiểm định cho đối tượng TableDataset.

        Input:
            obj: Đối tượng TableDataset chứa dữ liệu cần kiểm định.

        Output:
            None.
        """
        return
        
    def run(self):
        """
        Thực thi quy trình kiểm định.

        Input:
            Không có.

        Output:
            None.
        """
        return super().run()
    
    def test():
        """
        Thực hiện tính toán kiểm định thống kê.

        Input:
            Không có (tham số do lớp con định nghĩa).

        Output:
            Kết quả kiểm định.
        """
        return

class StationarityTesting(Testing):
    """Lớp cơ sở cho các kiểm định tính dừng của chuỗi thời gian."""

    def __init__(self, column_name: str, alpha: float = 0.05):
        """
        Khởi tạo bộ kiểm định tính dừng.

        Input:
            column_name: Tên cột dữ liệu cần kiểm định.
            alpha: Mức ý nghĩa thống kê (mặc định 0.05).

        Output:
            None.
        """
        super().__init__()
        self.column_name = column_name
        self.alpha = alpha
        self.step_name = "Base Stationarity Test"
        self.dataset_name = "Unknown"
        self.status = "Pending"
        self.p_value = None
        self.is_stationary = False

    def log(self):
        """
        In kết quả kiểm định tính dừng ra màn hình.

        Input:
            Không có.

        Output:
            None (in ra màn hình).
        """
        print(f"Bước xử lý  : {self.step_name}")
        print(f"Thuộc tính  : {self.column_name}")
        if self.p_value is not None:
            print(f"p-value     : {self.p_value:.6f}")
            print(f"Kết luận    : {'DỪNG (Stationary)' if self.is_stationary else 'KHÔNG DỪNG (Non-Stationary)'}")
        print("-" * 50)

    def visitImageDataset(self, obj):
        """
        Không hỗ trợ dữ liệu hình ảnh cho kiểm định tính dừng.

        Input:
            obj: Đối tượng ImageDataset.

        Output:
            None.
        """
        pass
        
    def test(self):
        """
        Thực hiện tính toán kiểm định tính dừng (lớp con phải override).

        Input:
            Không có.

        Output:
            None.
        """
        pass

    def run(self, obj):
        """
        Điểm vào thực thi kiểm định: kiểm tra cột dữ liệu và gọi visitTableDataset.

        Input:
            obj: Đối tượng dataset chứa thuộc tính data (DataFrame).

        Output:
            None.
        """
        if hasattr(obj, 'data') and self.column_name in obj.data.columns:
            self.visitTableDataset(obj)
        else:
            self.status = "Failed (Column not found or Invalid Dataset)"
            self.log()