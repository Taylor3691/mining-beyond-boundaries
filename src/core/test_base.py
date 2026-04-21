from core import Testing

class DistributionTesting(Testing):
    def __init__(self):
        super().__init__()

    def log(self):
        return super().log()
    
    def visitImageDataset():
        return
        
    def visitTableDataset(self, obj):
        return
        
    def run(self):
        return super().run()
    
    def test():
        return

class StationarityTesting(Testing):
    def __init__(self, column_name: str, alpha: float = 0.05):
        super().__init__()
        self.column_name = column_name
        self.alpha = alpha
        self.step_name = "Base Stationarity Test"
        self.dataset_name = "Unknown"
        self.status = "Pending"
        self.p_value = None
        self.is_stationary = False

    def log(self):
        print(f"Bước xử lý  : {self.step_name}")
        print(f"Thuộc tính  : {self.column_name}")
        if self.p_value is not None:
            print(f"p-value     : {self.p_value:.6f}")
            print(f"Kết luận    : {'DỪNG (Stationary)' if self.is_stationary else 'KHÔNG DỪNG (Non-Stationary)'}")
        print("-" * 50)

    def visitImageDataset(self, obj):
        pass
        
    def test(self):
        pass

    def run(self, obj):
        if hasattr(obj, 'data') and self.column_name in obj.data.columns:
            self.visitTableDataset(obj)
        else:
            self.status = "Failed (Column not found or Invalid Dataset)"
            self.log()