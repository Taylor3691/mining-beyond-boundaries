import pandas as pd
from core import Object, Service
from utils import file  

class TimeSeriesDataset(Object):
    """Lớp quản lý dữ liệu chuỗi thời gian (Time-series Dataset)."""

    def __init__(self, path: str | None = None, time_column: str = 'Date'):
        """
        Khởi tạo tập dữ liệu chuỗi thời gian.

        Input:
            path: Đường dẫn tới file dữ liệu.
            time_column: Tên cột chứa trục thời gian (mặc định 'Date').
        """
        self._folder_path = path 
        self._time_column = time_column # Cột chứa trục thời gian
        
        self._data = None         
        self._features = None    
        self._target = None      
        self._target_column = None
        self._shape = (0, 0)
        self._columns = []

        if path:
            self.load()

    @property
    def data(self):
        return self._data
        
    @property
    def features(self):
        return self._features
        
    @property
    def target(self):
        return self._target

    @property
    def shape(self):
        return self._shape

    @property
    def columns(self):
        return self._columns
    
    @data.setter
    def data(self, value: pd.DataFrame):
        if value is None or value.empty:
            raise ValueError("Dữ liệu không được trống.")
        self._data = value
        self._shape = value.shape
        self._columns = value.columns.tolist()

    def load(self):
        """
        Nạp dữ liệu từ file CSV, ép kiểu cột thời gian và sắp xếp theo thời gian.

        Input:
            Không có (sử dụng _folder_path đã khởi tạo).

        Output:
            None (cập nhật _data, _columns, _shape nội bộ).
        """
        self._data = file.load_table(self._folder_path)
        self._columns = self._data.columns.tolist()
        
        if self._time_column in self._columns:
            # 1. Ép kiểu về Datetime chuẩn của Pandas
            self._data[self._time_column] = pd.to_datetime(self._data[self._time_column])
            
            # 2. Sắp xếp từ quá khứ đến hiện tại
            self._data = self._data.sort_values(by=self._time_column).reset_index(drop=True)
            print(f"[Time Series Mode] Đã nạp và sắp xếp dữ liệu theo '{self._time_column}'.")
        else:
            raise ValueError(f"Không tìm thấy cột thời gian '{self._time_column}' trong tập dữ liệu.")
            
        self._shape = self._data.shape
        return

    def save(self, folder_path: str, file_name: str = "processed_timeseries.csv"):
        """Lưu dữ liệu hiện tại vào file."""
        file.save_table(path=folder_path, data=self._data, file_name=file_name)
        return

    def set_target(self, target_column: str):
        """Thiết lập biến mục tiêu (target) cho các mô hình dự báo."""
        if self._data is None:
            raise ValueError("Data is not loaded")
        if target_column not in self._columns:
            raise ValueError(f"Couldn't find target column: {target_column}")

        self._target_column = target_column
        self._target = self._data[target_column]
        self._features = self._data.drop(columns=[target_column])
        return self._features, self._target

    def temporal_split(self, train_ratio: float = 0.8):
        """
        Chia dữ liệu thành Train/Test theo thứ tự thời gian (không xáo trộn).

        Input:
            train_ratio: Tỉ lệ dữ liệu huấn luyện (mặc định 0.8).

        Output:
            Tuple (train_df, test_df): Hai DataFrame chia theo thời gian.
        """
        if self._data is None:
            raise ValueError("Dataset is empty. Call load() first.")
            
        split_idx = int(len(self._data) * train_ratio)
        train_df = self._data.iloc[:split_idx].copy()
        test_df = self._data.iloc[split_idx:].copy()
        
        print(f"Temporal Split: Train ({len(train_df)} dòng) | Test ({len(test_df)} dòng)")
        return train_df, test_df

    def info(self):
        """
        In thông tin metadata tổng quan của tập dữ liệu chuỗi thời gian.

        Input:
            Không có.

        Output:
            None (in ra màn hình).
        """
        print("--- Metadata of Time Series Dataset ---")
        print(f"\tFile Path: {self._folder_path if self._folder_path else 'Empty'}")
        print(f"\tDataset Shape: {self._shape} (Rows, Cols)")
        print(f"\tTime Column: {self._time_column}")
        
        if self._data is not None and self._time_column in self._columns:
            start_date = self._data[self._time_column].min().date()
            end_date = self._data[self._time_column].max().date()
            print(f"\tTime Range: Từ {start_date} đến {end_date}")
            
            missing_pct = (self._data.isnull().sum().sum() / self._data.size) * 100
            print(f"\tTotal Missing Rate: {missing_pct:.2f}%")
        return

    def clone(self):
        """
        Tạo bản sao độc lập của tập dữ liệu chuỗi thời gian.

        Input:
            Không có.

        Output:
            TimeSeriesDataset: Đối tượng bản sao với dữ liệu được deep copy.
        """
        dataset_clone = TimeSeriesDataset(self._folder_path, self._time_column)
        if self._data is not None:
            dataset_clone.data = self._data.copy()
            if self._target_column:
                dataset_clone.set_target(self._target_column)
        return dataset_clone

    def accept(self, service: Service):
        """
        Chấp nhận một Service (Visitor) để thực thi tác vụ trên dataset.

        Input:
            service: Đối tượng Service cần thực thi.

        Output:
            None.
        """
        service.run(self)
        return
