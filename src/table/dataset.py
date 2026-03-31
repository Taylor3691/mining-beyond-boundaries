import pandas as pd
from core import Object, Service
from utils import file

class TableDataset(Object):
    def __init__(self, path: str | None = None):
        self._folder_path = path # file path 
        self._data = None         
        self._features = None    # Lưu X
        self._target = None      # Lưu Y
        self._target_column = None
        self._shape = (0, 0)
        self._columns = []

        if path:
            self.load()

    # Getter
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

    # Method
    def load(self):
        self._data = file.load_table(self._folder_path)
        self._shape = self._data.shape
        self._columns = self._data.columns.tolist()
        return

    def save(self, folder_path: str, file_name: str = "processed_data.csv"):
        file.save_table(path=folder_path, data=self._data, file_name=file_name)
        return

    def set_target(self, target_column: str):
        if self._data is None:
            raise ValueError("Data is not loaded")
        if target_column not in self._columns:
            raise ValueError(f"Couldn't find target column: {target_column}")

        self._target_column = target_column
        self._target = self._data[target_column]
        self._features = self._data.drop(columns=[target_column])
        return self._features, self._target

    def info(self):
        print("--- Metadata of Table Dataset ---")
        print(f"\tFile Path: {self._folder_path if self._folder_path else 'Empty'}")
        print(f"\tDataset Shape: {self._shape} (Rows, Cols)")
        print(f"\tTarget Column: {self._target_column if self._target_column else 'Chưa set'}")
        
        if self._data is not None:
            # Thống kê nhanh dữ liệu thiếu 
            missing_pct = (self._data.isnull().sum().sum() / self._data.size) * 100
            print(f"\tTotal Missing Rate: {missing_pct:.2f}%")
        return

    def clone(self):
        dataset_clone = TableDataset(self._folder_path)
        if self._data is not None:
            dataset_clone.data = self._data.copy()
            if self._target_column:
                dataset_clone.set_target(self._target_column)
        return dataset_clone

    def accept(self, service: Service):
        service.run(self)
        return