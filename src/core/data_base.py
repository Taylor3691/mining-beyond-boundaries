from __future__ import annotations
from abc import ABC, abstractmethod

class Object(ABC):
    @abstractmethod
    def load(path: str):
        """
        Nạp dữ liệu từ nguồn bên ngoài vào đối tượng.

        Input:
            path: Đường dẫn tới file hoặc thư mục dữ liệu.

        Output:
            None (cập nhật trực tiếp trạng thái nội bộ của đối tượng).
        """
        pass

    @abstractmethod
    def save(path: str):
        """
        Lưu dữ liệu hiện tại ra file hoặc thư mục.

        Input:
            path: Đường dẫn đích để lưu dữ liệu.

        Output:
            None.
        """
        pass

    @abstractmethod
    def info():
        """
        In thông tin metadata tổng quan của đối tượng dữ liệu.

        Input:
            Không có.

        Output:
            None (in ra màn hình).
        """
        pass

    @abstractmethod
    def clone():
        """
        Tạo một bản sao độc lập (deep copy) của đối tượng hiện tại.

        Input:
            Không có.

        Output:
            Đối tượng Object mới là bản sao của đối tượng gốc.
        """
        pass
    
    @abstractmethod
    def accept():
        """
        Chấp nhận một Service (Visitor) để thực thi tác vụ trên đối tượng dữ liệu.

        Input:
            Không có (tham số cụ thể do lớp con định nghĩa).

        Output:
            None.
        """
        pass