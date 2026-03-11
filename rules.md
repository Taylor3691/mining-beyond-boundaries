# Gợi ý cách đặt tên file
- Nếu công việc liên quan đến phân tích và không xử lý dữ liệu thì `analysis.xx.xx`
- Nếu công việc có xử lý dữ liệu `preprocessing.xx.xx`
- Các hàm trực quan được đặt đúng vào file có tên trùng với kiểu nhóm biểu đồ đó.
- Các hàm trong `/core` là những class abtrasct
- Nếu chưa có file thì tự động tạo file

# Phân chia trách nhiệm các file
- Các hàm trong `/core` là những class abtrasct khuôn mẫu cho các lớp con.
- các folder `/image`, `/table` là các folder tương ứng cho các kiểu dữ liệu và nó là controller/logic cho các hàm xử lý kiểu dữ liệu đó.
- Các nhóm biểu đồ được chia theo các file khác nhau, tùy vào dạng biểu đồ có ý nghĩa gì thì vào file tương ứng.
- `/statistic` là folder chứa các hàm thống kê, trong đồ án nó thường là các hàm kiểm định.
- Các file notebooks chỉ được dùng để báo cáo và gọi hàm, không được phép có xử lý logic.
- Tất cả các hoạt động test hoặc processing data phải được ghi log và đưa vào thư mục riêng.

# Quy tắc khi commit và push
- Commit phải có nghĩa, không được commit rác.
- Mỗi lần xong thì phải push, push lên nhánh riêng
- Mỗi commit đều phải có từ đặc biệt đi kèm phía trước gợi nhớ, ví dụ:
  - feature: add create function
  - docs: add A's weely docs
  - fix: fix bug wrong name
  - chore/refactor: create abstract class
- Commit lên thì code phải chạy được.