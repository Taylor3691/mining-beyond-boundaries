# Đồ án: Tiền xử lý dữ liệu 

## Mục tiêu đồ án:
- Thực hiện tiền xử lý với nhiều kiểu khác nhau như ảnh, bảng, văn bản và dữ liệu chuỗi thời gian.
- Giải thích dữ liệu bằng các công cụ thống kê, so sánh có kiểm soát và thiết kế pipeline có thể sử dụng cho nhiều loại dữ liệu.

## Yêu cầu chung đồ án
- Thực hiện chọn dataset và tiền xử lý trên các tập dữ liệu, các dạng dữ liệu được tùy ý chọn 3/4 đã cho, thực hiện các phương pháp tiền xử lý đã nêu trong đồ án.
- Thực hiện tạo repository quản lí và phân chia mã nguồn hợp lý, code clean. Được sử dụng AI  <=30%.
- Thực hiện viết báo cáo bằng Notebook, ngoài ra tổng hợp lại và ghi vào trong một PDF riêng. 
- Sử dụng mã nguồn hoàn toàn bằng Python hoặc Jupyter Notebook. Đảm bảo mã nguồn chạy được và phải Restart Kernel.
- Không được trễ Deadline.
- Chỉ được sử dụng các thư viện giới hạn.
- Mỗi kỹ thuật tiền xử lý đều phải có 3 phần như sau:
  - Ô markdown giải thích lý thuyết và công thức toán (nếu có).
  - Ô code cài đặt 
  - Ô markdown phân tích kết quả
- Thực hiện thiết kế hệ thống để linh hoạt cho việc tiền xử lý nhiều kiểu dữ liệu khác nhau.

## Timeline dự kiến
- Tuần 1 (11/3 - 18/3): Thực hiện xong nội dung Phần 1 và Phần 2 + Viết báo cáo lý thuyết các phương pháp.
- Tuần 2 (19/3 - 26/3): Thực hiện xong Phần nội dung 3/4 và viết báo cáo phần Discussion + thực hiện các phần nâng cao nếu có thời gian.
- Tuần 3 (27/3 - 3/4): Hoàn thiện đồ án và chuẩn bị thuyết trình giữa kì nếu có.

## Các công cụ dự kiến sử dụng
- Quản lý mã nguồn: Git, Github
- Quản lý dự án và theo dõi tiến độ: Google Sheet
- Họp: Google Meet
- Ngôn ngữ lập trình: Python


## Tiền xử lý dữ liệu
### Phần 1: Dữ liệu dạng ảnh
- Yêu cầu: >5 lớp và >5000 image
- Phân tích thống kê dữ liệu
  - **Tính và trực quan hóa phân phối các giá trị pixel** trên toàn tập (Histogram, KDE, một số các độ đo phân bố khác).
  - **Phân tích cân bằng lớp ?** Thế nào là cân bằng lớp ? Tính tỉ lệ mỗi lớp ? Có lớp nào chiếm tỉ lệ 3x so với lớp thấp nhất hay không ? Ngoài mất cân bằng về số lượng ra, dữ liệu có thể mất cân bằng về các yếu tố khác không ?
  - **Phát hiện ảnh trùng bằng hàm băm pHash ?** Thực hiện đo đặc và xử lý các ảnh trùng hoặc gần trùng (ảnh nào được giữ, ảnh nào không được giữ hoặc kết hợp 2 ảnh, yếu tố gì ảnh hưởng đến quyết định giữ/bỏ ?) - Thực hiện so sánh với một phương pháp khác cũng dùng để loại bỏ ảnh trùng, so sánh ưu nhược điểm 2 phương pháp ? Từ đó đưa ra kết luận về phương pháp pHash.
  - **Phân tích độ sáng và độ tương phản ?** Tính mean intensity và standard deviation ? Tại sao lại phải phân tích, người ta quan tâm điều gì khi lại thực hiện bước này trên dữ liệu ? Từ kết quả thống kê đo được, vấn đề dữ liệu qua cái nhìn thống kê.
- Các kỹ thuật xử lý và phân tích tác động 
  - **Thay đổi kích thước ảnh** lên 32x32, 64x64, 128x128, 256x256 và sử dụng các độ đo SSIM và PSNR để định lượng độ mất mát thông tin ? Lý thuyết về các độ đo này, độ đo này đo sự mất mát dựa vào gì ? vẽ đường PSNR và biện luận về size mình chọn.
  - **Chuyển đổi không gian màu**: Chọn ra 3 không gian màu RGB, GrayScale, HSV, LAB (Còn có không gian màu nào khác không ?) - Lý thuyết chuyển đổi không gian màu từ A sang B ? Tính Phương sai giải thích (là gì ?) theo PCA với k = 50 thành phần (tại sao lại là k =  50, k > 50 và k < 50 có gì khác không ?). Biện luận để chọn không gian màu phù hợp ?
  - **Chuẩn hóa**: chuẩn hóa theo 4 phương pháp Max-Min [0,1] , Max-Min[-1,1], Z-score toàn tập (Cách tính ?), Z-score từng kênh (khác gì so với toàn tập ?). Dùng kiểm định Kolmogorov-Smirnov đánh giá sự khác biệt phân phối trước và sau chuẩn hóa ? (Kiểm định này dựa vào gì để kiểm tra sự thay đổi ? Dùng trong các lĩnh vực gì ? Phát biểu như nào ?) Báo cáo p-value và diễn giải ý nghĩa thống kê  (Phát biểu là gì ? Cái gì cần được kiểm chứng ? Kiểm chứng nhằm mục tiêu là gì ? Kiểm xong thì làm gì tiếp theo ?)
  - **Làm giàu dữ liệu**: Cài pipeline cho phép thực hiện 5 phép biến đổi (phép biến đổi tuyến tính, phép biến đổi phi tuyến, cắt ngẫu nhiên ?, thêm nhiễu ?, xoay ảnh). Đánh giá sự tác động của các phương pháp đến với phân phối đặc trưng (đặc trưng mà có phân phối ????), sử dụng phương pháp t-SNE để visualize của tập gốc với tập đã augument.
  - Phần nâng cao (Chưa quyết định)

### Phần 2: Dữ liệu dạng bảng
- Yêu cầu: >10 thuộc tính và >10k record, phải có missing value >5% trên ít nhất 1 thuộc tính
- Phân tích thống kê dữ liệu:
  - **Kiểm tra phân phối**: Với mỗi thuộc tính kiểu số, thực hiện kiểm định Shapiro-Wilk hoặc D'Agostino-Pearson dựa vào n (Tại sao >5000 thì xài khác, <5000 thì xài khác ?). Phân loại phân phối xem thuộc tính này có phải phân phối chuẩn hay không dựa trên p-value (Phát biểu giả thuyết như nào, biện luận để chấp nhận hoặc bác bỏ giả thuyết ? Có giá trị thống kê là sao ? Có giá trị ở đây bản chất là gì ?), dựa trên kết quả này để lựa chọn phương pháp chuẩn hóa phù hợp.
  - **Phân tích tương quan đa biến**: Vẽ heatmap tương quan Pearson và Spearman (???), xác định các cặp thuộc tính có khả năng đa cộng tuyến (Đa cộng tuyến là gì ? Làm cách nào để xác định đa cộng tuyến ?), đề xuất một số phương pháp xử lý (ưu tiên phương pháp xử lý đặc biệt), biện luận và chọn ra biện pháp tốt nhất.
  - **Phân tích giá trị thiếu**: Trực quan hóa ma trận thiếu dữ liệu thông qua thư viện. Kiểm định giả thuyết MCAR bằng Little's MCAR Test (giải thích tìm hiểu lý thuyết nó là gì ? Cách làm ?). Phân loại cơ chế thiếu dữ liệu (MCAR/MAR/MNAR) và giải thích (đặc điểm từng loại ? Khác nhau và giống nhau ? Ưu nhược điểm ? Giải thích cái gì ? Cơ chế thiếu dữ liệu là gì ?)

- Các kĩ thuật và đánh giá định lượng
  - **Xử lý giá trị thiếu có kiểm soát**: Cài đặt 5 chiến lược và tính RMSE điền khuyết để đánh giá độ chính xác. Trình bày bảng và lựa chọn chiến lược tốt nhất có lý giải (Ở một bên khác, có phải lúc nào cũng điền 1 kiểu hay không hay kết hợp tùy trường hợp ? Nếu xét nhiều trường hợp, độ phức tạp tính toán sẽ như thế nào ?)
  - **Phát hiện và xử lý outlier bằng nhiều kĩ thuật**: Cài đặt và so sánh 4 phương pháp, đánh giá tỉ lệ phát hiện ngoại lai và sự chồng chéo giữa các tập ngoại lai (tại sao phải so sánh sự overlap?). Đánh giá việc tác động ngoại lai bằng KS test tới phân phối (Việc loại bỏ những điểm có giúp ta nhận ra gì tốt hơn không hay gây nhầm lẫn ?) 

  - **Chuẩn hóa dữ liệu có kiểm định**: Áp dụng các phương pháp chuẩn hóa, sử dụng kiểm định Levene's test để đánh giá sự đồng nhất phương sai (Tại sao phải dựa vào phương sai để đánh giá đồng nhất ? Đánh giá sự đồng nhất thì có mang lại ý nghĩa gì hay đóng góp gì trong việc chọn phương pháp phù hợp). Trực quan hóa bằng Violin Plot

  - **Mã hóa biến phân loại nâng cao**: Áp dụng các phương pháp đề cập, đo phương sai giải thích (là gì? nó có ý nghĩa gì ? Cách tính ?), làm sao mà phương sai giải thích đa cộng tuyến phát sinh ?

  - **Lựa chọn và giảm chiều đặc trưng**: Áp dụng phương pháp lọc/giảm chiều, báo cáo cross-validation F1-score. Vẽ biểu đồ hiệu năng theo số đặc trưng.

  ### Phần 3 (Chưa quyết định)


