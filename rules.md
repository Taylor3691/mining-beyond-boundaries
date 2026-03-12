# Gợi ý cách đặt tên file
- Nếu công việc liên quan đến phân tích và không xử lý dữ liệu thì `analysis.xx.xx`
- Nếu công việc có xử lý dữ liệu `preprocessing.xx.xx`
- Các hàm trực quan được đặt đúng vào file có tên trùng với kiểu nhóm biểu đồ đó.
- Các hàm trong `/core` là những class abtrasct
- Nếu chưa có file thì tự động tạo file

# Thông tin về các thư viện

## numpy
**Mục đích**: Xử lý mảng đa chiều và tính toán số học hiệu năng cao.
- **array, ndarray**: Tạo và thao tác mảng n chiều
- **reshape, transpose, flatten**: Thay đổi hình dạng mảng
- **mean, std, var, median**: Tính toán thống kê cơ bản
- **dot, matmul**: Nhân ma trận
- **random**: Sinh số ngẫu nhiên (random.rand, random.randn, random.choice)
- **linspace, arange**: Tạo dãy số
- **where, argmax, argmin**: Tìm kiếm và điều kiện
- **unique**: Lấy các giá trị duy nhất
- **concatenate, stack, split**: Nối và chia mảng
- **save, load**: Lưu/đọc mảng từ file .npy

## pandas
**Mục đích**: Xử lý và phân tích dữ liệu dạng bảng (tabular data).
- **DataFrame, Series**: Cấu trúc dữ liệu chính
- **read_csv, read_excel, to_csv**: Đọc/ghi file dữ liệu
- **head, tail, info, describe**: Xem tổng quan dữ liệu
- **isna, isnull, fillna, dropna**: Xử lý missing values
- **drop_duplicates**: Loại bỏ dữ liệu trùng lặp
- **groupby, agg, pivot_table**: Nhóm và tổng hợp dữ liệu
- **merge, concat, join**: Kết hợp nhiều DataFrame
- **apply, map, applymap**: Áp dụng hàm lên dữ liệu
- **loc, iloc**: Truy cập dữ liệu theo label/index
- **value_counts**: Đếm tần suất các giá trị
- **corr**: Tính ma trận tương quan
- **astype**: Chuyển đổi kiểu dữ liệu
- **sort_values, sort_index**: Sắp xếp dữ liệu
- **query, filter**: Lọc dữ liệu theo điều kiện

## matplotlib
**Mục đích**: Trực quan hóa dữ liệu cơ bản.
- **pyplot.plot**: Vẽ biểu đồ đường
- **pyplot.scatter**: Vẽ biểu đồ phân tán
- **pyplot.bar, barh**: Vẽ biểu đồ cột
- **pyplot.hist**: Vẽ histogram
- **pyplot.pie**: Vẽ biểu đồ tròn
- **pyplot.boxplot**: Vẽ boxplot
- **pyplot.subplot, subplots**: Tạo nhiều biểu đồ con
- **pyplot.xlabel, ylabel, title**: Thêm nhãn và tiêu đề
- **pyplot.legend**: Thêm chú thích
- **pyplot.savefig**: Lưu hình ảnh
- **pyplot.figure, figsize**: Thiết lập kích thước hình
- **pyplot.imshow**: Hiển thị hình ảnh

## seaborn
**Mục đích**: Trực quan hóa dữ liệu thống kê đẹp và dễ dùng.
- **heatmap**: Vẽ ma trận nhiệt (correlation matrix)
- **pairplot**: Vẽ scatter plot cho tất cả cặp biến
- **countplot**: Biểu đồ đếm categorical
- **boxplot, violinplot**: So sánh phân phối theo nhóm
- **histplot, kdeplot**: Vẽ histogram và density plot
- **scatterplot**: Scatter plot với hue/size
- **barplot**: Biểu đồ cột với confidence interval
- **lineplot**: Biểu đồ đường với confidence interval
- **catplot**: Categorical plot tổng hợp
- **jointplot**: Kết hợp scatter và histogram
- **regplot, lmplot**: Scatter plot với regression line
- **clustermap**: Heatmap với hierarchical clustering
- **set_theme, set_palette**: Thiết lập theme và màu sắc

## scikit-learn
**Mục đích**: Machine Learning và tiền xử lý dữ liệu.

### Tiền xử lý (preprocessing)
- **StandardScaler**: Chuẩn hóa z-score (mean=0, std=1)
- **MinMaxScaler**: Chuẩn hóa về khoảng [0,1]
- **RobustScaler**: Chuẩn hóa robust với outliers
- **LabelEncoder**: Mã hóa label thành số
- **OneHotEncoder**: Mã hóa one-hot
- **OrdinalEncoder**: Mã hóa ordinal
- **SimpleImputer**: Điền missing values

### Chia dữ liệu (model_selection)
- **train_test_split**: Chia train/test set
- **cross_val_score**: Cross-validation
- **GridSearchCV**: Tìm hyperparameter tốt nhất
- **StratifiedKFold**: K-Fold stratified

### Feature selection
- **SelectKBest**: Chọn k features tốt nhất
- **RFE**: Recursive Feature Elimination
- **VarianceThreshold**: Loại features variance thấp

### Decomposition
- **PCA**: Giảm chiều Principal Component Analysis
- **TruncatedSVD**: SVD cho sparse matrix

### Metrics
- **accuracy_score, precision_score, recall_score, f1_score**: Đánh giá classification
- **confusion_matrix**: Ma trận nhầm lẫn
- **classification_report**: Báo cáo phân loại
- **mean_squared_error, r2_score**: Đánh giá regression
- **silhouette_score**: Đánh giá clustering

### Models
- **LogisticRegression**: Hồi quy logistic
- **LinearRegression**: Hồi quy tuyến tính
- **DecisionTreeClassifier/Regressor**: Cây quyết định
- **RandomForestClassifier/Regressor**: Random Forest
- **KNeighborsClassifier**: K-Nearest Neighbors
- **SVC, SVR**: Support Vector Machine
- **KMeans**: Clustering K-Means
- **DBSCAN**: Clustering density-based

## scipy
**Mục đích**: Tính toán khoa học và kiểm định thống kê.

### scipy.stats - Kiểm định thống kê
- **ttest_ind, ttest_rel**: T-test độc lập/cặp
- **chi2_contingency**: Chi-square test
- **f_oneway**: ANOVA một chiều
- **pearsonr, spearmanr**: Tương quan Pearson/Spearman
- **shapiro**: Kiểm định chuẩn Shapiro-Wilk
- **normaltest**: Kiểm định chuẩn D'Agostino
- **kstest**: Kiểm định Kolmogorov-Smirnov
- **mannwhitneyu**: Mann-Whitney U test
- **wilcoxon**: Wilcoxon signed-rank test
- **kruskal**: Kruskal-Wallis H test
- **levene, bartlett**: Kiểm định phương sai đồng nhất
- **zscore**: Tính z-score

### scipy.spatial
- **distance**: Tính khoảng cách (euclidean, cosine, etc.)

### scipy.cluster
- **hierarchy**: Hierarchical clustering

## statsmodels
**Mục đích**: Mô hình thống kê và kiểm định nâng cao.
- **OLS**: Ordinary Least Squares regression
- **Logit**: Logistic regression với chi tiết thống kê
- **tsa.stattools.adfuller**: Kiểm định ADF (stationarity)
- **stats.durbin_watson**: Kiểm định tự tương quan
- **stats.jarque_bera**: Kiểm định chuẩn Jarque-Bera
- **graphics.gofplots.qqplot**: Vẽ Q-Q plot
- **stats.shapiro**: Shapiro-Wilk test
- **stats.diagnostic**: Các kiểm định chẩn đoán mô hình
- **VIF (variance_inflation_factor)**: Kiểm tra đa cộng tuyến
- **summary()**: Báo cáo chi tiết mô hình

## opencv-python (cv2)
**Mục đích**: Xử lý ảnh và computer vision.
- **imread, imwrite**: Đọc/ghi ảnh
- **imshow, waitKey**: Hiển thị ảnh
- **cvtColor**: Chuyển đổi không gian màu (BGR to RGB, Grayscale)
- **resize**: Thay đổi kích thước ảnh
- **GaussianBlur, medianBlur**: Làm mờ/khử nhiễu
- **Canny**: Phát hiện cạnh
- **threshold, adaptiveThreshold**: Ngưỡng hóa ảnh
- **findContours, drawContours**: Tìm và vẽ contour
- **equalizeHist**: Cân bằng histogram
- **CLAHE**: Adaptive histogram equalization
- **calcHist**: Tính histogram
- **filter2D**: Convolution với kernel tùy chỉnh
- **morphologyEx**: Các phép biến đổi hình thái học (erosion, dilation, opening, closing)
- **warpAffine, warpPerspective**: Biến đổi hình học

## Pillow (PIL)
**Mục đích**: Xử lý ảnh cơ bản.
- **Image.open, Image.save**: Đọc/ghi ảnh
- **Image.resize**: Thay đổi kích thước
- **Image.rotate**: Xoay ảnh
- **Image.crop**: Cắt ảnh
- **Image.convert**: Chuyển đổi mode (RGB, L, RGBA)
- **Image.filter**: Áp dụng filter
- **ImageEnhance**: Điều chỉnh brightness, contrast, sharpness
- **ImageDraw**: Vẽ lên ảnh
- **Image.thumbnail**: Tạo thumbnail

## nltk
**Mục đích**: Xử lý ngôn ngữ tự nhiên.
- **word_tokenize, sent_tokenize**: Tách từ/câu
- **stopwords**: Danh sách stopwords
- **PorterStemmer, SnowballStemmer**: Stemming
- **WordNetLemmatizer**: Lemmatization
- **pos_tag**: Part-of-speech tagging
- **FreqDist**: Phân phối tần suất từ
- **ngrams**: Tạo n-grams
- **concordance**: Tìm ngữ cảnh của từ
- **sentiment.vader**: Sentiment analysis

## spacy
**Mục đích**: Xử lý ngôn ngữ tự nhiên hiệu năng cao.
- **nlp()**: Pipeline xử lý text
- **doc.ents**: Named Entity Recognition (NER)
- **token.lemma_**: Lemma của từ
- **token.pos_**: Part-of-speech tag
- **token.dep_**: Dependency parsing
- **token.is_stop**: Kiểm tra stopword
- **doc.similarity()**: Tính độ tương đồng văn bản
- **Matcher**: Pattern matching
- **displacy**: Visualization dependency/NER

## missingno
**Mục đích**: Trực quan hóa missing data.
- **matrix**: Ma trận missing values
- **bar**: Biểu đồ cột missing values
- **heatmap**: Heatmap correlation missing values
- **dendrogram**: Dendrogram clustering missing values

## imbalanced-learn
**Mục đích**: Xử lý dữ liệu mất cân bằng (imbalanced data).

### Over-sampling
- **RandomOverSampler**: Over-sampling ngẫu nhiên
- **SMOTE**: Synthetic Minority Over-sampling
- **ADASYN**: Adaptive Synthetic Sampling
- **BorderlineSMOTE**: SMOTE cho borderline samples

### Under-sampling
- **RandomUnderSampler**: Under-sampling ngẫu nhiên
- **TomekLinks**: Loại bỏ Tomek links
- **EditedNearestNeighbours**: ENN under-sampling
- **NearMiss**: Near Miss under-sampling

### Combination
- **SMOTEENN**: Kết hợp SMOTE và ENN
- **SMOTETomek**: Kết hợp SMOTE và Tomek Links

### Pipeline
- **Pipeline**: Pipeline riêng cho imbalanced-learn

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