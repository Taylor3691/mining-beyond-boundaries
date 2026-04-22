import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from IPython.display import display, Markdown

class ImbalanceHandlerBonus:
    def __init__(self, X, y, test_size=0.2, random_state=42):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        
        # Biến lưu trữ tập dữ liệu
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        
        # Biến lưu kết quả
        self.results = []
        
    def run(self):
        # 1. Kiểm tra mức độ mất cân bằng
        counts = self.y.value_counts()
        print("Phân phối nhãn gốc:")
        print(counts)
        if counts.max() / counts.min() < 1.5:
            print("Dữ liệu này không quá mất cân bằng, hiệu quả của Resampling có thể không rõ rệt.")

        # 2. CHIA TẬP TRAIN/TEST (BƯỚC QUAN TRỌNG NHẤT: BẮT BUỘC LÀM TRƯỚC)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state, stratify=self.y
        )
        
        # 3. Khởi tạo các phương pháp Resampling
        samplers = {
            "Original (Baseline)": None,
            "Random Under-sampling (RUS)": RandomUnderSampler(random_state=self.random_state),
            "SMOTE": SMOTE(random_state=self.random_state),
            "ADASYN": ADASYN(random_state=self.random_state)
        }
        
        # 4. Huấn luyện và Đánh giá
        print("\nĐang chạy thử nghiệm các chiến lược Resampling...")
        for name, sampler in samplers.items():
            if sampler is None:
                X_res, y_res = self.X_train, self.y_train
            else:
                X_res, y_res = sampler.fit_resample(self.X_train, self.y_train)
                
            # Dùng Random Forest làm mô hình đánh giá baseline
            clf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
            clf.fit(X_res, y_res)
            
            # Dự đoán TRÊN TẬP TEST GỐC (Chưa hề bị chạm vào)
            y_pred = clf.predict(self.X_test)
            y_proba = clf.predict_proba(self.X_test)
            
            # Tính toán Metrics: Precision, Recall, F1-macro, AUC-ROC
            precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, y_pred, average='macro')
            
            # Xử lý AUC cho nhị phân hoặc đa lớp
            if len(np.unique(self.y)) == 2:
                auc = roc_auc_score(self.y_test, y_proba[:, 1])
            else:
                auc = roc_auc_score(self.y_test, y_proba, multi_class='ovr')
                
            self.results.append({
                "Strategy": name,
                "Train_Size": len(y_res),
                "Precision (Macro)": precision,
                "Recall (Macro)": recall,
                "F1-Macro": f1,
                "AUC-ROC": auc
            })
            
        self._analyze()

    def _analyze(self):
        # Chuyển kết quả thành DataFrame để in cho đẹp
        df_res = pd.DataFrame(self.results).set_index("Strategy")
        
        md = f"### Báo cáo Hiệu năng: Xử lý Mất cân bằng Lớp (Class Imbalance)\n"
        md += "| Chiến lược Resampling | Số mẫu Train | Precision | Recall | F1-Macro | AUC-ROC |\n"
        md += "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
        
        best_f1 = df_res['F1-Macro'].max()
        best_strategy = ""
        
        for index, row in df_res.iterrows():
            mark = "🏆" if row['F1-Macro'] == best_f1 else ""
            if row['F1-Macro'] == best_f1: best_strategy = index
            md += f"| **{index}** | `{int(row['Train_Size'])}` | `{row['Precision (Macro)']:.4f}` | `{row['Recall (Macro)']:.4f}` | `{row['F1-Macro']:.4f}` {mark} | `{row['AUC-ROC']:.4f}` |\n"
            
        md += f"\n **Chiến lược đạt F1-Macro cao nhất:** `{best_strategy}`"
        display(Markdown(md))