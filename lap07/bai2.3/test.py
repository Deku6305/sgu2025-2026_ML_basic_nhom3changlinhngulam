# Bước 1: Import các thư viện cần thiết
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Bước 2: Đọc dữ liệu từ file CSV
data = pd.read_csv("data/iris.csv")
print("5 dòng đầu tiên của dữ liệu:")
print(data.head())
# Bước 3: Chuẩn bị dữ liệu X và y
X = data.iloc[:, :-1]   
y = data.iloc[:, -1]    
# Bước 4: Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
# Bước 5: Khởi tạo và huấn luyện mô hình Gaussian Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)
# Bước 6: Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)
# Bước 7: Đánh giá mô hình
print("\n Ma trận nhầm lẫn (Confusion Matrix):")
print(confusion_matrix(y_test, y_pred))
print("\n Báo cáo phân loại (Classification Report):")
print(classification_report(y_test, y_pred))
print("\n Độ chính xác (Accuracy):", accuracy_score(y_test, y_pred))
