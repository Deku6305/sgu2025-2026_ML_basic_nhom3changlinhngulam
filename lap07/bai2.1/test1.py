# Bước 1: Import thư viện
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Bước 2: Tải dữ liệu mẫu
iris = load_iris()
X = iris.data
y = iris.target

# Bước 3: Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Bước 4: Khởi tạo mô hình Random Forest
model = RandomForestClassifier(
    n_estimators=100,     
    max_depth=None,       
    random_state=42,      
    criterion='gini',     
    n_jobs=-1             
)
# Bước 5: Huấn luyện mô hình
model.fit(X_train, y_train)
# Bước 6: Dự đoán & đánh giá
y_pred = model.predict(X_test)
print("Độ chính xác (Accuracy):", accuracy_score(y_test, y_pred))
print("\nBáo cáo chi tiết:\n", classification_report(y_test, y_pred, target_names=iris.target_names))
