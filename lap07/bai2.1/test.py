# Bước 1: Import thư viện
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Bước 2: Tải dữ liệu mẫu (bộ Iris)
iris = load_iris()
X = iris.data      # các đặc trưng (features)
y = iris.target    # nhãn (labels)

# Bước 3: Chia dữ liệu thành train/test (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Bước 4: Xây dựng mô hình cây quyết định
model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)

# Bước 5: Huấn luyện mô hình
model.fit(X_train, y_train)

# Bước 6: Dự đoán & đánh giá
y_pred = model.predict(X_test)
print("Độ chính xác (Accuracy):", accuracy_score(y_test, y_pred))

# (Tùy chọn) Trực quan hóa cây quyết định
plt.figure(figsize=(10,6))
plot_tree(model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()
