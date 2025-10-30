# Bước 1: Import thư viện
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
# Bước 2: Tải dữ liệu
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
# Bước 3: Huấn luyện mô hình Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
# Bước 4: Lấy giá trị feature importance
importances = model.feature_importances_

# Bước 5: Tạo bảng kết quả
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
print(importance_df)
# Bước 6: Trực quan hóa
plt.figure(figsize=(8,5))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.gca().invert_yaxis()  
plt.title("Feature Importance trong Random Forest (Iris Dataset)")
plt.xlabel("Mức độ quan trọng")
plt.show()
