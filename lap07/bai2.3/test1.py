# # import pandas as pd
# # from sklearn.preprocessing import LabelEncoder
# # data = pd.DataFrame({
# #     'Age': [25, 30, 45],
# #     'Gender': ['Male', 'Female', 'Male'] 
# # })

# # print("Dữ liệu gốc:")
# # print(data)
# # encoder = LabelEncoder()
# # data['Gender'] = encoder.fit_transform(data['Gender'])

# # print("\nDữ liệu sau khi mã hóa:")
# # print(data)
# import pandas as pd

# # Giả sử ta có cột "Color"
# data = pd.DataFrame({'Color': ['Red', 'Blue', 'Green', 'Red']})

# # Mã hóa one-hot
# data_encoded = pd.get_dummies(data, columns=['Color'])
# print(data_encoded)
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB 
data = pd.DataFrame({
    'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue', 
              'Red', 'Green', 'Blue', 'Green', 'Red',
              'Blue', 'Blue', 'Green', 'Red', 'Green'],
    'Size': ['S', 'M', 'L', 'XL', 'M', 
             'S', 'L', 'M', 'XL', 'L',
             'S', 'M', 'L', 'S', 'M'],
    'Price': [10, 15, 20, 25, 30, 
              12, 22, 18, 28, 24,
              11, 16, 21, 13, 19],
    'Label': ['Yes', 'No', 'No', 'Yes', 'No', 
              'Yes', 'No', 'No', 'Yes', 'Yes',
              'Yes', 'No', 'No', 'Yes', 'No']})
X = data[['Color', 'Size', 'Price']]
y = data['Label']
categorical_features = ['Color', 'Size']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)],
    remainder='passthrough')
X_encoded = preprocessor.fit_transform(X)
print("Kích thước dữ liệu sau khi mã hóa:", X_encoded.shape) 
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nDự đoán:", y_pred)
print("Thực tế:", y_test.values)
print("Độ chính xác:", accuracy_score(y_test, y_pred))