import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
data = pd.DataFrame({
    'text': [
        'I love this movie', 
        'This film is terrible', 
        'Amazing acting and story', 
        'I hate this movie', 
        'This movie is great',
        'The plot is boring',
        'Fantastic film',
        'Bad acting and dull scenes'
    ],
    'label': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative', 'positive', 'negative']
})
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.3, random_state=42)
vectorizer = CountVectorizer(stop_words='english')  
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)
print(" Ma trận nhầm lẫn:")
print(confusion_matrix(y_test, y_pred))
print("\n Báo cáo phân loại:")
print(classification_report(y_test, y_pred))
print("\n Độ chính xác:", accuracy_score(y_test, y_pred))
