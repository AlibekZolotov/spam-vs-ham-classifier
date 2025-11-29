import pandas as pd
df =pd.read_csv("combined_dataset.csv", encoding="latin-1")
print(df.head())
print(df.columns)
print(df.isnull().sum())
df['target_num']=df['target'].map({'ham':0, 'spam':1})
print(df.head())
print(df['target_num'].value_counts())
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
X =df['text']
y= df['target_num']
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2, random_state=42
)
vectorizer =CountVectorizer()
X_train_counts=vectorizer.fit_transform(X_train)
X_test_counts=vectorizer.transform(X_test)

print("Training features shape:", X_train_counts.shape)
print("Testing features shape:", X_test_counts.shape)
print(vectorizer.get_feature_names_out()[:10])
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
model = MultinomialNB()
model.fit(X_train_counts,y_train)
y_pred=model.predict(X_test_counts)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confussion Matrix:\n", confusion_matrix(y_test,y_pred))
print("Classification Report \n", classification_report(y_test, y_pred))
while True:
    message=input("\n Enter a message (or 'quit' to exit): ")
    if message.lower() == 'quit':
        break
    
    
    message_counts=vectorizer.transform([message])
    prediction= model.predict(message_counts)[0]
    if prediction ==1:
        print("🚨 This is message is SPAM! ")
    else:
        print("✅ This is message is HAM (not spam)! ")
