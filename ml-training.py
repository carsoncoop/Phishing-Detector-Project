import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import confusion_matrix
import joblib

# Load the datasets
df1 = pd.read_csv("data/Ling(Spam&Ham).csv",)
df2 = pd.read_csv("data/Kaggle(50-50).csv")
df2 = pd.read_csv("data/Kaggle(24-76).csv")


# Clean empty values
df1["subject"] = df1["subject"].fillna("")
df1["body"] = df1["body"].fillna("")
df2["text"] = df2["text"].fillna("")

# Combine subject + body (df1)
df1["text"] = df1["subject"] + " " + df1["body"]

# Format individual datasets
df1 = df1[["text", "label"]]
df2 = df2[["text", "label"]]

# Combine datasets
df = pd.concat([df1, df2], ignore_index=True)

# Features (text) and labels (0/1)
X = df["text"] 
y = df["label"] 

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

# Build model pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=2000))
])

# Train
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

y_pred = model.predict(X_test)

recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Recall:", recall)
print("F1 Score:", f1)

precision = precision_score(y_test, y_pred) 
print("Precision:", precision)#Percentage of true positives

cm = confusion_matrix(y_test, y_pred)
print('True negatives: ', cm[0][0])
print('False positives: ', cm[0][1])
print('False nagetives', cm[1][0])
print('True positives: ', cm[1][1])
joblib.dump(model, "spam_pipeline.joblib")

# Now we can save the model and make a new script that lets users input an email, 
#   or maybe look into making an extension that automatically scans your emails? 
#   could start with gmail only or something maybe
