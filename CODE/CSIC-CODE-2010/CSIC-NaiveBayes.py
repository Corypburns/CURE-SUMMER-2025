import pandas as pd, time, csv, os, matplotlib.pyplot as mpl
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from array import array
from datetime import datetime

# CSV Headers
file_path = '/home/cory/code/CISResearchSummer2025/DATASETS/CSIC2010/csic_database.csv'
now = datetime.now().strftime("%y-%m-%d %Hh:%Mm:%Ss")
output_path = '/home/cory/code/CISResearchSummer2025/Outputs/NaiveBayes/NaiveBayes-CSIC'
joined_output_path = os.path.join(output_path, f'Output_{now}.csv')

if not os.path.exists(joined_output_path) or os.path.getsize(joined_output_path) == 0:
    with open(joined_output_path, mode='w') as f:
        f.write("Prediction, Actual, Confidence, Method, Content, URL\n")

# Load dataset
df = pd.read_csv(file_path)

# Fill missing values
df['content'] = df['content'].fillna('')
df['URL'] = df['URL'].fillna('')
df['Method'] = df['Method'].fillna('')

# Combine fields into a single request text string
df['request_text'] = df['Method'] + ' ' + df['URL'] + ' ' + df['content']

# Define features and labels
X = df['request_text']
y = df['classification']  # 0 = normal, 1 = attack

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict and evaluate w/ Confidence PER Prediction
X_test_index = X_test.index
y_pred = model.predict(X_test_vec)
conf = model.predict_proba(X_test_vec)

# Confusion Matrix with label fallback
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
else:
    tn = fp = fn = tp = 0

# Printing and saving output
try:
    with open(joined_output_path, mode='a') as f:
        for indx, (i, prediction) in enumerate(zip(X_test_index, y_pred)):
            label = df.loc[i, 'classification']
            url = df.loc[i, 'URL']
            method = df.loc[i, 'Method']
            content = df.loc[i, 'content']
            confidence = (conf[indx][prediction]) * 100
            print_statement = f"URL: {url}\nMethod: {method}\nContent: {content}\nPrediction: {prediction} | Actual: {label} | Confidence: {confidence:.2f}%\n"
            print(print_statement)
            f.write(f"{prediction}, {label}, {confidence:.2f}%, {method}, {content}, {url}\n")
            time.sleep(0.5)
    print(f"True Positives (TP): {tp}\nTrue Negatives (TN): {tn}\nFalse Positives (FP): {fp}\nFalse Negatives (FN): {fn}")
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    mpl.show()
except KeyboardInterrupt:
    print(classification_report(y_test, y_pred))
    print(f"True Positives (TP): {tp}\nTrue Negatives (TN): {tn}\nFalse Positives (FP): {fp}\nFalse Negatives (FN): {fn}")
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    os.chdir(output_path)
    with open('results.txt', mode='w') as f:
        f.write(classification_report(y_test, y_pred))
        f.write("\n=== Confusion Matrix Stats ===\n")
        f.write(f"True Positives (TP): {tp}\n")
        f.write(f"True Negatives (TN): {tn}\n")
        f.write(f"False Positives (FP): {fp}\n")
        f.write(f"False Negatives (FN): {fn}\n")
    mpl.show()
