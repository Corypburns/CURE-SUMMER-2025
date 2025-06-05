import pandas as pd, time, csv, os, matplotlib.pyplot as mpl
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from array import array
from datetime import datetime

# CSV Headers
file_path = '/home/cory/code/CISResearchSummer2025/DATASETS/GAMBLERYU/GAMBLERYU.csv'
now = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
output_path = '/home/cory/code/CISResearchSummer2025/Outputs/NaiveBayes/NaiveBayes-GAMBLERYU'
joined_output_path = os.path.join(output_path, f'Output_{now}.csv')

if not os.path.exists(joined_output_path) or os.path.getsize(joined_output_path) == 0:
    with open(joined_output_path, mode='w') as f:
        f.write("Query, Prediction, Confidence, Actual\n")


# Load dataset
df = pd.read_csv(file_path, usecols=['Query', 'Label'], on_bad_lines='skip')

# Drop data that is NULL
df = df[['Query', 'Label']].dropna()
df['Label'] = df['Label'].astype(int)

# Define features and labels
X = df['Query'] # INPUT DATA
y = df['Label']  # 0 = normal, 1 = attack (Predicts 0 or 1 from X = df['request_text'])

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

# True Variables Matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Printing
try:
    with open(joined_output_path, mode='a') as f:
        writer = csv.writer(f)
        for indx, (i, prediction) in enumerate(zip(X_test_index, y_pred)):
            query = df.loc[i, 'Query']
            label = df.loc[i, 'Label']
            confidence = (conf[indx][prediction]) * 100
            print_statement = f"\nSentence: {query}\nLabel: {label}\nPrediction: {prediction}\nConfidence: {confidence:.2f}"
            print(print_statement)
            writer.writerow([query, prediction, f"{confidence:.2f}%", label])
            time.sleep(.5)
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
