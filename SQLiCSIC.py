import pandas as pd, time, csv, os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# CSV Headers
file_name = 'output.csv'
file_path = '/home/cory/code/CISResearchSummer2025/DATASETS/CSIC2010'
joined_file_path = os.path.join(file_path, file_name)
output_path = '/home/cory/code/CISResearchSummer2025/Outputs/NaiveBayes'
joined_output_path = os.path.join(output_path, 'Output.csv')

if not os.path.exists(joined_file_path):
    with open('output.csv', mode='w') as f:
        f.write("Classification Actual Method Content URL")


# Load dataset
df = pd.read_csv("csic_database.csv")

# Fill missing values
df['content'] = df['content'].fillna('')
df['URL'] = df['URL'].fillna('')
df['Method'] = df['Method'].fillna('')

# Combine fields into a single request text string
df['request_text'] = df['Method'] + ' ' + df['URL'] + ' ' + df['content']

# Define features and labels
X = df['request_text'] # INPUT DATA
y = df['classification']  # 0 = normal, 1 = attack (Predicts 0 or 1 from X = df['request_text'])

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict and evaluate
X_test_index = X_test.index
y_pred = model.predict(X_test_vec)
try:
    with open(joined_output_path, mode='w') as f:
        for i, prediction in zip(X_test_index, y_pred):
            label = df.loc[i, 'Classification'] # Per row, in column 'Classification'
            url = df.loc[i, 'URL'] # Per row, in column 'URL'
            method = df.loc[i, 'Method'] # Per row, in column 'Method'
            content = df.loc[i, 'content']
            print_statement = f"URL: {url}\nMethod: {method}\nContent: {content}\nPrediction: {prediction} | Actual: {label}\n"
            print(print_statement)
            f.write(f"{label} {prediction} {method} {content} {url}\n")
            time.sleep(.5)
except KeyboardInterrupt:
    print(classification_report(y_test, y_pred))
    os.chdir(output_path)
    with open('results.txt', mode='w') as f:
        f.write(classification_report(y_test, y_pred))
