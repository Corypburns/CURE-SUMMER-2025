import xgboost as xgb, os, pandas as pd, time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# Files
output_file = 'XGBoostOutput.csv'
dataset_file = 'csic_database.csv'
dataset_path = '/home/cory/code/CISResearchSummer2025/DATASETS/CSIC2010'
output_path = '/home/cory/code/CISResearchSummer2025/Outputs/XDGBoost'

# Joined files with 'os'
joined_data = os.path.join(dataset_path, dataset_file)
joined_output = os.path.join(output_path, output_file)

if not os.path.exists(joined_output):
    with open(joined_output, mode='w') as f:
        f.write("Classification,Actual,Method,Content,URL\n")

# Load Dataset
df = pd.read_csv(joined_data)

# For missing content, fill with 'N/A'
df['content'] = df['content'].fillna('')
df['URL'] = df['URL'].fillna('')
df['Method'] = df['Method'].fillna('')

# Combination of the three things above
df['full_request_data'] = df['content'] + ' ' + df['URL'] + ' ' + df['Method']
df['labels'] = df['Classification'].map({'Normal': 0, 'Anomalous': 1})

# Input data + Labels
input_data = df['full_request_data']
label_data = df['labels']

# Training + Testing
X_train, X_test, y_train, y_test = train_test_split(input_data, label_data, test_size=0.2, random_state=42)

# Vectorizing
vector = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_vec = vector.fit_transform(X_train)
X_test_vec = vector.transform(X_test)

# XGB Model
model = xgb.XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=6, 
        use_label_encoder=False, 
        eval_metric='logloss'
)

model.fit(X_train_vec, y_train)

# Evaluation
prediction = model.predict(X_test_vec)
X_index = X_test.index

# Get confusion matrix values
tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()

# Printing
try:
    with open(joined_output, mode='w') as f:
        f.write("Classification,Prediction,Method,Content,URL\n")
        for i, predict in zip(X_index, prediction):
            label = df.loc[i, 'Classification']
            url = df.loc[i, 'URL']
            method = df.loc[i, 'Method']
            content = df.loc[i, 'content']
            print_statement = f"URL: {url}\nMethod: {method}\nContent: {content}\nPrediction: {predict} | Actual: {label}\n"
            print(print_statement)
            f.write(f"{label},{predict},{method},{content},{url}\n")
            time.sleep(.5)
except KeyboardInterrupt:
    print(classification_report(y_test, prediction))
    os.chdir(output_path)
    with open('results.txt', mode='w') as f:
        f.write(classification_report(y_test, prediction))
        f.write("\n=== Confusion Matrix Stats ===\n")
        f.write(f"True Positives (TP): {tp}\n")
        f.write(f"True Negatives (TN): {tn}\n")
        f.write(f"False Positives (FP): {fp}\n")
        f.write(f"False Negatives (FN): {fn}\n")
