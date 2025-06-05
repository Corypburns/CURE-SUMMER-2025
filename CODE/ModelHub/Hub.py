import subprocess

def main():

    bayes_csic = '/home/cory/code/CISResearchSummer2025/CODE/CSIC-CODE-2010/CSIC-NaiveBayes.py'
    bayes_sqliv3 = '/home/cory/code/CISResearchSummer2025/CODE/SQLiV3-CODE/SQLiV3-NaiveBayes.py' 
    bayes_gambleryu = '/home/cory/code/CISResearchSummer2025/CODE/GAMBLERYU-CODE/GAMBLERYU-NaiveBayes.py' 
    xgboost_csic = '/home/cory/code/CISResearchSummer2025/CODE/CSIC-CODE-2010/CSIC-XGBoost.py'
    xgboost_sqliv3 = '/home/cory/code/CISResearchSummer2025/CODE/SQLiV3-CODE/SQLiV3-XGBoost.py'
    xgboost_gambleryu = '/home/cory/code/CISResearchSummer2025/CODE/GAMBLERYU-CODE/GAMBLERYU-XGBoost.py'

    print("""
    1) Naive Bayes - CSIC
    2) Naive Bayes - SQLiV3
    3) Naive Bayes - GAMBLERYU
    4) XGBoost - CSIC
    5) XGBoost - SQLiV3
    6) XGBoost - GAMBLERYU
    """)

    input_ = int(input("-> "))

    match(input_):
        case 1:
            subprocess.run(["python3", bayes_csic])
        case 2:
            subprocess.run(["python3", bayes_sqliv3])
        case 3:
            subprocess.run(["python3", bayes_gambleryu])
        case 4:
            subprocess.run(["python3", xgboost_csic])
        case 5:
            subprocess.run(["python3", xgboost_sqliv3])
        case 6:
            subprocess.run(["python3", xgboost_gambleryu])

if __name__ == '__main__':
    main()
