import pandas as pd
train_df=pd.read_csv("train_data.csv")
test_df=pd.read_csv("test_data.csv")        #Load the dataset
print(train_df.head())
R=train_df['Loan_Status'].value_counts()        #Count of each category in Loan_Status
print(R)
S=train_df['Loan_Status'].value_counts(normalize=True) * 100      #Percentage of each category in Loan_Status
print(S)
T=train_df.isnull().sum()             #Count of missing values in each column
print(T)
U=(train_df.isnull().sum()/len(train_df))*100            #Percentage of missing values in each column
print(U)
train_df.info()
D=pd.crosstab(train_df['Credit_History'],train_df['Loan_Status'],normalize='index')      
print(D)
gender_mode = train_df["Gender"].mode()[0]        #filling the missing values 
married_mode = train_df["Married"].mode()[0]
dependents_mode = train_df["Dependents"].mode()[0]
self_emp_mode = train_df["Self_Employed"].mode()[0]
credit_mode = train_df["Credit_History"].mode()[0]
loan_term_median = train_df["Loan_Amount_Term"].median()
loan_amt_median = train_df["LoanAmount"].median()
for df in [train_df, test_df]:
    df["Gender"].fillna(gender_mode, inplace=True)
    df["Married"].fillna(married_mode, inplace=True)
    df["Dependents"].fillna(dependents_mode, inplace=True)
    df["Self_Employed"].fillna(self_emp_mode, inplace=True)
    df["Credit_History"].fillna(credit_mode, inplace=True)
    df["Loan_Amount_Term"].fillna(loan_term_median, inplace=True)
    df["LoanAmount"].fillna(loan_amt_median, inplace=True)
test_ids = test_df["Loan_ID"]
train_df.drop("Loan_ID", axis=1, inplace=True)     #dropping Loan_Id column as it is not necessary for prediction 
test_df.drop("Loan_ID", axis=1, inplace=True)
for df in [train_df, test_df]:
    df["Married"] = df["Married"].map({'Yes':1,'No':0})     #Encoding the categorial value (changing the input in form of 0 and 1)
    df["Self_Employed"] = df["Self_Employed"].map({'Yes':1,'No':0})
    df["Gender"] = df["Gender"].map({'Male':1,'Female':0})
    df["Education"] = df["Education"].map({'Graduate':1,'Not Graduate':0})
    df["Dependents"] = df["Dependents"].replace('3+',3).astype(int)
train_df["Loan_Status"] = train_df["Loan_Status"].map({'Y':1,'N':0})
train_df = pd.get_dummies(train_df, columns=['Property_Area'], drop_first=True)
test_df = pd.get_dummies(test_df, columns=['Property_Area'], drop_first=True)
X = train_df.drop("Loan_Status", axis=1)
y = train_df["Loan_Status"]
test_df = test_df.reindex(columns=X.columns, fill_value=0)
from sklearn.preprocessing import StandardScaler
Scaler=StandardScaler()
num_cols=['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']
X[num_cols] = Scaler.fit_transform(X[num_cols])          #scaling the features 
test_df[num_cols] = Scaler.transform(test_df[num_cols])
print(train_df.describe())
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(                         #model trainning using logistic regression
    class_weight='balanced',
    max_iter=1000
)
model.fit(X,y)
test_predictions = model.predict(test_df)
test_probabilities = model.predict_proba(test_df)[:, 1]
print(test_predictions)
print(test_probabilities)
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y, model.predict(X))          #evaluating the model
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n")
print(classification_report(y, model.predict(X)))
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
y_prob = model.predict_proba(X)[:, 1]
fpr, tpr, thresholds = roc_curve(y, y_prob)
auc_score = roc_auc_score(y, y_prob)
print("ROC-AUC Score:", auc_score)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Loan Eligibility Model")
plt.legend()
plt.show()



