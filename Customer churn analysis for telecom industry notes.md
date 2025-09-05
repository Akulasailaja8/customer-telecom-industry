**How to use:**



1.Install required packages:

**pip install pandas numpy scikit-learn matplotlib seaborn eli5** 

2.Open the Python script in a Jupyter notebook or run as a .py after updating the DATA\_PATH variable.



**Step by step process:**



import numpy as np

import pandas as pd

\# give you file path here

dataset = pd.read\_csv('OneDrive/Attachments/Telco-Customer-Churn.csv')

dataset.head()



print(dataset.isnull().sum())

print(dataset.describe())



import seaborn as sns

import matplotlib.pyplot as plt

print(dataset\['Churn'].value\_counts())

sns.countplot(x='Churn', data=dataset, hue='Churn', palette='coolwarm', legend=False)

plt.title('Churn Distribution')

plt.xlabel('Churn (0 = No, 1 = Yes)')

plt.ylabel('Count')

plt.show()





dataset\['TotalCharges'] = pd.to\_numeric(dataset\['TotalCharges'], errors='coerce')

dataset\['TotalCharges'] = dataset\['TotalCharges'].fillna(dataset\['TotalCharges'].median())



from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

categorical\_cols = \['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 

&nbsp;                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 

&nbsp;                   'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']

for col in categorical\_cols:

&nbsp;   dataset\[col] = labelencoder.fit\_transform(dataset\[col])



from sklearn.model\_selection import train\_test\_split

X = dataset.drop(\['customerID', 'Churn'], axis=1)

y = dataset\['Churn']

X\_train, X\_test, y\_train, y\_test = train\_test\_split(X, y, test\_size=0.2, random\_state=0)



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X\_train = scaler.fit\_transform(X\_train)

X\_test = scaler.transform(X\_test)



from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

clf.fit(X\_train, y\_train)

y\_pred = clf.predict(X\_test)



from sklearn.metrics import accuracy\_score

accuracy = accuracy\_score(y\_test, y\_pred)

print(f"Model Accuracy: {accuracy:.2f}")



**Model Accuracy: 0.79**



from sklearn.metrics import confusion\_matrix, ConfusionMatrixDisplay

cm = confusion\_matrix(y\_test, y\_pred)

disp = ConfusionMatrixDisplay(confusion\_matrix=cm, display\_labels=\["No Churn", "Churn"])

disp.plot(cmap="coolwarm")

plt.title('Confusion Matrix')

plt.show()



**# Install eli5 if not already installed**

**# !pip install eli5**  



import eli5

from eli5.sklearn import PermutationImportance

\# Compute feature importance using permutation importance

perm = PermutationImportance(clf, random\_state=42).fit(X\_test, y\_test)

\# Show weights

eli5.show\_weights(perm, feature\_names=X.columns.tolist())



\# Add predictions \& probabilities to dataset

clf.fit(X, y)   # Make sure X is a DataFrame, not X.values

dataset\['Churn\_Prob'] = clf.predict\_proba(X)\[:,1]

dataset\['Churn\_Pred'] = clf.predict(X)

\# Create Segments

def segment\_customer(row):

&nbsp;   if row\['Churn\_Pred'] == 1 and row\['Churn\_Prob'] > 0.6:

&nbsp;       return "At Risk"    # Likely to churn

&nbsp;   elif row\['Churn\_Pred'] == 0 and row\['tenure'] > 24:

&nbsp;       return "Loyal"      # Long tenure, not churning

&nbsp;   else:

&nbsp;       return "Dormant"    # Low tenure or uncertain case

dataset\['Segment'] = dataset.apply(segment\_customer, axis=1)

\# Segment distribution

print(dataset\['Segment'].value\_counts())

sns.countplot(x='Segment', data=dataset, hue='Segment', palette='viridis', legend=False)

plt.title('Customer Segments')

plt.xlabel('Segment')

plt.ylabel('Count')

plt.show()



**Segment**

**Loyal      3295**

**Dormant    1913**

**At Risk    1835**

**Name: count, dtype: int64**







