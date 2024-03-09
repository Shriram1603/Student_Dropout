import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv('dropout_dataset.csv', dtype={'Student Conduct': 'object'})

# Replace values in 'Father Occupation' column

# Select desired columns
columns_to_keep = ['Father Occupation', 'Father Education', 'Mother Occupation', 'Mother Education', 'Gender', 'Religion', 'Medium of Instruction', 'Community', 'Disability Group Name','Attendance','Long Absentees Above 15days(Yes/no)','Single Parent','Smoker','Conduct','Target']

data = data[columns_to_keep]
data=data.rename(columns={'If yes means why?':'Reason'})
last_index=data.index[-1]
data=data.drop(last_index)

data['Father Occupation'].fillna('Unknown', inplace=True)
data['Father Occupation'] = data['Father Occupation'].replace({'Self-employed': 1, 'Daily wages': 2, 'Private': 3, 'Unknown': 0, 'Government': 4})
# Convert 'Class' column to numeric
#data['Class'] = data['Class'].replace({'VII': 7, 'VI': 6, 'IX': 9, 'X': 10, 'VIII': 8})

# Replace values in 'Mother Occupation' column
data['Mother Occupation'].fillna('Unknown', inplace=True)
data['Mother Occupation'] = data['Mother Occupation'].replace({'Un-employed': 1, 'Unknown': 0, 'Daily wages': 2, 'Private': 3, 'Self-employed': 4})

# Fill missing values in 'Gender' column and replace values
data['Gender'].fillna('Male', inplace=True)
data['Gender'] = data['Gender'].replace({'': 'Male', 'Male': 1, 'Female': 2})

# Replace values in 'Religion' column
data['Religion'] = data['Religion'].replace({'Hindu': 0, 'Muslim': 1, 'Christian': 0})

# Calculate mode of 'Religion' column
religion_mode = data['Religion'].mode()
religion_mode=int(religion_mode)
print(religion_mode)
data['Religion'].fillna(religion_mode,inplace=True)
Father_edu_mean=data['Father Education'].mean()
Father_edu_mean=int(Father_edu_mean)
print(Father_edu_mean)
data['Father Education'].fillna(Father_edu_mean,inplace=True)
mother_edu_mean=data['Mother Education'].mean()
mother_edu_mean=int(mother_edu_mean)
data['Mother Education'].fillna(mother_edu_mean,inplace=True)
print(data['Mother Education'].count())
print(data['Medium of Instruction'].unique())
data['Medium of Instruction']=data['Medium of Instruction'].replace({'English':1,'Tamil':2,'Telugu':3})
medium_mode=data['Medium of Instruction'].mode()
medium_mode=int(medium_mode)
data['Medium of Instruction'].fillna(medium_mode,inplace=True)
print(data['Community'].unique())

data['Community']=data['Community'].replace({'BC-Muslim':0, 'BC-Others':1, 'SC-Others':2, 'MBC':3 ,'SC-Arunthathiyar':4 ,'OC':5})
community_mode=data['Community'].mode()
community_mode=int(community_mode)
data['Community'].fillna(community_mode,inplace=True)

print(data['Disability Group Name'].unique())

data['Disability Group Name']=data['Disability Group Name'].replace({'Not Applicable':0, 'Autism':1, 'Intellectual Disability':2,'Specific Learning disability':3, 'Cerebral Palsy':4 ,'Multiple disability':5,'Locomotor Impairment/Handicap':6})
disability_mode=int(data['Disability Group Name'].mode())
data['Disability Group Name'].fillna(disability_mode,inplace=True)

print(data['Attendance'].unique())
data['Attendance']=data['Attendance'].replace({'Regular':0,'Not Regular':1})


print(data['Long Absentees Above 15days(Yes/no)'].unique())

data['Long Absentees Above 15days(Yes/no)']=data['Long Absentees Above 15days(Yes/no)'].replace({'No':0,'Yes':1,'yes':1})


# print(data['Reason'].unique())
# data['Reason'].fillna('no',inplace=True)

# #Label Encoding
# label_encoder = LabelEncoder()
# data['Reason']=label_encoder.fit_transform(data['Reason'])

print(data['Single Parent'].unique())
data['Single Parent']=data['Single Parent'].replace({'Yes':1,'No':0})


print(data['Smoker'].unique())
data['Smoker']=data['Smoker'].replace({'Yes':1,'No':0})


print(data['Target'].unique())
data['Target']=data['Target'].replace({'Continued':1,'Dropout':0})
# print(data['Student Conduct'].unique())
# # print(data['Student Conduct'].isna().sum())
# print(data.iloc[:,-2])
print(data)
print(data['Conduct'].dtype)
print(data['Conduct'].unique())
data['Conduct']=data['Conduct'].replace({'Bad':0,'Good':1,'Very Good':2,'bad':0})
print(data)

x=data.iloc[:,:-1]
print(x)
y=data.iloc[:,-1]
print(y)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train the Random Forest model
rf_model = RandomForestClassifier(
    max_depth=None,
    min_samples_leaf=1,
    min_samples_split=5,
    n_estimators=150,
    random_state=42
)
rf_model.fit(X_train_smote, y_train_smote)


#dumping model

pickle.dump(rf_model,open('dropout.pkl','wb'))
model=pickle.load(open('dropout.pkl','rb'))

# Apply PCA on the features after the Random Forest model
pca = PCA(n_components=5)
principalcomponents = pca.fit_transform(X_train_smote)

# Get the feature names corresponding to the top features for each principal component
features = x.columns
for i in range(5):
    top_feature_index = pca.components_[i].argmax()
    top_feature_name = features[top_feature_index]
    print(f"Top Feature for Principal Component {i + 1}: {top_feature_name}")

# Visualize the explained variance ratio
# explained_var_ratio = pca.explained_variance_ratio_
# print("\nExplained Variance Ratio:", explained_var_ratio)

# # Create a DataFrame to identify the top principal components and their contributions
# components_df = pd.DataFrame(data={'Principal Component': range(1, 6), 'Explained Variance Ratio': explained_var_ratio})
# print("Top Principal Components:")
# print(components_df)

# Plotting the explained variance ratio
# plt.bar(range(1, len(explained_var_ratio) + 1), explained_var_ratio, alpha=0.5, align='center')
# plt.xlabel('Principal Components')
# plt.ylabel('Explained Variance Ratio')
# plt.title('Explained Variance Ratio for Each Principal Component')
# plt.show()