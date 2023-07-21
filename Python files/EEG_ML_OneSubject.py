import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

F01 = pd.read_csv("Dataset/F01.csv")
F02 = pd.read_csv("Dataset/F02.csv")
F03 = pd.read_csv("Dataset/F03.csv")
F04 = pd.read_csv("Dataset/F04.csv")
F05 = pd.read_csv("Dataset/F05.csv")
F06 = pd.read_csv("Dataset/F06.csv")
F07 = pd.read_csv("Dataset/F07.csv")
F08 = pd.read_csv("Dataset/F08.csv")
F09 = pd.read_csv("Dataset/F09.csv")
Focus = pd.concat([F01,F02,F03,F04,F05,F06,F07,F08,F09],ignore_index=True)
Focus['label'] = [1 for i in range(len(Focus))]

R01 = pd.read_csv("Dataset/R01.csv")
R02 = pd.read_csv("Dataset/R02.csv")
R03 = pd.read_csv("Dataset/R03.csv")
R04 = pd.read_csv("Dataset/R04.csv")
R05 = pd.read_csv("Dataset/R05.csv")
R06 = pd.read_csv("Dataset/R06.csv")
R07 = pd.read_csv("Dataset/R07.csv")
R08 = pd.read_csv("Dataset/R08.csv")
R09 = pd.read_csv("Dataset/R09.csv")
Relax = pd.concat([R01,R02,R03,R04,R05,R06,R07,R08,R09],ignore_index=True)
Relax['label'] = [0 for i in range(len(Relax))]

Global_Data = pd.concat([Focus,Relax],ignore_index=True)
# Global_Data['F1'] = [Global_Data['Gamma'][i]**3 for i in range(len(Global_Data))]
# Global_Data['F2'] = [Global_Data['Gamma'][i]+Global_Data['Beta'][i] for i in range(len(Global_Data))]
# Global_Data['F3'] = [Global_Data['Alpha'][i] - Global_Data['Gamma'][i]+Global_Data['Beta'][i] for i in range(len(Global_Data))]
# Gamma_mean = np.mean(Global_Data['Gamma'])
# Global_Data['F4'] = [Gamma_mean for i in range(len(Global_Data))]

X = Global_Data[['Delta','Theta','Alpha','Beta','Gamma']]
y = Global_Data[['label']]

X_train,X_cv,y_train,y_cv = train_test_split(X,y,test_size=0.2)

randf = RandomForestClassifier()
clf = GradientBoostingClassifier()

cv = StratifiedKFold(5,shuffle=True)
scores = cross_validate(clf,X_train,y_train.values.ravel(),cv=cv,return_estimator=True)
sorted(scores.keys())
print("Max validation score : ",max(scores['test_score']))
print("Mean validation score : ",scores['test_score'].mean())

top_score = 0
model = None
for i in range(len(scores)):
    if scores['estimator'][i].score(X_cv,y_cv) > top_score:
        top_score = scores['estimator'][i].score(X_cv,y_cv)
        model = scores['estimator'][i]

print("Train : ",model.score(X_train,y_train))
print("Split Test : ",model.score(X_cv,y_cv))

F_test = pd.read_csv("Dataset/F_test.csv")
F_test['label'] = [1 for i in range(len(F_test))]
R_test = pd.read_csv("Dataset/R_test.csv")
R_test['label'] = [0 for i in range(len(R_test))]

Data_test = pd.concat([F_test,R_test])

X_test = Data_test[['Delta','Theta','Alpha','Beta','Gamma']]
y_test = Data_test[['label']]

print("Test score : ",model.score(X_test,y_test))

# F_test['F1'] = [F_test['Gamma'][i]**3 for i in range(len(F_test))]
# F_test['F2'] = [F_test['Gamma'][i] + F_test['Beta'][i] for i in range(len(F_test))]
# F_test['F3'] = [F_test['Alpha'][i] - F_test['Gamma'][i] + F_test['Beta'][i] for i in range(len(F_test))]
# Gamma_mean = np.mean(F_test['Gamma'])
# F_test['F4'] = [Gamma_mean for i in range(len(F_test))]
# X_new = F_test[['Delta','Theta','Alpha','Beta','Gamma']]

# prediction = model.predict(X_new)
# print("Prediction : ",prediction)
# count = 0
# for i in prediction:
#     if i == 1:
#         count += 1
# print("Number of F points : ",count)
# print("Accuracy : ",count/len(X_new))

# print("\n")

# R_test['F1'] = [R_test['Gamma'][i]**3 for i in range(len(R_test))]
# R_test['F2'] = [R_test['Gamma'][i] + R_test['Beta'][i] for i in range(len(R_test))]
# R_test['F3'] = [R_test['Alpha'][i] - R_test['Gamma'][i] + R_test['Beta'][i] for i in range(len(R_test))]
# Gamma_mean = np.mean(R_test['Gamma'])
# R_test['F4'] = [Gamma_mean for i in range(len(R_test))]
# X_new2 = R_test[['Delta','Theta','Alpha','Beta','Gamma']]

# prediction2 = model.predict(X_new2)
# print("Prediction : ",prediction2)
# count2 = 0
# for i in prediction2:
#     if i == 0:
#         count2 += 1
# print("Number of R points : ",count2)
# print("Accuracy : ",count2/len(X_new2))