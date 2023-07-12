import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

F01 = pd.read_csv("Dataset2/F01.csv")
F02 = pd.read_csv("Dataset2/F02.csv")
F03 = pd.read_csv("Dataset2/F03.csv")
F04 = pd.read_csv("Dataset2/F04.csv")
F05 = pd.read_csv("Dataset2/F05.csv")
F06 = pd.read_csv("Dataset2/F06.csv")
F07 = pd.read_csv("Dataset2/F07.csv")
F08 = pd.read_csv("Dataset2/F08.csv")
F09 = pd.read_csv("Dataset2/F09.csv")
Focus = pd.concat([F01,F02,F03,F04,F05,F06,F07,F08,F09],ignore_index=True)
Focus['label'] = [1 for i in range(len(Focus))]

R01 = pd.read_csv("Dataset2/R01.csv")
R02 = pd.read_csv("Dataset2/R02.csv")
R03 = pd.read_csv("Dataset2/R03.csv")
R04 = pd.read_csv("Dataset2/R04.csv")
R05 = pd.read_csv("Dataset2/R05.csv")
R06 = pd.read_csv("Dataset2/R06.csv")
R07 = pd.read_csv("Dataset2/R07.csv")
Relax = pd.concat([R01,R02,R03,R04,R05,R06,R07],ignore_index=True)
Relax['label'] = [0 for i in range(len(Relax))]

Global_Data = pd.concat([Focus,Relax],ignore_index=True)

X = Global_Data[['1' ,'2' ,'3' ,'4' ,'5' ,'6' ,'7' ,'8' ,'9' ,'10' ,'11' ,'12' ,'13' ,'14' ,'15' ,'16' ,'17' ,'18' ,
'19' ,'20' ,'21' ,'22' ,'23' ,'24' ,'25' ,'26' ,'27' ,'28' ,'29' ,'30' ,'31' ,'32' ,'33' ,'34' ,'35' ,'36' ,'37' ,'38' ,'39' ,'40' ,'41' ,'42' ,'43' ,'44' ,'45' ,'46' ,'47']]
y = Global_Data[['label']]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

randf = RandomForestClassifier()

xgboost = GradientBoostingClassifier()

svc = SVC()

log = LogisticRegression()

knn = KNeighborsClassifier()

cv = StratifiedKFold(5,shuffle=True)
scores = cross_validate(svc,X_train,y_train.values.ravel(),cv=cv,return_estimator=True)
sorted(scores.keys())
print("Max validation score : ",max(scores['test_score']))
print("Mean validation score : ",scores['test_score'].mean())

top_score = 0
model = None
for i in range(len(scores)):
    if scores['estimator'][i].score(X_test,y_test) > top_score:
        top_score = scores['estimator'][i].score(X_test,y_test)
        model = scores['estimator'][i]


print("Train : ",model.score(X_train,y_train))
print("Test : ",model.score(X_test,y_test))

F_test = pd.read_csv("Dataset2/F_test.csv")
X_new = F_test[['1' ,'2' ,'3' ,'4' ,'5' ,'6' ,'7' ,'8' ,'9' ,'10' ,'11' ,'12' ,'13' ,'14' ,'15' ,'16' ,'17' ,'18' ,
'19' ,'20' ,'21' ,'22' ,'23' ,'24' ,'25' ,'26' ,'27' ,'28' ,'29' ,'30' ,'31' ,'32' ,'33' ,'34' ,'35' ,'36' ,'37' ,'38' ,'39' ,'40' ,'41' ,'42' ,'43' ,'44' ,'45' ,'46' ,'47']]

prediction = model.predict(X_new)
print("Prediction : ",prediction)
count = 0
for i in prediction:
    if i == 1:
        count += 1
print("Number of F points : ",count)
print("Accuracy : ",count/len(X_new))

print("\n")

R_test = pd.read_csv("Dataset2/R_test.csv")
X_new2 = R_test[['1' ,'2' ,'3' ,'4' ,'5' ,'6' ,'7' ,'8' ,'9' ,'10' ,'11' ,'12' ,'13' ,'14' ,'15' ,'16' ,'17' ,'18' ,
'19' ,'20' ,'21' ,'22' ,'23' ,'24' ,'25' ,'26' ,'27' ,'28' ,'29' ,'30' ,'31' ,'32' ,'33' ,'34' ,'35' ,'36' ,'37' ,'38' ,'39' ,'40' ,'41' ,'42' ,'43' ,'44' ,'45' ,'46' ,'47']]

prediction2 = model.predict(X_new2)
print("Prediction : ",prediction2)
count2 = 0
for i in prediction2:
    if i == 0:
        count2 += 1
print("Number of R points : ",count2)
print("Accuracy : ",count2/len(X_new2))