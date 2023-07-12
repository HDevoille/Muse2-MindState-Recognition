from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.model_selection import KFold,StratifiedKFold,StratifiedGroupKFold
from sklearn import preprocessing
import pandas as pd
import numpy as np

F01 = pd.read_csv("Dataset/F01.csv")
F02 = pd.read_csv("Dataset/F02.csv")
F03 = pd.read_csv("Dataset/F03.csv")
Focus_subject0 = pd.concat([F01,F02,F03],ignore_index=True)
Focus_subject0['label'] = [1 for i in range(len(Focus_subject0))]

# F11 = pd.read_csv("Dataset/F11.csv")
# F12 = pd.read_csv("Dataset/F12.csv")
# F13 = pd.read_csv("Dataset/F13.csv")
# Focus_subject1 = pd.concat([F11,F12,F13],ignore_index=True)
# Focus_subject1['label'] = [1 for i in range(len(Focus_subject1))]

F21 = pd.read_csv("Dataset/F21.csv")
Focus_subject2 = pd.concat([F21],ignore_index=True)
Focus_subject2['label'] = [1 for i in range(len(Focus_subject2))]

F31 = pd.read_csv("Dataset/F31.csv")
F32 = pd.read_csv("Dataset/F32.csv")
F33 = pd.read_csv("Dataset/F33.csv")
Focus_subject3 = pd.concat([F31,F32,F33],ignore_index=True)
Focus_subject3['label'] = [1 for i in range(len(Focus_subject3))]

R21 = pd.read_csv("Dataset/R21.csv")
Relax_subject2 = pd.concat([R21],ignore_index=True)
Relax_subject2['label'] = [0 for i in range(len(Relax_subject2))]

R01 = pd.read_csv("Dataset/R01.csv")
R02 = pd.read_csv("Dataset/R02.csv")
R03 = pd.read_csv("Dataset/R03.csv")
Relax_subject0 = pd.concat([R01,R02,R03],ignore_index=True)
Relax_subject0['label'] = [0 for i in range(len(Relax_subject0))]

# R11 = pd.read_csv("Dataset/R11.csv")
# R12 = pd.read_csv("Dataset/R12.csv")
# R13 = pd.read_csv("Dataset/R13.csv")
# Relax_subject1 = pd.concat([R11,R12,R13],ignore_index=True)
# Relax_subject1['label'] = [0 for i in range(len(Relax_subject1))]

R31 = pd.read_csv("Dataset/R31.csv")
R32 = pd.read_csv("Dataset/R32.csv")
Relax_subject3 = pd.concat([R31,R32],ignore_index=True)
Relax_subject3['label'] = [0 for i in range(len(Relax_subject3))]

# Data Scaling

Data_subject0 = pd.concat([Focus_subject0,Relax_subject0],ignore_index=True)

# Data_subject1 = pd.concat([Focus_subject1,Relax_subject1],ignore_index=True)
# X_subject1 = Data_subject1[['Delta','Theta','Alpha','Beta','Gamma']]
# X_subject1_scaled = preprocessing.scale(X_subject1)

Data_subject2 = pd.concat([Focus_subject2,Relax_subject2],ignore_index=True)

Data_subject3 = pd.concat([Focus_subject3,Relax_subject3],ignore_index=True)

Global_Data = pd.concat([Data_subject0,Data_subject2,Data_subject3],ignore_index=True)
Global_Data['F1'] = [Global_Data['Gamma'][i]**3 for i in range(len(Global_Data))]
Global_Data['F2'] = [Global_Data['Gamma'][i]+Global_Data['Beta'][i] for i in range(len(Global_Data))]
Global_Data['F3'] = [Global_Data['Alpha'][i] - Global_Data['Gamma'][i]+Global_Data['Beta'][i] for i in range(len(Global_Data))]
Gamma_mean = np.mean(Global_Data['Gamma'])
Global_Data['F4'] = [Gamma_mean for i in range(len(Global_Data))]


# X = np.concatenate((X_subject0_scaled,X_subject1_scaled))
X = Global_Data[['Delta','Theta','Alpha','Beta','Gamma','F1','F4']]
# X = preprocessing.scale(X)

y = Global_Data[['label']]
# print(y)

# X_validation = pd.concat([Focus_subject2[['Delta','Theta','Alpha','Beta','Gamma']],Relax_subject2[['Delta','Theta','Alpha','Beta','Gamma']]],ignore_index=True)
# X_validation = preprocessing.scale(X_validation)
# y_validation = pd.concat([Focus_subject2[['label']],Relax_subject2[['label']]],ignore_index=True)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
print(X_train.shape,y_train.shape)
print(X_test.shape, y_test.shape)

# ML model

best_score = 0.0
best_algo = ""
best_model = None

print("\n")

l_log = LogisticRegression()
model = l_log.fit(X_train,y_train.values.ravel())

print("Logistic Regression score : ")
print("Train : ",model.score(X_train,y_train))
print("Test : ",model.score(X_test,y_test))
# print("Generalization error : ",model.score(X_validation,y_validation))

l_log_score = model.score(X_test,y_test)
if l_log_score > best_score:
    best_score = l_log_score
    best_algo = "Logistic Regression"
    best_model = model

print("\n")

knn = KNeighborsClassifier()
model = knn.fit(X_train,y_train.values.ravel())

print("KNN score : ")
print("Train : ",model.score(X_train,y_train))
print("Test : ",model.score(X_test,y_test))
# print("Generalization error : ",model.score(X_validation,y_validation))

knn_score = model.score(X_test,y_test)
if knn_score > best_score:
    best_score = knn_score
    best_algo = "KNN"
    best_model = model

print("\n")

clf = DecisionTreeClassifier()
model = clf.fit(X_train,y_train.values.ravel())

print("Decision Tree score : ")
print("Train : ",model.score(X_train,y_train))
print("Test : ",model.score(X_test,y_test))
# print("Generalization error : ",model.score(X_validation,y_validation))

clf_score = model.score(X_test,y_test)
if clf_score > best_score:
    best_score = clf_score
    best_algo = "Decision Tree"
    best_model = model

# perp = Perceptron()
# model = perp.fit(X_train,y_train.values.ravel())

# print("Perceptron score : ")
# print("Train : ",model.score(X_train,y_train))
# print("Test : ",model.score(X_test,y_test))

# perp_score = model.score(X_test,y_test)
# if perp_score > best_score:
#     best_score = perp_score
#     best_algo = "Perceptron"

print("\n")

svm = SVC()
model = svm.fit(X_train,y_train.values.ravel())

print("SVM Classifier score : ")
print("Train : ",model.score(X_train,y_train))
print("Test : ",model.score(X_test,y_test))
# print("Generalization error : ",model.score(X_validation,y_validation))

svm_score = model.score(X_test,y_test)
if svm_score > best_score:
    best_score = svm_score
    best_algo = "SVM"
    best_model = model

print("\n")

randf = RandomForestClassifier()
model = randf.fit(X_train,y_train.values.ravel())

print("Random Forest Classifier")
print("Train : ",model.score(X_train,y_train))
print("Test : ",model.score(X_test,y_test))
# print("Generalization error : ",model.score(X_validation,y_validation))

randf_score = model.score(X_test,y_test)
if randf_score > best_score:
    best_score = randf_score
    best_algo = "Random Forest"
    best_model = model

print("\n")

xgboost = GradientBoostingClassifier()
model = xgboost.fit(X_train,y_train.values.ravel())

print("xG Boost Classifier")
print("Train : ",model.score(X_train,y_train))
print("Test : ",model.score(X_test,y_test))
# print("Generalization error : ",model.score(X_validation,y_validation))

xgboost_score = model.score(X_test,y_test)
if xgboost_score > best_score:
    best_score = xgboost_score
    best_algo = "xG Boost"
    best_model = model

# mlpc = MLPClassifier()
# model = mlpc.fit(X_train,y_train.values.ravel())

# print("Multi Layer Perceptron score : ")
# print("Train : ",model.score(X_train,y_train))
# print("Test : ",model.score(X_test,y_test))

# mlpc_score = model.score(X_test,y_test)
# if mlpc_score > best_score:
#     best_score = mlpc_score
#     best_algo = "Multilayer Perceptron"

print("\n")
print("Meilleur algo : "+best_algo)
print("Score : ",best_score)

# print("\n")
# predictions = best_model.predict(X_validation)
# print("Prediction :",predictions)
# print("Vrai label : ",y_validation.values)
# print("Generalization error : ",best_model.score(X_validation,y_validation))

print("\n")

cv = StratifiedKFold(5,shuffle=True)
scores = cross_validate(best_model,X_train,y_train.values.ravel(),cv=cv,return_estimator=True)
sorted(scores.keys())
print("Max validation score : ",max(scores['test_score']))
print("Mean validation score : ",scores['test_score'].mean())

top_score = 0
top_model = None
for i in range(len(scores)):
    if scores['estimator'][i].score(X_test,y_test) > top_score:
        top_score = scores['estimator'][i].score(X_test,y_test)
        top_model = scores['estimator'][i]


print("Test score with best validation score model : ",top_score)

print("\n")

F_test = pd.read_csv("Dataset/F_test.csv")
F_test['F1'] = [F_test['Gamma'][i]**3 for i in range(len(F_test))]
F_test['F2'] = [F_test['Gamma'][i] + F_test['Beta'][i] for i in range(len(F_test))]
F_test['F3'] = [F_test['Alpha'][i] - F_test['Gamma'][i] + F_test['Beta'][i] for i in range(len(F_test))]
Gamma_mean = np.mean(F_test['Gamma'])
F_test['F4'] = [Gamma_mean for i in range(len(F_test))]
X_new = F_test[['Delta','Theta','Alpha','Beta','Gamma','F1','F4']]

prediction = top_model.predict(X_new)
print("Prediction : ",prediction)
count = 0
for i in prediction:
    if i == 1:
        count += 1
print("Number of F points : ",count)
acc = count/len(X_new)
print("Accuracy : ",acc)

print("\n")

R_test = pd.read_csv("Dataset/R_test.csv")
R_test['F1'] = [R_test['Gamma'][i]**3 for i in range(len(R_test))]
R_test['F2'] = [R_test['Gamma'][i] + R_test['Beta'][i] for i in range(len(R_test))]
R_test['F3'] = [R_test['Alpha'][i] - R_test['Gamma'][i] + R_test['Beta'][i] for i in range(len(R_test))]
Gamma_mean = np.mean(R_test['Gamma'])
R_test['F4'] = [Gamma_mean for i in range(len(R_test))]
X_new2 = R_test[['Delta','Theta','Alpha','Beta','Gamma','F1','F4']]

prediction2 = top_model.predict(X_new2)
print("Prediction : ",prediction2)
count2 = 0
for i in prediction2:
    if i == 0:
        count2 += 1
print("Number of R points : ",count2)
print("Accuracy : ",count2/len(X_new2))

print("\n")

X_new3 = pd.concat([X_new2.head(10),X_new.head(10)])
prediction3 = top_model.predict(X_new3)
print("Final test : ",prediction3[:10])
print(prediction3[10:])
