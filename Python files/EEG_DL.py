import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras

F01 = pd.read_csv("Dataset/F01.csv")
F02 = pd.read_csv("Dataset/F02.csv")
F03 = pd.read_csv("Dataset/F03.csv")
F04 = pd.read_csv("Dataset/F04.csv")
F05 = pd.read_csv("Dataset/F05.csv")
F06 = pd.read_csv("Dataset/F06.csv")
F07 = pd.read_csv("Dataset/F07.csv")
F08 = pd.read_csv("Dataset/F08.csv")
F09 = pd.read_csv("Dataset/F09.csv")
Focus_subject0 = pd.concat([F01,F02,F03,F04,F05,F06,F07,F08,F09],ignore_index=True)
Focus_subject0['label'] = [1 for i in range(len(Focus_subject0))]

# F11 = pd.read_csv("Dataset/F11.csv")
# F12 = pd.read_csv("Dataset/F12.csv")
# F13 = pd.read_csv("Dataset/F13.csv")
# Focus_subject1 = pd.concat([F11,F12,F13],ignore_index=True)
# Focus_subject1['label'] = [1 for i in range(len(Focus_subject1))]

# F21 = pd.read_csv("Dataset/F21.csv")
# Focus_subject2 = pd.concat([F21],ignore_index=True)
# Focus_subject2['label'] = [1 for i in range(len(Focus_subject2))]

# F31 = pd.read_csv("Dataset/F31.csv")
# F32 = pd.read_csv("Dataset/F32.csv")
# F33 = pd.read_csv("Dataset/F33.csv")
# Focus_subject3 = pd.concat([F31,F32,F33],ignore_index=True)
# Focus_subject3['label'] = [1 for i in range(len(Focus_subject3))]

# R21 = pd.read_csv("Dataset/R21.csv")
# Relax_subject2 = pd.concat([R21],ignore_index=True)
# Relax_subject2['label'] = [0 for i in range(len(Relax_subject2))]

R01 = pd.read_csv("Dataset/R01.csv")
R02 = pd.read_csv("Dataset/R02.csv")
R03 = pd.read_csv("Dataset/R03.csv")
R04 = pd.read_csv("Dataset/R04.csv")
R05 = pd.read_csv("Dataset/R05.csv")
R06 = pd.read_csv("Dataset/R06.csv")
R07 = pd.read_csv("Dataset/R07.csv")
R08 = pd.read_csv("Dataset/R08.csv")
R09 = pd.read_csv("Dataset/R09.csv")
Relax_subject0 = pd.concat([R01,R02,R03,R04,R05,R06,R07,R08,R09],ignore_index=True)
Relax_subject0['label'] = [0 for i in range(len(Relax_subject0))]

# R11 = pd.read_csv("Dataset/R11.csv")
# R12 = pd.read_csv("Dataset/R12.csv")
# R13 = pd.read_csv("Dataset/R13.csv")
# Relax_subject1 = pd.concat([R11,R12,R13],ignore_index=True)
# Relax_subject1['label'] = [0 for i in range(len(Relax_subject1))]

# R31 = pd.read_csv("Dataset/R31.csv")
# R32 = pd.read_csv("Dataset/R32.csv")
# Relax_subject3 = pd.concat([R31,R32],ignore_index=True)
# Relax_subject3['label'] = [0 for i in range(len(Relax_subject3))]

# Data Scaling

Global_Data = pd.concat([Focus_subject0,Relax_subject0],ignore_index=True)
Global_Data['F1'] = [Global_Data['Gamma'][i]**3 for i in range(len(Global_Data))]
Global_Data['F2'] = [Global_Data['Gamma'][i]+Global_Data['Beta'][i] for i in range(len(Global_Data))]
Global_Data['F3'] = [Global_Data['Alpha'][i] - Global_Data['Gamma'][i]+Global_Data['Beta'][i] for i in range(len(Global_Data))]
Gamma_mean = np.mean(Global_Data['Gamma'])
Global_Data['F4'] = [Gamma_mean for i in range(len(Global_Data))]

# Data_subject1 = pd.concat([Focus_subject1,Relax_subject1],ignore_index=True)

# Data_subject2 = pd.concat([Focus_subject2,Relax_subject2],ignore_index=True)

# Data_subject3 = pd.concat([Focus_subject3,Relax_subject3],ignore_index=True)

# X = np.concatenate((X_subject0_scaled,X_subject1_scaled))
X = Global_Data[['Delta','Theta','Alpha','Beta','Gamma']]
# X = preprocessing.scale(X)

y = Global_Data[['label']]

norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learns mean, variance
X = norm_l(X)

X_t = np.tile(X,(100,1))
y_t = np.tile(y,(100,1))

callback = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy',restore_best_weights=True,baseline=0.70)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(500,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dense(275,activation='relu',kernel_regularizer = tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dense(150,activation='relu',kernel_regularizer = tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dense(25,activation='relu',kernel_regularizer = tf.keras.regularizers.l2(0.0001)),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics= tf.keras.metrics.BinaryAccuracy()
)

model.fit(
    X_t,y_t,            
    epochs=10,
    validation_split= 0.2,
    callbacks = [callback]
)

F_test = pd.read_csv("Dataset/F_test.csv")
F_test['F1'] = [F_test['Gamma'][i]**3 for i in range(len(F_test))]
F_test['F2'] = [F_test['Gamma'][i] + F_test['Beta'][i] for i in range(len(F_test))]
F_test['F3'] = [F_test['Alpha'][i] - F_test['Gamma'][i] + F_test['Beta'][i] for i in range(len(F_test))]
Gamma_mean = np.mean(F_test['Gamma'])
F_test['F4'] = [Gamma_mean for i in range(len(F_test))]
F_test['label'] = [ 1 for i in range(len(F_test))]
X_new = F_test[['Delta','Theta','Alpha','Beta','Gamma']]
y_new = F_test[['label']]

X_new = norm_l(X_new)

# prediction = model.predict(X_new)
# count = 0
# for i in prediction:
#     if i > 0.5:
#         count += 1

# # print(prediction)
# print(count)
# print(count/180)

R_test = pd.read_csv("Dataset/R_test.csv")
R_test['F1'] = [R_test['Gamma'][i]**3 for i in range(len(R_test))]
R_test['F2'] = [R_test['Gamma'][i] + R_test['Beta'][i] for i in range(len(R_test))]
R_test['F3'] = [R_test['Alpha'][i] - R_test['Gamma'][i] + R_test['Beta'][i] for i in range(len(R_test))]
Gamma_mean = np.mean(R_test['Gamma'])
R_test['F4'] = [Gamma_mean for i in range(len(R_test))]
R_test['label'] = [ 0 for i in range(len(R_test))]
X_new2 = R_test[['Delta','Theta','Alpha','Beta','Gamma']]

X_new2 = norm_l(X_new2)
y_new2 = R_test[['label']]

Global_Test_Data = pd.concat([F_test,R_test])

X_test = Global_Test_Data[['Delta','Theta','Alpha','Beta','Gamma']]
y_test = Global_Test_Data[['label']]

norm_l.adapt(X_test)
X_test = norm_l(X_test)

print("Test accuracy : ",model.evaluate(X_test,y_test)[1])

# prediction2 = model.predict(X_new2)
# count2 = 0
# for i in prediction2:
#     if i < 0.5:
#         count2 += 1

# # print(prediction2)
# print(count2)
# print(count2/180)