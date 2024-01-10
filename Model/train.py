import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from joblib import dump, load

def convertTrueToFalse(item):
    if (item == 'TRUE'):
        return 1
    else:
        return 0

def convertFightResult(item):
    if (item == 'NOT_OVER'):
        return 0
    elif (item == 'P1'):
        return 1
    elif (item == 'P2'):
        return 2

df1 = pd.read_csv('Player1.csv', header=None, names=['timer','fight_result','has_round_started','is_round_over','Player1_ID','health','x_coord','y_coord','is_jumping',
                                                     'is_crouching','is_player_in_move','move_id','player1_buttons up','player1_buttons down','player1_buttons right',
                                                     'player1_buttons left','Y','B','X','A','L','R','Player2_ID','Player2health','Player2x_coord','Player2y_coord',
                                                     'Player2is_jumping','Player2is_crouching','Player2is_player_in_move','Player2move_id','player2_buttons up',
                                                     'player2_buttons down','player2_buttons right','player2_buttons left','Player2 Y','Player2 B','Player2 X',
                                                     'Player2 A','Player2 L','Player2 R'])
df2 = pd.read_csv('Player2.csv', header=None, names=['timer','fight_result','has_round_started','is_round_over','Player1_ID','health','x_coord','y_coord','is_jumping',
                                                     'is_crouching','is_player_in_move','move_id','player1_buttons up','player1_buttons down','player1_buttons right',
                                                     'player1_buttons left','Y','B','X','A','L','R','Player2_ID','Player2health','Player2x_coord','Player2y_coord',
                                                     'Player2is_jumping','Player2is_crouching','Player2is_player_in_move','Player2move_id','player2_buttons up',
                                                     'player2_buttons down','player2_buttons right','player2_buttons left','Player2 Y','Player2 B','Player2 X',
                                                     'Player2 A','Player2 L','Player2 R'])

df1.drop(df1[df1['has_round_started'] == 'FALSE'].index, inplace=True)
df2.drop(df2[df2['has_round_started'] == 'FALSE'].index, inplace=True)

# array = df1.values

# # set numpy print options to display all elements
# np.set_printoptions(threshold=np.inf)

# # print the array
# print(array)

# df1['fight_result'] = df1['fight_result'].apply(convertFightResult)
# df2['fight_result'] = df2['fight_result'].apply(convertFightResult)

# df1['is_jumping'] = df1['is_jumping'].apply(convertTrueToFalse)
# df2['is_jumping'] = df2['is_jumping'].apply(convertTrueToFalse)

# df1['is_crouching'] = df1['is_crouching'].apply(convertTrueToFalse)
# df2['is_crouching'] = df2['is_crouching'].apply(convertTrueToFalse)

# df1['is_player_in_move'] = df1['is_player_in_move'].apply(convertTrueToFalse)
# df2['is_player_in_move'] = df2['is_player_in_move'].apply(convertTrueToFalse)

# df1['player1_buttons up'] = df1['player1_buttons up'].apply(convertTrueToFalse)
# df2['player1_buttons up'] = df2['player1_buttons up'].apply(convertTrueToFalse)

# df1['player1_buttons down'] = df1['player1_buttons down'].apply(convertTrueToFalse)
# df2['player1_buttons down'] = df2['player1_buttons down'].apply(convertTrueToFalse)

# df1['player1_buttons right'] = df1['player1_buttons right'].apply(convertTrueToFalse)
# df2['player1_buttons right'] = df2['player1_buttons right'].apply(convertTrueToFalse)

# df1['player1_buttons left'] = df1['player1_buttons left'].apply(convertTrueToFalse)
# df2['player1_buttons left'] = df2['player1_buttons left'].apply(convertTrueToFalse)

# df1['Player2is_jumping'] = df1['Player2is_jumping'].apply(convertTrueToFalse)
# df2['Player2is_jumping'] = df2['Player2is_jumping'].apply(convertTrueToFalse)

# df1['Player2is_crouching'] = df1['Player2is_crouching'].apply(convertTrueToFalse)
# df2['Player2is_crouching'] = df2['Player2is_crouching'].apply(convertTrueToFalse)

# df1['Player2is_player_in_move'] = df1['Player2is_player_in_move'].apply(convertTrueToFalse)
# df2['Player2is_player_in_move'] = df2['Player2is_player_in_move'].apply(convertTrueToFalse)

# df1['player2_buttons up'] = df1['player2_buttons up'].apply(convertTrueToFalse)
# df2['player2_buttons up'] = df2['player2_buttons up'].apply(convertTrueToFalse)

# df1['player2_buttons down'] = df1['player2_buttons down'].apply(convertTrueToFalse)
# df2['player2_buttons down'] = df2['player2_buttons down'].apply(convertTrueToFalse)

# df1['player2_buttons right'] = df1['player2_buttons right'].apply(convertTrueToFalse)
# df2['player2_buttons right'] = df2['player2_buttons right'].apply(convertTrueToFalse)

# df1['player2_buttons left'] = df1['player2_buttons left'].apply(convertTrueToFalse)
# df2['player2_buttons left'] = df2['player2_buttons left'].apply(convertTrueToFalse)

# df1['Y'] = df1['Y'].apply(convertTrueToFalse)
# df2['Y'] = df2['Y'].apply(convertTrueToFalse)

# df1['A'] = df1['A'].apply(convertTrueToFalse)
# df2['A'] = df2['A'].apply(convertTrueToFalse)

# df1['X'] = df1['X'].apply(convertTrueToFalse)
# df2['X'] = df2['X'].apply(convertTrueToFalse)

# df1['B'] = df1['B'].apply(convertTrueToFalse)
# df2['B'] = df2['B'].apply(convertTrueToFalse)

# df1['L'] = df1['L'].apply(convertTrueToFalse)
# df2['L'] = df2['L'].apply(convertTrueToFalse)

# df1['R'] = df1['R'].apply(convertTrueToFalse)
# df2['R'] = df2['R'].apply(convertTrueToFalse)

# df1['Player2 Y'] = df1['Player2 Y'].apply(convertTrueToFalse)
# df2['Player2 Y'] = df2['Player2 Y'].apply(convertTrueToFalse)

# df1['Player2 A'] = df1['Player2 A'].apply(convertTrueToFalse)
# df2['Player2 A'] = df2['Player2 A'].apply(convertTrueToFalse)

# df1['Player2 X'] = df1['Player2 X'].apply(convertTrueToFalse)
# df2['Player2 X'] = df2['Player2 X'].apply(convertTrueToFalse)

# df1['Player2 B'] = df1['Player2 B'].apply(convertTrueToFalse)
# df2['Player2 B'] = df2['Player2 B'].apply(convertTrueToFalse)

# df1['Player2 L'] = df1['Player2 L'].apply(convertTrueToFalse)
# df2['Player2 L'] = df2['Player2 L'].apply(convertTrueToFalse)

# df1['Player2 R'] = df1['Player2 R'].apply(convertTrueToFalse)
# df2['Player2 R'] = df2['Player2 R'].apply(convertTrueToFalse)



df1.drop(columns=['timer','has_round_started','is_round_over','Player1_ID','fight_result','Player2_ID'],inplace=True, axis=1)
df2.drop(columns=['timer','has_round_started','is_round_over','Player1_ID','fight_result','Player2_ID'],inplace=True, axis=1)

y1 = df1[['player1_buttons up','player1_buttons down','player1_buttons right','player1_buttons left','Y','B','X','A','L','R']].values
x1 = df1.drop(columns=['player1_buttons up','player1_buttons down','player1_buttons right','player1_buttons left','Y','B','X','A','L','R'],inplace=False, axis=1)

#///////////////////////////////////////////

# model1 = MLPRegressor(hidden_layer_sizes=(400,300,200,100,50), max_iter=5000)

# model1.fit(X_train1,y_train1)

# y_pred1 = model1.predict(X_test1)

# accuracy = r2_score(y_test1, y_pred1)
# print('Accuracy of Player 1:', accuracy)

y2 = df2[['player2_buttons up','player2_buttons down','player2_buttons right','player2_buttons left','Player2 Y','Player2 B','Player2 X',
                                                     'Player2 A','Player2 L','Player2 R']].values
x2 = df2.drop(columns=['player2_buttons up','player2_buttons down','player2_buttons right','player2_buttons left','Player2 Y','Player2 B','Player2 X',
                                        'Player2 A','Player2 L','Player2 R'], inplace=False, axis=1)

#///////////////////////////////////////////

# model2 = MLPRegressor(hidden_layer_sizes=(19,19,19,19,19), max_iter=1000)

# model2.fit(X_train2,y_train2)

# y_pred2 = model1.predict(X_test2)

# accuracy = r2_score(y_test2, y_pred2)
# print('Accuracy of Player 1:', accuracy)

#///////////////////////////////////////////

# # Scale the data
# scaler = StandardScaler()
# X_train1 = scaler.fit_transform(X_train1)
# X_test1 = scaler.transform(X_test1)
# Y_train1 = scaler.fit_transform(y_train1)
# Y_test1 = scaler.transform(y_test1)

# # Define the ANN model
# # model1 = keras.Sequential([
# #     keras.layers.Dense(32, activation='relu', input_shape=(X_train1.shape[1],)),
# #     keras.layers.Dense(64, activation='relu'),
# #     keras.layers.Dense(32, activation='relu'),
# #     keras.layers.Dense(16, activation='relu'),
# #     keras.layers.Dense(Y_train1.shape[1], activation='sigmoid')
# # ])

# # Define the neural network model
# model1 = keras.Sequential()
# model1.add(keras.layers.Dense(32, input_dim=X_train1.shape[1], activation='relu'))
# model1.add(keras.layers.Dropout(0.2))
# model1.add(keras.layers.Dense(64, activation='relu'))
# model1.add(keras.layers.Dense(32, activation='relu'))
# model1.add(keras.layers.Dense(16, activation='relu'))
# model1.add(keras.layers.Dense(Y_train1.shape[1], activation='sigmoid'))
# model1.compile(loss='mse', optimizer='adam')

# # Define early stopping callback
# early_stop = EarlyStopping(monitor='accuracy', patience=50)

# # Train the model
# history = model1.fit(X_train1, y_train1, epochs=200, batch_size=30, callbacks=[early_stop])
#///////////////////////////////////////////
# split the dataset into training and testing sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=0.3, random_state=10)

# create a linear regression model
#model1 = LinearRegression()

model1 = DecisionTreeRegressor(max_depth=75)

# fit the model to the training data
model1.fit(X_train1, y_train1)

# # Evaluate the model
# test_loss = model1.evaluate(X_train1, y_train1)
#calculate accuracy
y_pred1 = model1.predict(X_test1)

np.set_printoptions(threshold=np.inf)
y_pred1 = np.round(y_pred1)
y_pred1 = np.absolute(y_pred1)
print(y_pred1)

mse = mean_squared_error(y_test1, y_pred1)
print('Mean squared error:', mse)

r2_scores = r2_score(y_test1, y_pred1)

print("R2 scores:", r2_scores)

# print('Test loss:', test_loss)

dump(model1, 'Player1.joblib')



# split the dataset into training and testing sets
X_train2, X_test2, y_train2, y_test2 = train_test_split(x2, y2, test_size=0.3, random_state=10)

# create a linear regression model
#model2 = LinearRegression()

model2 = DecisionTreeRegressor(max_depth=75)

# fit the model to the training data
model2.fit(X_train2, y_train2)

# make predictions on the testing data
y_pred2 = model2.predict(X_test2)

np.set_printoptions(threshold=np.inf)
y_pred2 = np.round(y_pred2)
y_pred2 = np.absolute(y_pred2)
print(y_pred2)

# # Evaluate the model
# test_loss = model1.evaluate(X_train1, y_train1)
#calculate accuracy

mse = mean_squared_error(y_test2, y_pred2)
print('Mean squared error:', mse)

r2_scores = r2_score(y_test2, y_pred2)

print("R2 scores:", r2_scores)

# print('Test loss:', test_loss)

dump(model2, 'Player2.joblib')

