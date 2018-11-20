import scipy.io
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.regularizers import l1_l2
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from keras import backend as BK
import numpy as np
from sklearn.metrics import mean_absolute_error

#constants and data
mat = scipy.io.loadmat('data/MSdata.mat')
train_x = mat['trainx']
train_y = mat['trainy'].ravel()
test_x = mat['testx']
scaler = StandardScaler()
rescaledX_train = scaler.fit_transform(train_x)
rescaledX_test = scaler.transform(test_x)
min_label = min(train_y)
max_label = max(train_y)+1
features = 90
num_classes = max_label-min_label

def mapping_to_target_range( x, target_min=min_label, target_max=max_label ) :
    x02 = BK.tanh(x) + 1 # x in range(0,2)
    scale = ( target_max-target_min )/2.
    return x02 * scale + target_min


# define base model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(128, input_dim=features, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(1, activation=mapping_to_target_range))

    # Compile model
    model.compile(loss='mae', optimizer='adam', metrics=['mae'])
    return model


# Train model
model = baseline_model()
model.summary()
history = model.fit(rescaledX_train, train_y,
                    batch_size=32,
                    epochs=500,
                    verbose=1,
                    validation_split=0.1)

predicted_train = model.predict(rescaledX_train)
predicted_test = model.predict(rescaledX_test)


int_predict =np.round(predicted_test).ravel()
mse_train = mean_absolute_error(predicted_train, train_y)

np.savetxt("results_6.csv", np.dstack((np.arange(1, int_predict.size+1),int_predict))[0],"%d,%d",header="dataid,prediction")

