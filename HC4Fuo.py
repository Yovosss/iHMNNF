import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np
from datetime import datetime
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
# from keras_tuner.tuners import RandomSearch, Hyperband
from sklearn.model_selection import KFold,StratifiedKFold, train_test_split, cross_val_score

from Data_helper import LoadData, LoadDataEE
from BuildModel import buildModel_DNN, BuildModel_DNN_EE
# from BuildModel import buildModel_DNN, MyHyperModel, BuildModel_DNN_EE

def TrainModel_DNN():

    MEMORY_MB_MAX = 1600000 # maximum memory you can use
    batch_size = 1 # batch size in Level 1
    epochs = 1

    # control the model
    L1_model = True
    L21_model = False
    L22_model = False
    L31_model = False
    L32_model = False

    # Load the data
    ld = LoadData('fuo', 'label4visit')
    X_train_YL1, y_train_YL1, X_train_YL21, y_train_YL21, X_train_YL22, y_train_YL22, X_train_YL31, y_train_YL31, X_train_YL32, y_train_YL32, X_test, y_test = ld.splitData()
    print("=======================================================")
    print("Loading Data is Done!")
    print("=======================================================")

    #######################DNN Level 1########################
    if L1_model:
        print("=======================================================")
        print("Create the DNN Model of L1")
        print("=======================================================")
        # split the data into train and test (optional)
        # y_train_YL1 = to_categorical(y_train_YL1, num_classes=None)
        X_train, X_test, y_train, y_test = train_test_split(X_train_YL1, y_train_YL1, test_size=0.2, random_state=1, stratify=y_train_YL1)
        model = buildModel_DNN(X_train.shape[1], 2, 8, 500, dropout=0.25)
        model.fit(X_train, y_train,
                  validation_data=(X_test, y_test),
                  epochs=epochs,
                  verbose=2,
                  batch_size=batch_size)

        # hypermodel = MyHyperModel(X_train.shape[1], 2, dropout=0.25)
        # tuner = Hyperband(hypermodel,
        #                   objective='val_accuracy',
        #                   max_epochs=10,
        #                   factor=3,
        #                   overwrite = True,
        #                   directory=os.path.join(os.path.dirname(__file__), 'model_dl'),
        #                   project_name='20211012FUO')
        # tuner.search_space_summary()
        # tuner.search(X_train, y_train,
        #      batch_size=batch_size,
        #      epochs=10,
        #      validation_data=(X_test, y_test))
        # # Get the optimal hyperparameters
        # best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

        # # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
        # model = tuner.hypermodel.build(best_hps)
        # history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

        # val_acc_per_epoch = history.history['val_accuracy']
        # best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        # print('Best epoch: %d' % (best_epoch,))

        # hypermodel = tuner.hypermodel.build(best_hps)
        # # Retrain the model
        # hypermodel.fit(X_train, y_train, epochs=best_epoch, validation_data=(X_test, y_test))

        # hypermodel.save('/mnt/data/wzx/jupyter_notebook/HC4FUO/model_dl/model4L1_sparse_loss.h5')
        # hypermodel.summary()
        # eval_result = hypermodel.evaluate(X_test, y_test)
        # print("[test loss, test accuracy]:", eval_result)
    #######################DNN Level 2-1########################
    if L21_model:
        print("=======================================================")
        print("Create the DNN Model of L2-1")
        print("=======================================================")
        # split the data into train and test (optional)
        # y_train_YL21 = to_categorical(y_train_YL21, num_classes=None)
        X_train, X_test, y_train, y_test = train_test_split(X_train_YL21, y_train_YL21, test_size=0.2, random_state=0, stratify=y_train_YL21)
        # model = buildModel_DNN(X_train.shape[1], 3, 8, 500, dropout=0.25)
        # model.fit(X_train, y_train,
        #           validation_data=(X_test, y_test),
        #           epochs=epochs,
        #           verbose=2,
        #           batch_size=batch_size)

        hypermodel = MyHyperModel(X_train.shape[1], 3, dropout=0.25)
        tuner = Hyperband(hypermodel,
                          objective='val_accuracy',
                          max_epochs=10,
                          factor=3,
                          overwrite = True,
                          directory=os.path.join(os.path.dirname(__file__), 'model_dl'),
                          project_name='20211012FUO')
        tuner.search_space_summary()
        tuner.search(X_train, y_train,
             batch_size=batch_size,
             epochs=10,
             validation_data=(X_test, y_test))
        # Get the optimal hyperparameters
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

        # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

        val_acc_per_epoch = history.history['val_accuracy']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        print('Best epoch: %d' % (best_epoch,))

        hypermodel = tuner.hypermodel.build(best_hps)
        # Retrain the model
        hypermodel.fit(X_train, y_train, epochs=best_epoch, validation_data=(X_test, y_test))

        hypermodel.save('/mnt/data/wzx/jupyter_notebook/HC4FUO/model_dl/model4L21.h5')
        hypermodel.summary()
        # eval_result = hypermodel.evaluate(X_test, y_test)
        # print("[test loss, test accuracy]:", eval_result)
    #######################DNN Level 2-2########################
    if L22_model:
        print("=======================================================")
        print("Create the DNN Model of L2-2")
        print("=======================================================")
        # split the data into train and test (optional)
        # y_train_YL22 = to_categorical(y_train_YL22, num_classes=None)
        X_train, X_test, y_train, y_test = train_test_split(X_train_YL22, y_train_YL22, test_size=0.2, random_state=0, stratify=y_train_YL22)
        # model = buildModel_DNN(X_train.shape[1], 2, 8, 500, dropout=0.25)
        # model.fit(X_train, y_train,
        #           validation_data=(X_test, y_test),
        #           epochs=epochs,
        #           verbose=2,
        #           batch_size=batch_size)

        hypermodel = MyHyperModel(X_train.shape[1], 2, dropout=0.25)
        tuner = Hyperband(hypermodel,
                          objective='val_accuracy',
                          max_epochs=10,
                          factor=3,
                          overwrite = True,
                          directory=os.path.join(os.path.dirname(__file__), 'model_dl'),
                          project_name='20211012FUO')
        tuner.search_space_summary()
        tuner.search(X_train, y_train,
             batch_size=batch_size,
             epochs=10,
             validation_data=(X_test, y_test))
        # Get the optimal hyperparameters
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

        # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

        val_acc_per_epoch = history.history['val_accuracy']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        print('Best epoch: %d' % (best_epoch,))

        hypermodel = tuner.hypermodel.build(best_hps)
        # Retrain the model
        hypermodel.fit(X_train, y_train, epochs=best_epoch, validation_data=(X_test, y_test))

        hypermodel.save('/mnt/data/wzx/jupyter_notebook/HC4FUO/model_dl/model4L22_sparse_loss.h5')
        hypermodel.summary()
        # eval_result = hypermodel.evaluate(X_test, y_test)
        # print("[test loss, test accuracy]:", eval_result)
    #######################DNN Level 3-1########################
    if L31_model:
        print("=======================================================")
        print("Create the DNN Model of L3-1")
        print("=======================================================")
        # split the data into train and test (optional)
        # y_train_YL31 = to_categorical(y_train_YL31, num_classes=None)
        X_train, X_test, y_train, y_test = train_test_split(X_train_YL31, y_train_YL31, test_size=0.2, random_state=0, stratify=y_train_YL31)
        # model = buildModel_DNN(X_train.shape[1], 2, 8, 500, dropout=0.25)
        # model.fit(X_train, y_train,
        #           validation_data=(X_test, y_test),
        #           epochs=epochs,
        #           verbose=2,
        #           batch_size=batch_size)

        hypermodel = MyHyperModel(X_train.shape[1], 2, dropout=0.25)
        tuner = Hyperband(hypermodel,
                          objective='val_accuracy',
                          max_epochs=10,
                          factor=3,
                          overwrite = True,
                          directory=os.path.join(os.path.dirname(__file__), 'model_dl'),
                          project_name='20211012FUO')
        tuner.search_space_summary()
        tuner.search(X_train, y_train,
             batch_size=batch_size,
             epochs=10,
             validation_data=(X_test, y_test))
        # Get the optimal hyperparameters
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

        # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

        val_acc_per_epoch = history.history['val_accuracy']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        print('Best epoch: %d' % (best_epoch,))

        hypermodel = tuner.hypermodel.build(best_hps)
        # Retrain the model
        hypermodel.fit(X_train, y_train, epochs=best_epoch, validation_data=(X_test, y_test))

        hypermodel.save('/mnt/data/wzx/jupyter_notebook/HC4FUO/model_dl/model4L31.h5')
        hypermodel.summary()
        # eval_result = hypermodel.evaluate(X_test, y_test)
        # print("[test loss, test accuracy]:", eval_result)
    #######################DNN Level 3-2########################
    if L32_model:
        print("=======================================================")
        print("Create the DNN Model of L3-2")
        print("=======================================================")
        # split the data into train and test (optional)
        # y_train_YL32 = to_categorical(y_train_YL32, num_classes=None)
        X_train, X_test, y_train, y_test = train_test_split(X_train_YL32, y_train_YL32, test_size=0.1, random_state=0, stratify=y_train_YL32)
        # model = buildModel_DNN(X_train.shape[1], 2, 8, 500, dropout=0.25)
        # model.fit(X_train, y_train,
        #           validation_data=(X_test, y_test),
        #           epochs=epochs,
        #           verbose=2,
        #           batch_size=batch_size)

        hypermodel = MyHyperModel(X_train.shape[1], 2, dropout=0.25)
        tuner = Hyperband(hypermodel,
                          objective='val_accuracy',
                          max_epochs=10,
                          factor=3,
                          overwrite = True,
                          directory=os.path.join(os.path.dirname(__file__), 'model_dl'),
                          project_name='20211012FUO')
        tuner.search_space_summary()
        tuner.search(X_train, y_train,
             batch_size=batch_size,
             epochs=10,
             validation_data=(X_test, y_test))
        # Get the optimal hyperparameters
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

        # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

        val_acc_per_epoch = history.history['val_accuracy']
        best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
        print('Best epoch: %d' % (best_epoch,))

        hypermodel = tuner.hypermodel.build(best_hps)
        # Retrain the model
        hypermodel.fit(X_train, y_train, epochs=best_epoch, validation_data=(X_test, y_test))

        hypermodel.save('/mnt/data/wzx/jupyter_notebook/HC4FUO/model_dl/model4L32.h5')
        hypermodel.summary()
        # eval_result = hypermodel.evaluate(X_test, y_test)
        # print("[test loss, test accuracy]:", eval_result)

def TrainModel_DNN_EE():

    working_path = '.'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

    # control the model
    L1_model = False
    L21_model = True
    L22_model = False
    L31_model = False
    L32_model = False

    # Load the data
    ld = LoadDataEE('fuo', 'label4visit')
    X_train_YL1, y_train_YL1, X_train_YL21, y_train_YL21, X_train_YL22, y_train_YL22, X_train_YL31, y_train_YL31, X_train_YL32, y_train_YL32, X_test, y_test, col_range_cat, col_range_val, embedding_size = ld.splitDataEE()
    print("=======================================================")
    print("Loading Data is Done!")
    print("=======================================================")

    if L1_model:
        print("=======================================================")
        print("Create the DNN_EE Model of L1")
        print("=======================================================")
        # split the data into train and test (optional)
        X_train, X_val, y_train, y_val = train_test_split(X_train_YL1, y_train_YL1, test_size=0.2, random_state=1, stratify=y_train_YL1)
        BuildModel_DNN_EE(X_train, X_val, y_train, y_val, col_range_cat, col_range_val, embedding_size, 2, working_path, timestamp)

    if L21_model:
        print("=======================================================")
        print("Create the DNN_EE Model of L21")
        print("=======================================================")
        # split the data into train and test (optional)
        X_train, X_val, y_train, y_val = train_test_split(X_train_YL21, y_train_YL21, test_size=0.2, random_state=1, stratify=y_train_YL21)
        BuildModel_DNN_EE(X_train, X_val, y_train, y_val, col_range_cat, col_range_val, embedding_size, 3, working_path, timestamp)

if __name__ == '__main__':

    # Train and validate the model of pure DNN
    TrainModel_DNN()

    # Train and validate the model of DNN with embedding layers
    # TrainModel_DNN_EE()