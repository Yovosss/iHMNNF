
import os
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sqlalchemy import Table, select, text, distinct, types
from sqlalchemy.dialects.oracle import BFILE, BLOB, CHAR, CLOB, DATE, DOUBLE_PRECISION, FLOAT, INTERVAL, LONG, NCLOB, NUMBER, NVARCHAR, NVARCHAR2, RAW, TIMESTAMP, VARCHAR, VARCHAR2

from config.orcl import db_connection

# The input data for DNN model
VAL_DATA_DIR = 'data/p1_feature_val.csv'
CAT_DATA_DIR = 'data/p1_feature_cat.csv'

# The input data for DNN_EE model
VAL_DATA_DIR_EE = 'data/p2_feature_val.csv'
CAT_DATA_DIR_EE = 'data/p2_feature_cat.csv'

HERE = os.path.dirname(__file__)

# read the label data
class LoadData():
    """
    mainly preprocess the data of p1_feature_val.csv and p1_feature_cat.csv, which will be the input
    of DNN model
    """

    def __init__(self, db_name, label_table):
        self.db_conn = db_connection(db_name)
        self.label = Table(label_table, self.db_conn.metadata, autoload=True)

    def getlabel(self):
        s = select([self.label.c.visit_record_id_new_1,
                    self.label.c.source_union,
                    self.label.c.label1,
                    self.label.c.label2,
                    self.label.c.label3,
                    self.label.c.label4,
                    self.label.c.label5]).where(self.label.c.source_union != None)
        label = self.db_conn.conn.execute(s).fetchall()
        label = pd.DataFrame(label, columns=['visit_record_id_new_1', 'source_union', 'label1','label2','label3','label4','label5'])

        # drop the duplicated label and orginal index
        label = label.loc[label['source_union'].isin(['1-1','1-2','1-3','2-1-1','2-1-2','2-2-1','2-2-2'])]
        label.sort_values(by=['visit_record_id_new_1'], ascending=True, inplace=True)
        label.drop_duplicates(subset=['visit_record_id_new_1', 'source_union'], keep='first', inplace=True)
        label.reset_index(inplace=True)
        label.drop(columns=['index'], inplace=True)
        print("=======================================================")
        print("The shape of original label is {}".format(label.shape))
        print("The distribution of visit of YL1 is {}".format(label['label1'].value_counts().sort_index().values))
        print("The distribution of visit of YL2-1 is {}".format(label['label2'].value_counts().sort_index().values))
        print("The distribution of visit of YL2-2 is {}".format(label['label3'].value_counts().sort_index().values))
        print("The distribution of visit of YL3-1 is {}".format(label['label4'].value_counts().sort_index().values))
        print("The distribution of visit of YL3-2 is {}".format(label['label5'].value_counts().sort_index().values))
        print("=======================================================")

        return label

    def data_cat_preprocessing(self, data_cat):
        # process the categorical missing data
        data_cat.reset_index(inplace=True)
        col_cat = data_cat.columns.values.tolist()
        imp = SimpleImputer(strategy="most_frequent")
        data_cat = imp.fit_transform(data_cat)
        data_cat = pd.DataFrame(data_cat, columns=col_cat)
        data_cat.set_index(['visit_record_id_new_1'], inplace=True)

        # transform the categorical data into different value
        oe = preprocessing.OrdinalEncoder()
        data_cat = oe.fit_transform(data_cat)
        # transform into onehot-encoder format
        ohe = preprocessing.OneHotEncoder()
        data_cat = ohe.fit_transform(data_cat).toarray()
        data_cat = pd.DataFrame(data_cat)

        return data_cat

    def data_val_preprocessing(self, data_val):
        # process the numbered missing data
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        data_val = imp.fit_transform(data_val)
        data_val = pd.DataFrame(data_val)

        return data_val

    def load_data(self):
        # read the number and category data
        label = self.getlabel()
        data_val = pd.read_csv(os.path.join(os.path.dirname(__file__), VAL_DATA_DIR_EE))
        data_cat = pd.read_csv(os.path.join(os.path.dirname(__file__), CAT_DATA_DIR_EE))
        data_val.drop(columns=['Unnamed: 0'], inplace=True)
        data_cat.drop(columns=['Unnamed: 0'], inplace=True)
        print("=======================================================")
        print("The shape of original data_val is {}".format(data_val.shape))
        print("The shape of original data_cat is {}".format(data_cat.shape))
        print("=======================================================")

        # get the data in  label
        data_val = label.merge(data_val, how='left', on='visit_record_id_new_1')
        data_val.drop(columns=['source_union', 'label1','label2','label3','label4','label5'], inplace=True)
        data_val.set_index(['visit_record_id_new_1'], inplace=True)

        data_cat = label.merge(data_cat, how='left', on='visit_record_id_new_1')
        data_cat.drop(columns=['source_union','label1','label2','label3','label4','label5'], inplace=True)
        data_cat.set_index(['visit_record_id_new_1'], inplace=True)

        # deal with the abnormal value
        data_cat = data_cat.applymap(lambda x: np.NaN if str(x)=='None' else x)
        data_val = data_val.applymap(lambda x: np.NaN if str(x)=='None' else x)
        data_val = data_val.applymap(lambda x: 0 if str(x)=='0.0.' else x)
        data_val = data_val.applymap(lambda x: 0 if str(x)=='0-1.' else x)
        data_val = data_val.applymap(lambda x: 0 if str(x)=='0-1' else x)
        data_val = data_val.applymap(lambda x: 0 if str(x)=='0。0' else x)
        data_val =pd.DataFrame(data_val, dtype=np.float64)

        # preprocess the data_val and data_cat
        data_cat = self.data_cat_preprocessing(data_cat)
        data_val = self.data_val_preprocessing(data_val)
        data = pd.concat([data_cat, data_val], axis=1)

        # standardize the data
        scaler = preprocessing.StandardScaler()
        data = scaler.fit_transform(data)
        data = pd.DataFrame(data)
        print("=======================================================")
        print("The shape of data after preprocessing is {}".format(data.shape))
        print("=======================================================")

        # split the data into train and test randomly according the ratio of fined-label
        y_test = pd.DataFrame([])
        for i in label['source_union'].value_counts().index:
            y_test = y_test.append(label.loc[label['source_union']==i].sample(frac=0.2))
        X_test = data[data.index.isin(y_test.index)]
        # the train data
        X_train = data[~data.index.isin(y_test.index)]
        y_train = label[~label.index.isin(y_test.index)]
        print("=======================================================")
        print("The shape of X_train is {}".format(X_train.shape))
        print("The shape of X_test is {}".format(X_test.shape))
        print("=======================================================")

        return X_train, y_train, X_test, y_test

    def splitData(self):

        X_train, y_train, X_test, y_test = self.load_data()

        # get the data and label of infectious and non-infectious(YL1)
        y_train_YL1 = y_train.loc[y_train['label1'].notnull(), ['label1']]
        y_train_YL1.rename(columns={'label1': 'label'}, inplace=True)
        X_train_YL1 = X_train[X_train.index.isin(y_train_YL1.index)]
        X_train_YL1 = X_train_YL1.values
        y_train_YL1 = y_train_YL1['label'].values
        y_train_YL1 = y_train_YL1.astype(np.int32)

        # get the data and label of bacterial, viral and fungal(YL21)
        y_train_YL21 = y_train.loc[y_train['label2'].notnull(), ['label2']]
        y_train_YL21.rename(columns={'label2': 'label'}, inplace=True)
        X_train_YL21 = X_train[X_train.index.isin(y_train_YL21.index)]
        X_train_YL21 = X_train_YL21.values
        y_train_YL21 = y_train_YL21['label'].values
        y_train_YL21 = y_train_YL21.astype(np.int32)

        # get the data and label of Neo- and NIID(YL22)
        y_train_YL22 = y_train.loc[y_train['label3'].notnull(), ['label3']]
        y_train_YL22.rename(columns = {'label3': 'label'}, inplace=True)
        X_train_YL22 = X_train[X_train.index.isin(y_train_YL22.index)]
        print("=======================================================")
        print("The distribution of visit of y_train_YL22 is {}".format(y_train_YL22['label'].value_counts().sort_index().values))
        print("=======================================================")
        X_train_YL22 = X_train_YL22.values
        y_train_YL22 = y_train_YL22['label'].values
        y_train_YL22 = y_train_YL22.astype(np.int32)

        # get the data and label of AID and AIF(YL31)
        y_train_YL31 = y_train.loc[y_train['label4'].notnull(), ['label4']]
        y_train_YL31.rename(columns={'label4': 'label'}, inplace=True)
        X_train_YL31 = X_train[X_train.index.isin(y_train_YL31.index)]
        X_train_YL31 = X_train_YL31.values
        y_train_YL31 = y_train_YL31['label'].values
        y_train_YL31 = y_train_YL31.astype(np.int32)

        # get the data and label of HM and SM(YL32)
        y_train_YL32 = y_train.loc[y_train['label5'].notnull(), ['label5']]
        y_train_YL32.rename(columns={'label5': 'label'}, inplace=True)
        X_train_YL32 = X_train[X_train.index.isin(y_train_YL32.index)]
        print("=======================================================")
        print("The distribution of visit of y_train_YL32 is {}".format(y_train_YL32['label'].value_counts().sort_index().values))
        print("=======================================================")
        X_train_YL32 = X_train_YL32.values
        y_train_YL32 = y_train_YL32['label'].values
        y_train_YL32 = y_train_YL32.astype(np.int32)

        return X_train_YL1, y_train_YL1, X_train_YL21, y_train_YL21, X_train_YL22, y_train_YL22, X_train_YL31, y_train_YL31, X_train_YL32, y_train_YL32, X_test, y_test

class LoadDataEE(LoadData):

    def __init__(self, db_name, label_table):
        super().__init__(db_name, label_table)

    def data_cat_preprocessing_ee(self, data):

        # process the categorical missing data
        col_cat = data.columns.values.tolist()
        # fill the NaN with '0'
        data = data.applymap(lambda x: '0' if x=='nan' else x)

        # transform the categorical data into different value
        oe = preprocessing.OrdinalEncoder()
        data = oe.fit_transform(data)
        data = pd.DataFrame(data, columns=col_cat)

        return data

    def data_val_preprocessing_ee(self, data):
        # process the numbered missing data
        col_val = data.columns.values.tolist()
        # fill the missing value with "mean" value
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        data = imp.fit_transform(data)

        # standardize the data
        scaler = preprocessing.StandardScaler()
        data = scaler.fit_transform(data)
        data = pd.DataFrame(data, columns=col_val)

        return data

    def cal_embedding_size(self, data):

        col_list = data.columns.values.tolist()
        cat_size = [(c, data[c].nunique()) for c in col_list]
        embedding_size = [(c, min(50, (c+1)//2)) for _, c in cat_size]

        return embedding_size


    def LoadDataEE(self):

        # read the number and category data
        label = self.getlabel()
        data_val = pd.read_csv(os.path.join(os.path.dirname(__file__), VAL_DATA_DIR_EE))
        data_cat = pd.read_csv(os.path.join(os.path.dirname(__file__), CAT_DATA_DIR_EE))
        data_val.drop(columns=['Unnamed: 0'], inplace=True)
        data_cat.drop(columns=['Unnamed: 0'], inplace=True)
        print("=======================================================")
        print("The shape of original data_val is {}".format(data_val.shape))
        print("The shape of original data_cat is {}".format(data_cat.shape))
        print("=======================================================")

        # get the data in  label
        data_val = label.merge(data_val, how='left', on='visit_record_id_new_1')
        data_val.drop(columns=['source_union', 'label1','label2','label3','label4','label5'], inplace=True)
        data_val.set_index(['visit_record_id_new_1'], inplace=True)

        data_cat = label.merge(data_cat, how='left', on='visit_record_id_new_1')
        data_cat.drop(columns=['source_union','label1','label2','label3','label4','label5'], inplace=True)
        data_cat.set_index(['visit_record_id_new_1'], inplace=True)

        # deal with the abnormal value
        data_cat = data_cat.applymap(lambda x: np.NaN if str(x)=='None' else x)
        data_val = data_val.applymap(lambda x: np.NaN if str(x)=='None' else x)
        data_val = data_val.applymap(lambda x: 0 if str(x)=='0.0.' else x)
        data_val = data_val.applymap(lambda x: 0 if str(x)=='0-1.' else x)
        data_val = data_val.applymap(lambda x: 0 if str(x)=='0-1' else x)
        data_val = data_val.applymap(lambda x: 0 if str(x)=='0。0' else x)
        data_val =pd.DataFrame(data_val, dtype=np.float64)
        data_cat = data_cat.astype(str)

        data_cat = self.data_cat_preprocessing_ee(data_cat)
        data_val = self.data_val_preprocessing_ee(data_val)

        embedding_size = self.cal_embedding_size(data_cat)

        data = pd.concat([data_cat, data_val], axis=1)

        col_range_cat = data_cat.shape
        col_range_val = data_val.shape

        # split the data into train and test randomly according the ratio of fined-label
        y_test = pd.DataFrame([])
        for i in label['source_union'].value_counts().index:
            y_test = y_test.append(label.loc[label['source_union']==i].sample(frac=0.2))
        X_test = data[data.index.isin(y_test.index)]
        # the train data
        X_train = data[~data.index.isin(y_test.index)]
        y_train = label[~label.index.isin(y_test.index)]
        print("=======================================================")
        print("The shape of X_train is {}".format(X_train.shape))
        print("The shape of X_test is {}".format(X_test.shape))
        print("=======================================================")

        #split the data according the hierarchical structure of FUO

        return X_train, y_train, X_test, y_test, col_range_cat, col_range_val, embedding_size

    def splitDataEE(self):

        X_train, y_train, X_test, y_test, col_range_cat, col_range_val, embedding_size = self.LoadDataEE()

        # get the data and label of infectious and non-infectious(YL1)
        y_train_YL1 = y_train.loc[y_train['label1'].notnull(), ['label1']]
        y_train_YL1.rename(columns={'label1': 'label'}, inplace=True)
        X_train_YL1 = X_train[X_train.index.isin(y_train_YL1.index)]
        X_train_YL1 = X_train_YL1.values
        y_train_YL1 = y_train_YL1['label'].values
        y_train_YL1 = y_train_YL1.astype(np.int32)

        # get the data and label of bacterial, viral and fungal(YL21)
        y_train_YL21 = y_train.loc[y_train['label2'].notnull(), ['label2']]
        y_train_YL21.rename(columns={'label2': 'label'}, inplace=True)
        X_train_YL21 = X_train[X_train.index.isin(y_train_YL21.index)]
        X_train_YL21 = X_train_YL21.values
        y_train_YL21 = y_train_YL21['label'].values
        y_train_YL21 = y_train_YL21.astype(np.int32)

        # get the data and label of Neo- and NIID(YL22)
        y_train_YL22 = y_train.loc[y_train['label3'].notnull(), ['label3']]
        y_train_YL22.rename(columns = {'label3': 'label'}, inplace=True)
        X_train_YL22 = X_train[X_train.index.isin(y_train_YL22.index)]
        print("=======================================================")
        print("The distribution of visit of y_train_YL22 is {}".format(y_train_YL22['label'].value_counts().sort_index().values))
        print("=======================================================")
        X_train_YL22 = X_train_YL22.values
        y_train_YL22 = y_train_YL22['label'].values
        y_train_YL22 = y_train_YL22.astype(np.int32)

        # get the data and label of AID and AIF(YL31)
        y_train_YL31 = y_train.loc[y_train['label4'].notnull(), ['label4']]
        y_train_YL31.rename(columns={'label4': 'label'}, inplace=True)
        X_train_YL31 = X_train[X_train.index.isin(y_train_YL31.index)]
        X_train_YL31 = X_train_YL31.values
        y_train_YL31 = y_train_YL31['label'].values
        y_train_YL31 = y_train_YL31.astype(np.int32)

        # get the data and label of HM and SM(YL32)
        y_train_YL32 = y_train.loc[y_train['label5'].notnull(), ['label5']]
        y_train_YL32.rename(columns={'label5': 'label'}, inplace=True)
        X_train_YL32 = X_train[X_train.index.isin(y_train_YL32.index)]
        print("=======================================================")
        print("The distribution of visit of y_train_YL32 is {}".format(y_train_YL32['label'].value_counts().sort_index().values))
        print("=======================================================")
        X_train_YL32 = X_train_YL32.values
        y_train_YL32 = y_train_YL32['label'].values
        y_train_YL32 = y_train_YL32.astype(np.int32)

        return X_train_YL1, y_train_YL1, X_train_YL21, y_train_YL21, X_train_YL22, y_train_YL22, X_train_YL31, y_train_YL31, X_train_YL32, y_train_YL32, X_test, y_test, col_range_cat, col_range_val, embedding_size

class LoadDataMM():
    """
    Load 'data.npz' and 'fold.npz' for hierarchical multi-modality model training and testing.
    In 'data.npz':
        Required: 'input', 'masking', 'timestamp', 'static_val', 'static_cat', 'label'
        Shape: (n_samples,)
    In 'fold.npz':
        Required: 'mean_hy$[0-4]$_temporal' and 'std_hy$[0-4]$_temporal', 'mean_hy$[0-4]$_static', 'std_hy$[0-4]$_static'
        Shape: (n_splits, 3)
    """
    def __init__(self, data_path, label_name, max_steps=None, max_timestamp=None):
        
        self._input_dim_t = None
        self._input_dim_s_val = None
        self._input_dim_s_cat = None
        self._folds = None
        self._output_activation = None
        self._loss_function = None
        self._embedding_size = None

        self._data_file = os.path.join(data_path, 'data.npz')
        self._fold_file = os.path.join(data_path, 'fold.npz')
        self._load_data(label_name)
        self._max_steps = max_steps
        self._max_timestamp = max_timestamp

    def _load_data(self, label_name):
        
        if not os.path.exists(self._data_file):
            raise ValueError('Data file does not exist...')
        if not os.path.exists(self._fold_file):
            raise ValueError('Fold file does not exist...')

        data = np.load(self._data_file, allow_pickle=True)
        fold = np.load(self._fold_file, allow_pickle=True)

        self._data = {}
        for i in ['input', 'masking', 'timestamp', 'static_val', 'static_cat']:
            self._data[i] = data[i]

        self._data['label'] = data['label'][:,[label_name + 1]].astype(int)

        for i in ['hy', 'mean-temporal', 'std-temporal', 'mean-static', 'std-static']:
            self._data[i] = fold[i + '-' + str(label_name)]

        # cal the embedding size for categorical variables
        unique_num = [len(np.unique(self._data['static_cat'][:, i])) for i in range(self._data['static_cat'].shape[1])]
        self._embedding_size = [(c, min(50, (c+1)//2)) for c in unique_num]
        
        self._input_dim_t = self._data['input'][0].shape[-1]
        self._input_dim_s_val = self._data['static_val'].shape[1]
        self._input_dim_s_cat = self._data['static_cat'].shape[1]
        self._folds = self._data['hy'].shape[0]
        self._loss_function = 'binary_crossentropy'   # binary_crossentropy sparse_categorical_crossentropy
        self._output_activation = 'sigmoid'  # sigmoid softmax
        self._nclasses = len(np.unique(self._data['label']))

    def _fillnan(self, x, mean):
        """
        Args:
            x: A np.array of static variables with shape (batch_size, d)
            mean: A np.array of mean value of each variables with shape (d,)
        
        Returns:
            same shape as x with filled with mean value of training set data 
        """

        for i in range(x.shape[1]):
            x[:, i][np.isnan(x[:, i])] = mean[i]

        return x

    def _rescale(self, x, mean, std):
        """
        Args:
            x: A np.array of several np.array with shape (t_i, d).
            mean: A np.array of shape (d,).
            std: A np.array of shape (d,).

        Returns:
            Same shape as x with rescaled values.
        """
        if x[0].ndim == 1:
            return np.asarray([(xx - mean) / std for xx in x])
        elif x[0].ndim == 2:
            return np.asarray([(xx - mean[np.newaxis, :]) / std[np.newaxis, :] for xx in x])

    def _filter(self, ts, max_timestamp=None, max_timesteps=None):
        """
        Args:
            ts: A np.array of n np.array with shape (t_i, d).
            max_timestamp: an Integer > 0 or None.
            max_timesteps: an Integer > 0 or None.

        Returns:
            A np.array of n Integers. Its i-th element (x_i) indicates that
                we will take the first x_i numbers from i-th data sample. 
        """
        if max_timestamp is None:
            ret = np.asarray([len(tt) for tt in ts])
        else:
            ret = np.asarray([np.sum(tt - tt[0] <= max_timestamp) for tt in ts])
        if max_timesteps is not None:
            ret = np.minimum(ret, max_timesteps)
        return ret
    
    def _pad(self, x, lens):
        """
        Args:
            x: A np.array of n np.array with shape (t_i, d).
            lens: A np.array of n Integers > 0.

        Returns:
            A np.array of shape (n, t, d), where t = min(max_length, max(lens))
        """
        n = len(x) #32个sample
        t = max(lens) # 列表内的最大值也就是最大的时间点个数，即48小时内最长的时间跨度
        d = 1 if x[0].ndim == 1 else x[0].shape[1] # 136
        ret = np.zeros([n, t, d], dtype=float) # [32,t,136]
        if x[0].ndim == 1:
            for i, xx in enumerate(x):
                ret[i, :lens[i]] = xx[:lens[i], np.newaxis]
        else:
            for i, xx in enumerate(x):
                # 取每一个样本的前lens(i)个时间点的所有列数据
                ret[i, :lens[i]] = xx[:lens[i]]
        return ret

    def _split_cat(self, x):

        inputs = []
        inputs += x[:-1]
        for i in range(self._input_dim_s_cat):
            x_cat = x[-1][..., [i]]
            inputs.append(x_cat)
        return inputs

    def _get_generator(self, i, i_fold, shuffle, batch_size, return_targets):
        if not return_targets and shuffle:
            raise ValueError('Do not shuffle when targets are not returned.')
        # The mean/std used in validation/test fold should also be from the training fold
        fold = np.copy(self._data['hy'][i_fold][i])
        mean_temporal = self._data['mean-temporal'][i_fold][0]
        std_temporal = self._data['std-temporal'][i_fold][0]
        mean_static = self._data['mean-static'][i_fold][0]
        std_static = self._data['std-static'][i_fold][0]

        folds = len(fold)  # The number of training sample

        def _generator():
            while True:
                if shuffle:
                    np.random.shuffle(fold)  #shffule the data before split it into batch
                batch_from = 0
                while batch_from < folds:
                    # extract the data by batch_size
                    batch_fold = fold[batch_from: batch_from + batch_size]
                    inputs = [self._data[s][batch_fold] for s in ['input', 'masking', 'timestamp', 'static_val', 'static_cat']]

                    inputs[0] = self._rescale(inputs[0], mean_temporal, std_temporal)

                    inputs[3] = self._fillnan(inputs[3], mean_static)
                    inputs[3] = self._rescale(inputs[3], mean_static, std_static)

                    lens = self._filter(inputs[2], self._max_timestamp, self._max_steps)
                    inputs[:3] = [self._pad(x, lens) for x in inputs[:3]]

                    inputs = self._split_cat(inputs)

                    targets = self._data['label'][batch_fold]

                    yield (inputs, targets)
                    batch_from += batch_size
                    print('.', end='')
        
        def _input_generator():
            for inputs, _ in _generator():
                yield inputs

        if not return_targets:
            return _input_generator()
        return _generator()

    def training_generator(self, i_fold, batch_size):
        return self._get_generator(i=0, i_fold=i_fold, shuffle=True, 
                                   batch_size=batch_size, return_targets=True)

    def validation_generator(self, i_fold, batch_size):
            return self._get_generator(i=1, i_fold=i_fold, shuffle=False,
                                    batch_size=batch_size, return_targets=True)

    def testing_generator(self, i_fold, batch_size):
        return self._get_generator(i=2, i_fold=i_fold, shuffle=False,
                                   batch_size=batch_size, return_targets=True)

    def _steps(self, i, i_fold, batch_size):
        return (self._data['hy'][i_fold][i].size - 1) // batch_size + 1

    def training_steps(self, i_fold, batch_size):
        return self._steps(i=0, i_fold=i_fold, batch_size=batch_size)

    def validation_steps(self, i_fold, batch_size):
        return self._steps(i=1, i_fold=i_fold, batch_size=batch_size)

    def testing_steps(self, i_fold, batch_size):
        return self._steps(i=2, i_fold=i_fold, batch_size=batch_size)

    def training_y(self, i_fold):
        return self._data['label'][self._data['hy'][i_fold][0]]

    def validation_y(self, i_fold):
        return self._data['label'][self._data['hy'][i_fold][1]]

    def testing_y(self, i_fold):
        return self._data['label'][self._data['hy'][i_fold][2]]
    
    def training_generator_x(self, i_fold, batch_size):
        return self._get_generator(i=0, i_fold=i_fold, shuffle=False,
                                   batch_size=batch_size, return_targets=False)

    def validation_generator_x(self, i_fold, batch_size):
        return self._get_generator(i=1, i_fold=i_fold, shuffle=False,
                                   batch_size=batch_size, return_targets=False)

    def testing_generator_x(self, i_fold, batch_size):
        return self._get_generator(i=2, i_fold=i_fold, shuffle=False,
                                   batch_size=batch_size, return_targets=False)

    @property
    def folds(self):
        return self._folds

    @property
    def input_dim_t(self):
        return self._input_dim_t
    
    @property
    def input_dim_s_val(self):
        return self._input_dim_s_val
    
    @property
    def input_dim_s_cat(self):
        return self._input_dim_s_cat
    
    @property
    def embedding_size(self):
        return self._embedding_size

    @property
    def output_dim(self):
        return self._nclasses

    @property
    def loss_function(self):
        return self._loss_function

    @property
    def output_activation(self):
        return self._output_activation
        
class LoadDataTrial():
    """
    Load 'data.npz' and 'fold.npz' for multi-type model training and testing.
    In 'data.npz':
        Required: 'input', 'masking', 'timestamp', 'static_val', 'static_cat', 'label'
        Shape: (n_samples,)
    In 'fold.npz':
        Required: 'mean_hy$[0-4]$_temporal' and 'std_hy$[0-4]$_temporal', 'mean_hy$[0-4]$_static', 'std_hy$[0-4]$_static'
        Shape: (n_splits, 3)
    """

    def __init__(self, data_path, model_type, label_name, max_timestep, max_timestamp):
        
        self._input_dim_t = None
        self._input_dim_s_val = None
        self._input_dim_s_cat = None
        self._folds = None
        self._embedding_size = None
        self.model_type = model_type
        self._max_timestep = max_timestep
        self._max_timestamp = max_timestamp

        self._data_file = os.path.join(data_path, 'data.npz')
        self._fold_file = os.path.join(data_path, 'fold.npz')
        self._load_data(label_name)

    def _load_data(self, label_name):
        """
        Label_name: 
            the index of different label ranging from [0-4]
        Return:
            this func is for DNN model, which only including the structured data
        """
        if not os.path.exists(self._data_file):
            raise ValueError('Data file does not exist...')
        if not os.path.exists(self._fold_file):
            raise ValueError('Fold file does not exist...')

        data = np.load(self._data_file, allow_pickle=True)
        fold = np.load(self._fold_file, allow_pickle=True)

        self._data = {}
        for i in ['input', 'masking', 'timestamp', 'static_val', 'static_cat', 'static_cat_onehot']:
            self._data[i] = data[i]

        self._data['label'] = data['label'][:,label_name + 1]
        
        if label_name == -1:
            for i in ['hy', 'mean-temporal', 'std-temporal', 'mean-static', 'std-static']:
                self._data[i] = fold[i + '-' + str(label_name+1)]
        else:
            for i in ['hy', 'mean-temporal', 'std-temporal', 'mean-static', 'std-static']:
                self._data[i] = fold[i + '-' + str(label_name)]

        # cal the embedding size for categorical variables
        unique_num = [len(np.unique(self._data['static_cat'][:, i])) for i in range(self._data['static_cat'].shape[1])]
        self._embedding_size = [(c, min(50, (c+1)//2)) for c in unique_num]

        self._input_dim_t = self._data['input'][0].shape[-1]
        self._input_dim_s_val = self._data['static_val'].shape[1]
        self._input_dim_s_cat = self._data['static_cat'].shape[1]
        self._input_dim_s_cat_onehot = self._data['static_cat_onehot'].shape[1]
        self._folds = self._data['hy'].shape[0]

        # calculate the max timestep
        ret = np.asarray([np.sum(tt - tt[0] <= self._max_timestamp) for tt in self._data['timestamp']])
        self.max_step = max(ret)
  
    def _fillnan(self, x, mean):
        """
        Args:
            x: A np.array of static variables with shape (batch_size, d)
            mean: A np.array of mean value of each variables with shape (d,)
        
        Returns:
            same shape as x with filled with mean value of training set data 
        """

        for i in range(x.shape[1]):
            x[:, i][np.isnan(x[:, i])] = mean[i]

        return x

    def _rescale(self, x, mean, std):
        """
        Args:
            x: A np.array of several np.array with shape (t_i, d).
            mean: A np.array of shape (d,).
            std: A np.array of shape (d,).

        Returns:
            Same shape as x with rescaled values.
        """
        if x[0].ndim == 1:
            return np.asarray([(xx - mean) / std for xx in x])
        elif x[0].ndim == 2:
            return np.asarray([(xx - mean[np.newaxis, :]) / std[np.newaxis, :] for xx in x])

    def _filter(self, ts, max_timestamp=None, max_timestep=None):
        """
        Args:
            ts: A np.array of n np.array with shape (t_i, d).
            max_timestamp: an Integer > 0 or None.
            max_timesteps: an Integer > 0 or None.

        Returns:
            A np.array of n Integers. Its i-th element (x_i) indicates that
                we will take the first x_i numbers from i-th data sample. 
        """
        if max_timestamp is None:
            ret = np.asarray([len(tt) for tt in ts])
        else:
            ret = np.asarray([np.sum(tt - tt[0] <= max_timestamp) for tt in ts])
        if max_timestep is not None:
            ret = np.minimum(ret, max_timestep)
        return ret
    
    def _pad(self, x, lens):
        """
        Args:
            x: A np.array of n np.array with shape (t_i, d).
            lens: A np.array of n Integers > 0.

        Returns:
            A np.array of shape (n, t, d), where t = min(max_length, max(lens))
        """
        n = len(x) # number of sample
        t = self.max_step # the max timesteps
        d = 1 if x[0].ndim == 1 else x[0].shape[1] # 7
        ret = np.zeros([n, t, d], dtype=float)
        if x[0].ndim == 1:
            for i, xx in enumerate(x):
                ret[i, :lens[i]] = xx[:lens[i], np.newaxis]
        else:
            for i, xx in enumerate(x):
                # 取每一个样本的前lens(i)个时间点的所有列数据
                ret[i, :lens[i]] = xx[:lens[i]]
        return ret

    def _split_cat(self, x):

        inputs = []
        inputs += x[:-1]
        for i in range(self._input_dim_s_cat):
            x_cat = x[-1][..., [i]]
            inputs.append(x_cat)
        return inputs

    def _get_generator(self, i, i_fold, shuffle, return_targets):

        fold = np.copy(self._data['hy'][i_fold][i])
        mean_static = self._data['mean-static'][i_fold][0]
        std_static = self._data['std-static'][i_fold][0]
        folds = len(fold)  # The number of training sample

        if shuffle:
            np.random.shuffle(fold)  #shffule the data before split it into batch
        inputs = [self._data[s][fold] for s in ['static_val', 'static_cat_onehot']]
        inputs[0] = self._fillnan(inputs[0], mean_static)
        inputs[0] = self._rescale(inputs[0], mean_static, std_static)
        inputs = np.concatenate((inputs[0], inputs[1]), axis=1)
        # delete the time-series related data
        # inputs = np.concatenate([inputs[:, :5], inputs[:, 35:]], axis=1)

        shapes = inputs.shape
        targets = self._data['label'][fold].astype(int)
        nclasses = len(np.unique(targets))

        return (inputs, targets, nclasses, fold, shapes)

    def _get_generator_ee(self, i, i_fold, shuffle, return_targets):

        fold = np.copy(self._data['hy'][i_fold][i])
        mean_static = self._data['mean-static'][i_fold][0]
        std_static = self._data['std-static'][i_fold][0]

        if shuffle:
            np.random.shuffle(fold)  #shffule the data before split it into batch
        inputs = [self._data[s][fold] for s in ['static_val', 'static_cat']]
        inputs[0] = self._fillnan(inputs[0], mean_static)
        inputs[0] = self._rescale(inputs[0], mean_static, std_static)
        inputs = self._split_cat(inputs)

        shapes = (inputs[0].shape, (len(inputs[1]), len(inputs)-1))

        targets = self._data['label'][fold].astype(int)
        nclasses = len(np.unique(targets))

        return (inputs, targets, nclasses, fold, shapes)

    def _get_generator_ts(self, i, i_fold, shuffle, return_targets):

        fold = np.copy(self._data['hy'][i_fold][i])
        mean_temporal = self._data['mean-temporal'][i_fold][0]
        std_temporal = self._data['std-temporal'][i_fold][0]

        folds = len(fold)  # The number of training sample
        if shuffle:
            np.random.shuffle(fold) 

        inputs = [self._data[s][fold] for s in ['input', 'masking', 'timestamp']]
        inputs[0] = self._rescale(inputs[0], mean_temporal, std_temporal)
        # to set the original 0 after z-score to be 0
        # mask_bool = inputs[1].astype(np.bool)
        # inputs[0][~mask_bool] = 0
        lens = self._filter(inputs[2], self._max_timestamp, self._max_timestep)
        inputs[:3] = [self._pad(x, lens) for x in inputs[:3]]

        targets = self._data['label'][fold].astype(int)
        nclasses = len(np.unique(targets))

        shapes = (None, None)
        return (inputs, targets, nclasses, fold, shapes)

    def _get_generator_mm(self, i, i_fold, shuffle, return_targets):

        # The mean/std used in validation/test fold should also be from the training fold
        fold = np.copy(self._data['hy'][i_fold][i])
        mean_temporal = self._data['mean-temporal'][i_fold][0]
        std_temporal = self._data['std-temporal'][i_fold][0]
        mean_static = self._data['mean-static'][i_fold][0]
        std_static = self._data['std-static'][i_fold][0]

        folds = len(fold)  # The number of training sample
        if shuffle:
            np.random.shuffle(fold)  #shffule the data before split it into batch
        # extract the data by batch_size
        inputs = [self._data[s][fold] for s in ['input', 'masking', 'timestamp', 'static_val', 'static_cat_onehot']]

        inputs[0] = self._rescale(inputs[0], mean_temporal, std_temporal)

        inputs[3] = self._fillnan(inputs[3], mean_static)
        inputs[3] = self._rescale(inputs[3], mean_static, std_static)
        inputs[3] = np.concatenate((inputs[3], inputs[4]), axis=1)
        inputs = inputs[0:4]

        lens = self._filter(inputs[2], self._max_timestamp, self._max_timestep)
        inputs[:3] = [self._pad(x, lens) for x in inputs[:3]]

        targets = self._data['label'][fold].astype(int)
        nclasses = len(np.unique(targets))

        shapes = (None, None)

        return (inputs, targets, nclasses, fold, shapes)

    def training_generator(self, i_fold):
        if self.model_type == 'DNN' or self.model_type == 'SVM' or self.model_type == 'LR' or self.model_type == 'RF':
            return self._get_generator(i=0, i_fold=i_fold, shuffle=True, 
                                   return_targets=True)

        if self.model_type == 'DNN_EE':
            return self._get_generator_ee(i=0, i_fold=i_fold, shuffle=True, 
                                   return_targets=True)

        if self.model_type == 'GRUD':
            return self._get_generator_ts(i=0, i_fold=i_fold, shuffle=True, 
                                   return_targets=True)

        return self._get_generator_mm(i=0, i_fold=i_fold, shuffle=True, 
                                   return_targets=True)

    def validation_generator(self, i_fold):
        if self.model_type == 'DNN' or self.model_type == 'SVM' or self.model_type == 'LR' or self.model_type == 'RF':
            return self._get_generator(i=1, i_fold=i_fold, shuffle=True, 
                                   return_targets=True)

        if self.model_type == 'DNN_EE':
            return self._get_generator_ee(i=1, i_fold=i_fold, shuffle=True, 
                                   return_targets=True)
        
        if self.model_type == 'GRUD':
            return self._get_generator_ts(i=1, i_fold=i_fold, shuffle=True, 
                                   return_targets=True)

        return self._get_generator_mm(i=1, i_fold=i_fold, shuffle=True, 
                                   return_targets=True)
    
    def test_generator(self, i_fold):

        if self.model_type == 'DNN' or self.model_type == 'SVM' or self.model_type == 'LR' or self.model_type == 'RF':
            return self._get_generator(i=2, i_fold=i_fold, shuffle=True, 
                                   return_targets=True)

        if self.model_type == 'DNN_EE':
            return self._get_generator_ee(i=2, i_fold=i_fold, shuffle=True, 
                                   return_targets=True)
        
        if self.model_type == 'GRUD':
            return self._get_generator_ts(i=2, i_fold=i_fold, shuffle=True, 
                                   return_targets=True)

        return self._get_generator_mm(i=2, i_fold=i_fold, shuffle=True, 
                                   return_targets=True)

    @property
    def folds(self):
        return self._folds

    @property
    def embedding_size(self):
        return self._embedding_size
    
    @property
    def input_dim_t(self):
        return self._input_dim_t

    @property
    def input_dim_s_val(self):
        return self._input_dim_s_val
    
    @property
    def max_timestep(self):
        return self.max_step

class LoadDataImputed():
    """
    Date: 2022-03-20
    The class for load data for imputed time-series data
    Load 'data.npz' and 'fold.npz' for multi-type model training and testing.
    In 'data.npz':
        Required: 'input', 'masking','static_val', 'static_cat', 'label'
        Shape: (n_samples,)
    In 'fold.npz':
        Required: 'mean_hy$[0-4]$_temporal' and 'std_hy$[0-4]$_temporal', 'mean_hy$[0-4]$_static', 'std_hy$[0-4]$_static'
        Shape: (n_splits, 3)
    """

    def __init__(self, data_path, model_type, label_name, max_timestep, max_timestamp):
        
        self._input_dim_t = None
        self._input_dim_s_val = None
        self._input_dim_s_cat = None
        self._folds = None
        self._embedding_size = None
        self.model_type = model_type
        self._max_timestep = max_timestep
        self._max_timestamp = max_timestamp

        self._data_file = os.path.join(data_path, 'data.npz')
        self._fold_file = os.path.join(data_path, 'fold.npz')
        self._load_data(label_name)

    def _load_data(self, label_name):
        """
        Label_name: 
            the index of different label ranging from [0-4]
        """
        if not os.path.exists(self._data_file):
            raise ValueError('Data file does not exist...')
        if not os.path.exists(self._fold_file):
            raise ValueError('Fold file does not exist...')

        data = np.load(self._data_file, allow_pickle=True)
        fold = np.load(self._fold_file, allow_pickle=True)

        self._data = {}
        for i in ['input', 'static_val', 'static_cat', 'static_cat_onehot']:
            self._data[i] = data[i]

        self._data['label'] = data['label'][:,label_name + 1]
        
        if label_name == -1:
            for i in ['hy', 'mean-temporal', 'std-temporal', 'mean-static', 'std-static']:
                self._data[i] = fold[i + '-' + str(label_name+1)]
        else:
            for i in ['hy', 'mean-temporal', 'std-temporal', 'mean-static', 'std-static']:
                self._data[i] = fold[i + '-' + str(label_name)]

        # cal the embedding size for categorical variables
        unique_num = [len(np.unique(self._data['static_cat'][:, i])) for i in range(self._data['static_cat'].shape[1])]
        self._embedding_size = [(c, min(50, (c+1)//2)) for c in unique_num]

        self._input_dim_t = self._data['input'][0].shape[-1]
        self._input_dim_s_val = self._data['static_val'].shape[1]
        self._input_dim_s_cat = self._data['static_cat'].shape[1]
        self._input_dim_s_cat_onehot = self._data['static_cat_onehot'].shape[1]
        self._folds = self._data['hy'].shape[0]
    
    def _fillnan(self, x, mean):
        """
        Args:
            x: A np.array of static variables with shape (batch_size, d)
            mean: A np.array of mean value of each variables with shape (d,)
        
        Returns:
            same shape as x with filled with mean value of training set data 
        """

        for i in range(x.shape[1]):
            x[:, i][np.isnan(x[:, i])] = mean[i]

        return x

    def _rescale(self, x, mean, std):
        """
        Args:
            x: A np.array of several np.array with shape (t_i, d).
            mean: A np.array of shape (d,).
            std: A np.array of shape (d,).

        Returns:
            Same shape as x with rescaled values.
        """
        if x[0].ndim == 1:
            return np.asarray([(xx - mean) / std for xx in x])
        elif x[0].ndim == 2:
            return np.asarray([(xx - mean[np.newaxis, :]) / std[np.newaxis, :] for xx in x])

    def _filter(self, ts, max_timestamp=None, max_timesteps=None):
        """
        Args:
            ts: A np.array of n np.array with shape (t_i, d).
            max_timestamp: an Integer > 0 or None.
            max_timesteps: an Integer > 0 or None.

        Returns:
            A np.array of n Integers. Its i-th element (x_i) indicates that
                we will take the first x_i numbers from i-th data sample. 
        """
        if max_timestamp is None:
            ret = np.asarray([len(tt) for tt in ts])
        else:
            ret = np.asarray([np.sum(tt - tt[0] <= max_timestamp) for tt in ts])
        if max_timesteps is not None:
            ret = np.minimum(ret, max_timesteps)
        return ret
    
    def _pad(self, x, lens):
        """
        Args:
            x: A np.array of n np.array with shape (t_i, d).
            lens: A np.array of n Integers > 0.

        Returns:
            A np.array of shape (n, t, d), where t = min(max_length, max(lens))
        """
        n = len(x) #32个sample
        # t = max(lens) # 列表内的最大值也就是最大的时间点个数，即48小时内最长的时间跨度
        t = 38
        d = 1 if x[0].ndim == 1 else x[0].shape[1] # 136
        ret = np.zeros([n, t, d], dtype=float) # [32,t,136]
        if x[0].ndim == 1:
            for i, xx in enumerate(x):
                ret[i, :lens[i]] = xx[:lens[i], np.newaxis]
        else:
            for i, xx in enumerate(x):
                # 取每一个样本的前lens(i)个时间点的所有列数据
                ret[i, :lens[i]] = xx[:lens[i]]
        return ret

    def _split_cat(self, x):

        inputs = []
        inputs += x[:-1]
        for i in range(self._input_dim_s_cat):
            x_cat = x[-1][..., [i]]
            inputs.append(x_cat)
        return inputs

    def _get_generator(self, i, i_fold, shuffle, return_targets):

        fold = np.copy(self._data['hy'][i_fold][i])
        mean_static = self._data['mean-static'][i_fold][0]
        std_static = self._data['std-static'][i_fold][0]
        folds = len(fold)  # The number of training sample

        if shuffle:
            np.random.shuffle(fold)  #shffule the data before split it into batch
        inputs = [self._data[s][fold] for s in ['static_val', 'static_cat_onehot']]
        inputs[0] = self._fillnan(inputs[0], mean_static)
        inputs[0] = self._rescale(inputs[0], mean_static, std_static)
        inputs = np.concatenate((inputs[0], inputs[1]), axis=1)
        shapes = inputs.shape
        targets = self._data['label'][fold].astype(int)
        nclasses = len(np.unique(targets))

        return (inputs, targets, nclasses, fold, shapes)

    def _get_generator_ee(self, i, i_fold, shuffle, return_targets):

        fold = np.copy(self._data['hy'][i_fold][i])
        mean_static = self._data['mean-static'][i_fold][0]
        std_static = self._data['std-static'][i_fold][0]

        if shuffle:
            np.random.shuffle(fold)  #shffule the data before split it into batch
        inputs = [self._data[s][fold] for s in ['static_val', 'static_cat']]
        inputs[0] = self._fillnan(inputs[0], mean_static)
        inputs[0] = self._rescale(inputs[0], mean_static, std_static)
        inputs = self._split_cat(inputs)

        shapes = (inputs[0].shape, (len(inputs[1]), len(inputs)-1))

        targets = self._data['label'][fold].astype(int)
        nclasses = len(np.unique(targets))

        return (inputs, targets, nclasses, fold, shapes)

    def _get_generator_ts(self, i, i_fold, shuffle, return_targets):

        fold = np.copy(self._data['hy'][i_fold][i])
        
        mean_temporal = self._data['mean-temporal'][i_fold][0]
        std_temporal = self._data['std-temporal'][i_fold][0]

        folds = len(fold)  # The number of training sample
        if shuffle:
            np.random.shuffle(fold) 

        inputs = self._data['input'][fold]
        inputs = self._rescale(inputs, mean_temporal, std_temporal)
        targets = self._data['label'][fold].astype(int)
        nclasses = len(np.unique(targets))

        shapes = inputs[0].shape
        # deal with np.nan in inputs
        inputs_nonan = np.where(np.isnan(inputs), 0, inputs)
        return (inputs_nonan, targets, nclasses, fold, shapes)

    def _get_generator_mm(self, i, i_fold, shuffle, return_targets):

        # The mean/std used in validation/test fold should also be from the training fold
        fold = np.copy(self._data['hy'][i_fold][i])
        mean_temporal = self._data['mean-temporal'][i_fold][0]
        std_temporal = self._data['std-temporal'][i_fold][0]
        mean_static = self._data['mean-static'][i_fold][0]
        std_static = self._data['std-static'][i_fold][0]

        folds = len(fold)  # The number of training sample
        if shuffle:
            np.random.shuffle(fold)  #shffule the data before split it into batch
        # extract the data by batch_size
        inputs = [self._data[s][fold] for s in ['input', 'static_val', 'static_cat']]

        inputs[0] = self._rescale(inputs[0], mean_temporal, std_temporal)

        inputs[1] = self._fillnan(inputs[1], mean_static)
        inputs[1] = self._rescale(inputs[1], mean_static, std_static)

        inputs = self._split_cat(inputs)

        targets = self._data['label'][fold].astype(int)
        nclasses = len(np.unique(targets))

        shapes = (None, None)

        return (inputs, targets, nclasses, fold, shapes)

    def training_generator(self, i_fold):
        if self.model_type == 'DNN' or self.model_type == 'SVM' or self.model_type == 'LR' or self.model_type == 'RF':
            return self._get_generator(i=0, i_fold=i_fold, shuffle=True, 
                                   return_targets=True)

        if self.model_type == 'DNN_EE':
            return self._get_generator_ee(i=0, i_fold=i_fold, shuffle=True, 
                                   return_targets=True)

        if self.model_type == 'LSTM':
            return self._get_generator_ts(i=0, i_fold=i_fold, shuffle=True, 
                                   return_targets=True)

        return self._get_generator_mm(i=0, i_fold=i_fold, shuffle=True, 
                                   return_targets=True)

    def validation_generator(self, i_fold):
        if self.model_type == 'DNN' or self.model_type == 'SVM' or self.model_type == 'LR' or self.model_type == 'RF':
            return self._get_generator(i=1, i_fold=i_fold, shuffle=True, 
                                   return_targets=True)

        if self.model_type == 'DNN_EE':
            return self._get_generator_ee(i=1, i_fold=i_fold, shuffle=True, 
                                   return_targets=True)
        
        if self.model_type == 'LSTM':
            return self._get_generator_ts(i=1, i_fold=i_fold, shuffle=True, 
                                   return_targets=True)

        return self._get_generator_mm(i=1, i_fold=i_fold, shuffle=True, 
                                   return_targets=True)
    
    def test_generator(self, i_fold):

        if self.model_type == 'DNN' or self.model_type == 'SVM' or self.model_type == 'LR' or self.model_type == 'RF':
            return self._get_generator(i=2, i_fold=i_fold, shuffle=True, 
                                   return_targets=True)

        if self.model_type == 'DNN_EE':
            return self._get_generator_ee(i=2, i_fold=i_fold, shuffle=True, 
                                   return_targets=True)
        
        if self.model_type == 'LSTM':
            return self._get_generator_ts(i=2, i_fold=i_fold, shuffle=True, 
                                   return_targets=True)

        return self._get_generator_mm(i=2, i_fold=i_fold, shuffle=True, 
                                   return_targets=True)

    @property
    def folds(self):
        return self._folds

    @property
    def embedding_size(self):
        return self._embedding_size
    
    @property
    def input_dim_t(self):
        return self._input_dim_t

    @property
    def input_dim_s_val(self):
        return self._input_dim_s_val
