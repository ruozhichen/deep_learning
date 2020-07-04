# -*- coding: utf-8 -*-
"""
A data parser for Porto Seguro's Safe Driver Prediction competition's dataset.
URL: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction
"""
import pandas as pd
import sys
sys.path.append("../model")
import config

class FeatureDictionary(object):
    def __init__(self, train_file=None, test_file=None,
                 df_train=None, df_test=None):
        self.train_file = train_file
        self.test_file = test_file
        self.df_train = df_train
        self.df_test = df_test
        self.gen_feat_dict()

    def gen_feat_dict(self):
        # if self.df_train is None:
        #     df_train = pd.read_csv(self.train_file)
        # else:
        #     df_train = self.df_train
        # if self.df_test is None:
        #     df_test = pd.read_csv(self.test_file)
        # else:
        #     df_test = self.df_test
        # df = pd.concat([df_train, df_test])
        self.col2feat_id = {}
        total_dim = 0
        for col, feature in config.feature_dict.items():
            if feature['hashlength'] == 1:
                # map to a single index
                self.col2feat_id[col] = total_dim
                total_dim += 1
            else:
                hashlength = feature['hashlength']
                self.col2feat_id[col] = dict(zip(range(hashlength), range(total_dim, hashlength + total_dim)))
                total_dim += hashlength
            print(col, total_dim)
        self.feat_dim = total_dim


class DataParser(object):
    def __init__(self, feat_dict, label_col):
        self.feat_dict = feat_dict
        self.label_col = label_col

    def parse(self, infile=None, df=None, has_label=True):
        if infile is None:
            df_ids = df.copy()
        else:
            df_ids = pd.read_csv(infile)
        label = None
        if has_label:
            label = df_ids[self.label_col].values.tolist()
        df_ids = df_ids[config.feature_dict.keys()]
        # dfi for feature index
        # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)
        df_vals = df_ids.copy()
        for col, feature in config.feature_dict.items():
            if feature['hashlength'] == 1:
                df_ids[col] = self.feat_dict.col2feat_id[col]
            else:
                #取hash后，再转换成对应的feature_id
                df_ids[col] = df_ids[col] % feature['hashlength']
                # 因为是离散的，需要根据其值来获取对应的index
                df_ids[col] = df_ids[col].map(self.feat_dict.col2feat_id[col])
                df_vals[col] = 1

        # list of list of feature indices of each sample in the dataset
        ids = df_ids.values.tolist() # 相同维度的数组
        # list of list of feature values of each sample in the dataset
        vals = df_vals.values.tolist()
        return ids, vals, label

if __name__ == '__main__':
    feat_dict = FeatureDictionary()
    print(feat_dict.col2feat_id.keys())
    dataparser = DataParser(feat_dict, "gender")
    ids, vals, label = dataparser.parse(infile="F:\\deep\deep_learning\\data\\train\\test_sample.csv")
    print(ids[0])
    print(vals[0])
    print(label[0])
