# -*- coding: utf-8 -*-
"""
author: Sunny.Xia
date: 2020-08-09
"""
import sys
import pandas as pd
sys.path.append("../model")
import config
from collections import Counter,defaultdict
from optparse import OptionParser

class Preprocessing(object):
    def __init__(self, home, type):
        self._home = home
        self._path = home + '\\%s' % type
        # self._path = home + '\\tmp'
        self._out_file = home + '\\%s\\data.csv' % type
        # self._out_file = home + '\\tmp\\data.csv'
        self._type = type

    def _set_zero(self, value):
        if value == '\\N':
            return 0
        return int(value)

    def _load_click(self, path):
        # time: 1-91 creative_id: 2481135
        log_path = path +'\\click_log.csv'
        logDF = pd.read_csv(log_path)
        print(logDF.dtypes)
        for col in config.CLICK_COLS:
            print("col: %s %d\n" % (col, len(logDF[col].unique())))
        return logDF

    def _load_ad(self, path):
        # train + test 各列的unique个数
        # creative_id: 3412772
        # ad_id: 3027360
        # product_id: 39057
        # product_category: 18
        # advertiser_id: 57870
        # industry: 332
        ad_path = path + '\\ad.csv'
        adDF = pd.read_csv(ad_path)
        adDF.loc[adDF['product_id'] == '\\N', 'product_id'] = 0
        adDF.loc[adDF['industry'] == '\\N', 'industry'] = 0
        # adDF['product_id'] = adDF['product_id'].map(self._set_zero)
        # adDF['industry'] = adDF['industry'].map(self._set_zero)
        print(adDF.dtypes)
        for col in config.AD_COLS:
            print("col: %s %d\n" % (col, len(adDF[col].unique())))
        return adDF

    def _load_user(self, path):
        # user: 900000
        # age: 1-10
        # gender: 1-2
        user_path = path + '\\user.csv'
        userDF = pd.read_csv(user_path)
        print(userDF.dtypes)
        for col in config.USER_COLS:
            print("col: %s %d\n" % (col, len(userDF[col].unique())))
        return userDF

    def _load_data(self):
        clickDF = self._load_click(self._path)
        adDF = self._load_ad(self._path)
        df = pd.merge(clickDF, adDF, on=['creative_id'], how='left')
        if self._type == 'train':
            userDF = self._load_user(self._path)
            df = pd.merge(df, userDF, on=['user_id'], how='left')
            df = df.sample(frac=1).reset_index(drop=True)  # train数据打乱顺序
        else:
            df['age'] = [0] * len(df)
            df['gender'] = [0] * len(df)
        df['age'] = df['age'] -1 # label调整为从0开始
        df['gender'] = df['gender'] -1
        # df.sort_values(['time'], ascending=True, inplace=True)
        df.to_csv(self._out_file, index=False)
        return df

    def propress_user_feature(self):
        data_path = self._path + "\\data.csv"
        df = pd.read_csv(data_path)
        #df.groupby(['dt', 'order_plan_id', 'ad_zone_id', 'order_type']).sum().reset_index()
        df['weekday'] = df['time'] % 7
        df_user_feature = df[['user_id', 'click_times'] + config.USER_FEATURE_COLS]
        for col in config.USER_FEATURE_COLS:
            print("Begin to propress user_%s_click." % col)
            dim_key = ['user_id', col]
            select_key = ['user_id', col, 'click_times']
            res_df = df_user_feature.groupby(dim_key)['click_times'].sum().reset_index(name='user_%s_click' % col)
            res_df.sort_values(dim_key, ascending=True, inplace=True)
            res_df.to_csv(self._path + "\\user_%s_click.csv" % col, index=False)

    def join_user_feature(self):
        data_path = self._path + "\\data.csv"
        df = pd.read_csv(data_path)
        df['weekday'] = df['time'] % 7
        for col in config.USER_FEATURE_COLS:
            print("Begin to join user_%s_click." % col)
            user_feature_df = pd.read_csv(self._path + "\\user_%s_click.csv" % col)
            df = pd.merge(df, user_feature_df, on=['user_id', col], how='left')
        df.to_csv(self._path + "\\data_user.csv", index=False)

    def sample_train_data(self, frac=0.1):
        data_path = self._path + "\\data_user.csv"
        df = pd.read_csv(data_path)
        df = df.sample(frac=frac).reset_index(drop=True)  # train数据打乱顺序
        num = int(len(df) * 0.9)
        train_df = df[:num]
        test_df = df[num:]
        df.to_csv(self._path + "\\data_sample.csv", index=False)
        train_df.to_csv(self._path + "\\train_sample.csv", index=False)
        test_df.to_csv(self._path + "\\test_sample.csv", index=False)

    def run(self):
        self._load_data()
        self.propress_user_feature()
        self.join_user_feature()
        self.sample_train_data()

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-p", "--path", dest="data_path",
                      default="D:\deep_learning\deep_learning\data",
                      help="read data from path")
    parser.add_option("-t", "--type", dest="type",
                      default="train",
                      help="train | test")
    options, args = parser.parse_args()
    print(options)
    print(args)
    preprocessing = Preprocessing(args.data_path, args.type)
    preprocessing.run()

