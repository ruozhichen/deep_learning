# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import joblib

import config
"""
gender
train: 0.7140780438621694
test: 0.71382332350933

"""
def get_train_test(label):
    df = pd.read_csv("../data/train/data_user.csv")
    x = df[config.FEATURE_COLS]
    y = df[label]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    print("train: %d" % len(x_train))
    print("test: %d" % len(x_test))
    return x_train, y_train, x_test, y_test

def get_test_data():
    df = pd.read_csv("../data/test/data_user.csv")
    x = df[config.FEATURE_COLS]
    print("Get test data done.")
    return x

def run_gbdt(x_train, y_train, x_test, y_test, label):
    gbr = GradientBoostingClassifier(n_estimators=1000, max_depth=3, min_samples_split=500,
                                     min_samples_leaf=40, learning_rate=0.1)
    print("begin to fit.")
    gbr.fit(x_train, y_train)
    joblib.dump(gbr, '../data/model_gbdt/train_model_%s.pkl' % label)  # 保存模型
    # y_gbr = gbr.predict(x_train)
    # y_gbr1 = gbr.predict(x_test)
    acc_train = gbr.score(x_train, y_train)
    acc_test = gbr.score(x_test, y_test)
    print(acc_train)
    print(acc_test)


def predict(label):
    x_test = get_test_data()
    gpdt = joblib.load('../data/model_gbdt/train_model_%s.pkl' % label)
    print("Begin to predict x_test.")
    y = gpdt.predict(x_test)
    with open("../data/res_%s/%s_res_gbdt.txt" % (label, label), 'w') as fout:
        for value in y:
            fout.write("%s\n" % value)


if __name__ == '__main__':
    # x_train, y_train, x_test, y_test = get_train_test('age')
    # run_gbdt(x_train, y_train, x_test, y_test, 'age')
    predict('gender')