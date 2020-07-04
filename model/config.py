# -*- coding: utf-8 -*-
from enum import Enum
import tensorflow as tf

class ColumnType(Enum):
    CATEGORY = 1
    CONTINUOUS = 2

class ColumnTransform(Enum):
    HASH_BUCKET = 1
    VOCAB = 2
    IDENTITY = 3
    MIN_MAX = 4
    STANDARD = 5
    LOG = 6

USER_FEATURE_COLS = ['product_id', 'product_category', 'industry', 'weekday']
CLICK_COLS = ["time", "user_id", "creative_id", "click_times"]
AD_COLS = ["creative_id", "ad_id", "product_id", "product_category", "advertiser_id", "industry"]
USER_COLS = ["user_id", "age", "gender"]

# FEATURE_COLS = ["creative_id", "click_times", "ad_id", "product_id", "product_category", "advertiser_id", "industry"]

feature_dict = {
    # ad feature
    'creative_id': {
        'type': ColumnType.CATEGORY,
        'transform': ColumnTransform.HASH_BUCKET,
        'parameter': 1000000,
        'dtype': tf.int64,
        'is_deep': True,
        'is_wide': True
    },
    'click_times': {
        'type': ColumnType.CONTINUOUS,
        'transform': None,
        'parameter': None,
        'dtype': tf.int16,
        'is_deep': True,
        'is_wide': True
    },
    'ad_id': {
        'type': ColumnType.CATEGORY,
        'transform': ColumnTransform.HASH_BUCKET,
        'parameter': 1000000,
        'dtype': tf.int64,
        'is_deep': True,
        'is_wide': True
    },
    'product_id': {
        'type': ColumnType.CATEGORY,
        'transform': ColumnTransform.HASH_BUCKET,
        'parameter': 10000,
        'dtype': tf.int32,
        'is_deep': True,
        'is_wide': True
    },
    'product_category': {
        'type': ColumnType.CATEGORY,
        'transform': ColumnTransform.VOCAB,
        'parameter': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
        'dtype': tf.int16,
        'is_deep': True,
        'is_wide': True
    },
    'advertiser_id': {
        'type': ColumnType.CATEGORY,
        'transform': ColumnTransform.HASH_BUCKET,
        'parameter': 10000,
        'dtype': tf.int32,
        'is_deep': True,
        'is_wide': True
    },
    'industry': {
        'type': ColumnType.CATEGORY,
        'transform': ColumnTransform.HASH_BUCKET,
        'parameter': 300,
        'dtype': tf.int32,
        'is_deep': True,
        'is_wide': True
    },
    'weekday': {
        'type': ColumnType.CATEGORY,
        'transform': ColumnTransform.VOCAB,
        'parameter': [0, 1, 2, 3, 4, 5, 6, 7],
        'dtype': tf.int16,
        'is_deep': True,
        'is_wide': True
    },
    'user_product_id_click': {
        'type': ColumnType.CONTINUOUS,
        'transform': ColumnTransform.LOG,
        'parameter': None,
        'dtype': tf.int16,
        'is_deep': True,
        'is_wide': True
    },
    'user_product_category_click': {
        'type': ColumnType.CONTINUOUS,
        'transform': ColumnTransform.LOG,
        'parameter': None,
        'dtype': tf.int16,
        'is_deep': True,
        'is_wide': True
    },
    'user_industry_click': {
        'type': ColumnType.CONTINUOUS,
        'transform': ColumnTransform.LOG,
        'parameter': None,
        'dtype': tf.int16,
        'is_deep': True,
        'is_wide': True
    },
    'user_weekday_click': {
        'type': ColumnType.CONTINUOUS,
        'transform': ColumnTransform.LOG,
        'parameter': None,
        'dtype': tf.int16,
        'is_deep': True,
        'is_wide': True
    }
}
# time,user_id,creative_id,click_times,ad_id,product_id,product_category,advertiser_id,industry,age,gender,
# weekday,user_product_id_click,user_product_category_click,user_industry_click,user_weekday_click
FEATURE_COLS = feature_dict.keys()