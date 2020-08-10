# -*- coding: utf-8 -*-
from enum import Enum
import tensorflow as tf

class ColumnType(Enum):
    CATEGORY = 1
    CONTINUOUS = 2

class ColumnTransform(Enum):
    HASH_BUCKET = 1
    VOCAB = 2
    LOG = 3

USER_FEATURE_COLS = ['product_id', 'product_category', 'industry', 'weekday']
CLICK_COLS = ["time", "user_id", "creative_id", "click_times"]
AD_COLS = ["creative_id", "ad_id", "product_id", "product_category", "advertiser_id", "industry"]
USER_COLS = ["user_id", "age", "gender"]

feature_dict = {
    'creative_id': {
        'type': ColumnType.CATEGORY,
        'transform': ColumnTransform.HASH_BUCKET,
        'parameter': 100000,
        'dtype': tf.int64,
        'is_deep': True,
        'is_wide': True,
        'hashlength': 100000
    },
    'click_times': {
        'type': ColumnType.CONTINUOUS,
        'transform': None,
        'parameter': None,
        'dtype': tf.int16,
        'is_deep': True,
        'is_wide': True,
        'hashlength': 1
    },
    'ad_id': {
        'type': ColumnType.CATEGORY,
        'transform': ColumnTransform.HASH_BUCKET,
        'parameter': 100000,
        'dtype': tf.int64,
        'is_deep': True,
        'is_wide': True,
        'hashlength': 100000
    },
    'product_id': {
        'type': ColumnType.CATEGORY,
        'transform': ColumnTransform.HASH_BUCKET,
        'parameter': 1000,
        'dtype': tf.int32,
        'is_deep': True,
        'is_wide': True,
        'hashlength': 1000
    },
    'product_category': {
        'type': ColumnType.CATEGORY,
        'transform': ColumnTransform.VOCAB,
        'parameter': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],
        'dtype': tf.int16,
        'is_deep': True,
        'is_wide': True,
        'hashlength': 18
    },
    'advertiser_id': {
        'type': ColumnType.CATEGORY,
        'transform': ColumnTransform.HASH_BUCKET,
        'parameter': 1000,
        'dtype': tf.int32,
        'is_deep': True,
        'is_wide': True,
        'hashlength': 1000
    },
    'industry': {
        'type': ColumnType.CATEGORY,
        'transform': ColumnTransform.HASH_BUCKET,
        'parameter': 300,
        'dtype': tf.int32,
        'is_deep': True,
        'is_wide': True,
        'hashlength': 300
    },
    'weekday': {
        'type': ColumnType.CATEGORY,
        'transform': ColumnTransform.VOCAB,
        'parameter': [0, 1, 2, 3, 4, 5, 6],
        'dtype': tf.int16,
        'is_deep': True,
        'is_wide': True,
        'hashlength': 7
    },
    'user_product_id_click': {
        'type': ColumnType.CONTINUOUS,
        'transform': ColumnTransform.LOG,
        'parameter': None,
        'dtype': tf.int16,
        'is_deep': True,
        'is_wide': True,
        'hashlength': 1
    },
    'user_product_category_click': {
        'type': ColumnType.CONTINUOUS,
        'transform': ColumnTransform.LOG,
        'parameter': None,
        'dtype': tf.int16,
        'is_deep': True,
        'is_wide': True,
        'hashlength': 1
    },
    'user_industry_click': {
        'type': ColumnType.CONTINUOUS,
        'transform': ColumnTransform.LOG,
        'parameter': None,
        'dtype': tf.int16,
        'is_deep': True,
        'is_wide': True,
        'hashlength': 1
    },
    'user_weekday_click': {
        'type': ColumnType.CONTINUOUS,
        'transform': ColumnTransform.LOG,
        'parameter': None,
        'dtype': tf.int16,
        'is_deep': True,
        'is_wide': True,
        'hashlength': 1
    }
}
FEATURE_COLS = feature_dict.keys()