import pandas as pd
import tensorflow as tf

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']


def maybe_download():
    train_path = tf.keras.utils.get_file(fname=TRAIN_URL.split('/')[-1],
                                         origin=TRAIN_URL, cache_dir='.')
    test_path = tf.keras.utils.get_file(fname=TEST_URL.split('/')[-1],
                                        origin=TEST_URL, cache_dir='.')
    return train_path, test_path


def load_data(y_name='Species'):
    train_path, test_path = maybe_download()
    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)
    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size=256, num_epochs=1):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # shuffle(): 将数据打乱，数值越大，混乱程度越大
    # repeat(): 数据集重复了指定次数
    dataset = dataset.shuffle(1000).repeat(num_epochs).batch(batch_size)
    # return dataset
    iterator = dataset.make_one_shot_iterator()  # 数据取一次后就丢弃了
    feature, label = iterator.get_next()
    return feature, label


def eval_input_fn(features, labels, batch_size=256):
    features = dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    dataset = dataset.batch(batch_size)
    # return dataset
    iterator = dataset.make_one_shot_iterator()  # 数据取一次后就丢弃了
    feature, label = iterator.get_next()
    return feature, label


CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0.0]]


def _parse_line(line):
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)
    features = dict(zip(CSV_COLUMN_NAMES, fields))
    label = features.pop('Species')
    return features, label


def csv_input_fn(csv_path, batch_size):
    dataset = tf.data.TextLineDataset(csv_path).skip(1)
    dataset = dataset.map(_parse_line)
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset

import argparse
import tensorflow as tf

parse = argparse.ArgumentParser()
parse.add_argument('--batch_size', default=100, type=int,
                   help='batch size')
parse.add_argument('--train_steps', default=1000, type=int,
                   help='training steps')


def main(argv):
    args = parse.parse_args(argv[1:])
    (train_x, train_y), (test_x, test_y) = load_data()
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[10, 10],
        n_classes=3,
        optimizer=tf.train.AdamOptimizer(
            learning_rate=0.01
        )
    )
    classifier.train(
        input_fn=lambda: train_input_fn(train_x, train_y,
                                                  args.batch_size),
        steps=args.train_steps
    )
    eval_result = classifier.evaluate(
        input_fn=lambda: eval_input_fn(test_x, test_y,
                                                 args.batch_size)
    )
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }
    predictions = classifier.predict(
        input_fn=lambda: eval_input_fn(predict_x,
                                                 labels=None,
                                                 batch_size=args.batch_size)
    )
    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
    for pre_dict, expec in zip(predictions, expected):
        class_id = pre_dict['class_ids'][0]
        probability = pre_dict['probabilities'][class_id]
        print(template.format(read_data.SPECIES[class_id],
                              100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
