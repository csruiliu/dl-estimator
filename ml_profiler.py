import tensorflow as tf
from keras.datasets import cifar10
from keras.utils import to_categorical

from tensorflow_cifar.models.alexnet import AlexNet
from tensorflow_cifar.tools.model_tools import train_model
from tensorflow_cifar.tools.model_tools import evaluate_model

if __name__ == "__main__":

    model = AlexNet(num_classes=10)
    feature_ph = tf.placeholder(tf.float32, [None, 32, 32, 3])
    label_ph = tf.placeholder(tf.int32, [None, 10])
    logit = model.build(feature_ph)

    # count overall trainable parameters
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters

    '''
    opt = 'SGD'
    lr = 0.01

    train_op = train_model(logit, label_ph, opt, lr)
    eval_op = evaluate_model(logit, label_ph)


    (train_data, train_labels), (test_data, test_labels) = cifar10.load_data()
    '''