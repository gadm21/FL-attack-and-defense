

import numpy as np
import random
from datetime import datetime
import os 
from os.path import join
import argparse 
# from keras import backend as K
import json

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras import layers

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
# from mia.estimators import ShadowModelBundle, AttackModelBundle, prepare_attack_data
from sklearn.metrics import classification_report

from sklearn.ensemble import AdaBoostClassifier

from DP import DPAccountant
# from tensorflow_privacy.privacy.optimizers import dp_optimizer
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer

script_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(script_path)
RESULTS_PATH = os.path.join(parent_directory, 'results/')
DATA_PATH = os.path.join(parent_directory, 'data/')


HEIGHT = 32
WIDTH = 32
CHANNELS = 3
NUM_CLASSES = 10

if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)


def get_data(dataset) : 
    if dataset == 'cifar10' : 
        return cifar10_data()
    elif dataset == 'cifar100' :
        return cifar100_data()
    elif dataset == 'mnist' :
        return mnist_data()
    elif dataset == 'imdb' :
        return imdb_data()
    elif dataset == 'boston' :
        return boston_data()
    
    else :
        raise ValueError("Dataset not supported")


def update_args_with_dict(args, dict) :
    args_dict = vars(args)
    args_dict.update(dict)
    args = argparse.Namespace(**args_dict)
    return args

def load_json_field(file_path, field_name = None):
    with open(file_path, 'r') as file:
        data = json.load(file)

    if field_name is None:
        return data
    
    if field_name in data:
        return data[field_name]
    else:
        raise KeyError("Field '{}' not found in the JSON file.".format(field_name))

def cifar10_data() : 
    cifar_train, cifar_test = tf.keras.datasets.cifar10.load_data()
    cifar_class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    input_shape = cifar_train[0].shape[1:]
    num_classes = len(cifar_class_labels)
    dataset_metadata = {'input_shape': input_shape, 'num_classes': num_classes, 'class_labels': cifar_class_labels}
    return cifar_train, cifar_test, dataset_metadata

def cifar100_data() :
    cifar_train, cifar_test = tf.keras.datasets.cifar100.load_data()
    input_shape = cifar_train[0].shape[1:]
    num_classes = 100
    dataset_metadata = {'input_shape': input_shape, 'num_classes': num_classes}
    return cifar_train, cifar_test, dataset_metadata

def mnist_data() :
    mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()
    # add a channel dimension to data 
    mnist_train = (np.expand_dims(mnist_train[0], axis=-1), mnist_train[1])
    mnist_test = (np.expand_dims(mnist_test[0], axis=-1), mnist_test[1])

    mnist_class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    input_shape = mnist_train[0].shape[1:]
    num_classes = len(mnist_class_labels)
    dataset_metadata = {'input_shape': input_shape, 'num_classes': num_classes, 'class_labels': mnist_class_labels}
    return mnist_train, mnist_test, dataset_metadata

# nlp data
def imdb_data() :
    imdb_train, imdb_test = tf.keras.datasets.imdb.load_data()
    imdb_class_labels = ['0', '1']
    input_shape = imdb_train[0].shape[1:]
    num_classes = len(imdb_class_labels)
    dataset_metadata = {'input_shape': input_shape, 'num_classes': num_classes, 'class_labels': imdb_class_labels}
    return imdb_train, imdb_test, dataset_metadata

# tabular data from sklearn
def boston_data() :
    boston_train, boston_test = tf.keras.datasets.boston_housing.load_data()
    boston_class_labels = ['0', '1']
    input_shape = boston_train[0].shape[1:]
    num_classes = len(boston_class_labels)
    dataset_metadata = {'input_shape': input_shape, 'num_classes': num_classes, 'class_labels': boston_class_labels}
    return boston_train, boston_test, dataset_metadata


def split_data(source, num_clients, local_size, shuffle = True):   
    
    if shuffle:
        p = np.random.permutation(len(source[0]))
        source = (source[0][p], source[1][p])
    
    client_data = []
    for s in range(num_clients):
        start = s*local_size
        end = s*local_size + local_size
        client_data.append((source[0][start:end], source[1][start:end]))
    
    central_data = (np.array(source[0][:local_size*num_clients] , dtype=np.float32), source[1][:local_size*num_clients])
    external_data = (np.array(source[0][local_size*num_clients:] , dtype=np.float32), source[1][local_size*num_clients:]) 
    
    return central_data, client_data, external_data


def split_data_non_iid(source, num_clients, alpha):
    X, y = source
    n_classes = np.max(y) + 1
    
    # Count the number of data points for each class
    class_counts = [np.sum(y == class_id) for class_id in range(n_classes)]

    # Distribute data to clients according to Dirichlet distribution
    client_data = [[] for _ in range(num_clients)]
    for class_id, count in enumerate(class_counts):
        # Generate a distribution over clients for data of this class
        distribution = np.random.dirichlet(np.ones(num_clients) * alpha)

        # Get all data of this class
        class_data = X[y == class_id]

        # Shuffle data of this class
        np.random.shuffle(class_data)

        # Assign data to clients according to distribution
        start = 0
        for client_id, fraction in enumerate(distribution):
            end = start + int(fraction * count)
            client_data[client_id].extend(class_data[start:end])
            start = end

    # Convert lists to numpy arrays
    client_data = [(np.array(data), y) for data, y in client_data]

    return client_data

def create_cnn_keras_model(input_shape, num_classes, weight_decay=0.0000, compile_model = True):

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(8, (5, 5),
                               activation="relu",
                               padding="same",
                            #    kernel_regularizer=regularizers.l2(weight_decay),
                               input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(16, (5, 5),
                               activation="relu",
                               padding="same"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(), 
        tf.keras.layers.Dense(64,
                              activation="relu"),
        tf.keras.layers.Dense(num_classes)
    ])
    
    if compile_model :
        model.compile(
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer="adam",
            metrics=['accuracy']
        )
    
    return model


def create_new_cnn_keras_model(input_shape, num_classes, weight_decay=0.0000, compile_model = True):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            16,
            8,
            strides=2,
            padding='same',
            activation='relu',
            input_shape= input_shape),
        tf.keras.layers.MaxPool2D(2, 1),
        tf.keras.layers.Conv2D(
            32, 4, strides=2, padding='valid', activation='relu'),
        tf.keras.layers.MaxPool2D(2, 1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    if compile_model:
        model.compile(
            loss= tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer="adam",
            metrics=['accuracy']
        )
    
    return model 


def create_nlp_keras_model(input_shape, num_classes, weight_decay=0.0000, compile_model = True):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=5000, output_dim=128, input_length=input_shape[0]),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(weight_decay)),
        tf.keras.layers.Dense(num_classes, kernel_regularizer=regularizers.l2(weight_decay))
    ])

    if compile_model:
        model.compile(
            loss= tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer="adam",
            metrics=['accuracy']
        )

    return model


def create_tabular_keras_model(input_shape, num_classes, weight_decay=0.0000, compile_model = True):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(weight_decay), input_shape=input_shape),
        tf.keras.layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(weight_decay)),
        tf.keras.layers.Dense(num_classes, kernel_regularizer=regularizers.l2(weight_decay))
    ])

    if compile_model:
        model.compile(
            loss= tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer="adam",
            metrics=['accuracy']
        )

    return model


def create_model_based_on_data(metadata = None, weight_decay=0.0000, compile_model = True):
    if metadata is None:
        input_shape = (HEIGHT, WIDTH, CHANNELS)
        num_classes = NUM_CLASSES
    else:
        input_shape = metadata.input_shape
        num_classes = metadata.num_classes

    if len(input_shape) == 1:  # Tabular data usually have 1D input shape
        return create_tabular_keras_model(input_shape, num_classes, weight_decay, compile_model)
    elif len(input_shape) == 2:  # Text data usually have 2D input shape (samples, sequence_length)
        return create_nlp_keras_model(input_shape, num_classes, weight_decay, compile_model)
    elif len(input_shape) == 3:  # Image data usually have 3D input shape (samples, height, width)
        return create_new_cnn_keras_model(input_shape, num_classes, weight_decay, compile_model)
    else: raise ValueError('Invalid input shape')



def attack_model_fn():
    """Attack model that takes target model predictions and predicts membership.

    Following the original paper, this attack model is specific to the class of the input.
    AttachModelBundle creates multiple instances of this model for each class.
    """
    
    model = AdaBoostClassifier(n_estimators=50, random_state=0)
    return model



# train keras model
def train_keras_model(model, train_data, test_data = None, epochs = 1, batch_size = 32, verbose=1, early_stop_patience=None, lr_reduction_patience=None, csv_logger_path=None):
    
    callbacks = []
    if early_stop_patience is not None and early_stop_patience > 0 : 
        early_stopping = EarlyStopping(monitor='val_loss', patience=early_stop_patience, restore_best_weights=True)
        callbacks.append(early_stopping)
    if lr_reduction_patience is not None and lr_reduction_patience > 0 :
        lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=lr_reduction_patience, verbose=1, factor=0.7, min_lr=0.000_01)
        callbacks.append(lr_reduction)
    if csv_logger_path is not None :
        csv_logger = CSVLogger(csv_logger_path)
        callbacks.append(csv_logger)

    # Create a tf.data.Dataset from your data
    train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
    train_dataset = train_dataset.shuffle(len(train_data[0])).batch(batch_size, drop_remainder=True)

    # If test_data is provided, create a tf.data.Dataset for it as well
    if test_data is not None:
        test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
        test_dataset = test_dataset.batch(batch_size, drop_remainder=True)
    else:
        test_dataset = None

    return model.fit(train_dataset, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_data=test_dataset, callbacks = callbacks)


# evaluate keras model
def test_keras_model(model, test_data, verbose=0):
    return model.evaluate(test_data[0], test_data[1], verbose=verbose)


def compile_model(model, args, loss_fn = None) : 
    if 'fed' in args.learning_algorithm : 
        epochs = args.rounds * args.local_epochs
    else : 
        epochs = args.local_epochs
    
    if args.use_dp : 
        ac = DPAccountant(
            data_size= args.local_size,
            batch_size= args.batch_size,
            epochs = epochs,
            target_delta = args.dp_delta, 
            dp_type = args.dp_type
        )
        sigma = ac.get_noise_multiplier(target_epsilon=args.dp_epsilon)
        args.sigma = sigma
        print("sigma : ", sigma)
        opt = DPKerasSGDOptimizer(
                    l2_norm_clip=args.dp_norm_clip,
                    noise_multiplier=sigma,
                    num_microbatches= args.batch_size,
                    learning_rate=args.lr)
        
        if loss_fn is not None :
            loss = loss_fn
        else :
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE)

    else : 
        args.sigma = 0
        opt = "adam"
        if loss_fn is not None :
            loss = loss_fn
        else :
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        loss=loss, 
        optimizer = opt,
        metrics=['accuracy']
        )
    return model

