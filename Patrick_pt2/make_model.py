# set the matplotlib backend so figures can be saved in the background
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import os
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import time as tm
from tensorflow.keras.applications import VGG16
import json

from imagesearch import config

matplotlib.use("Agg")


def train_data(cluster):
    train_dir = os.path.sep.join([config.DATASET_DIR, cluster, config.TRAIN])
    validation_dir = os.path.sep.join([config.DATASET_DIR, cluster, config.VAL])
    sp_counted = len(os.listdir(train_dir))

    vgg_conv = VGG16(weights='imagenet',
                     include_top=False,
                     input_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, 3))

    for layer in vgg_conv.layers[:-4]:
        layer.trainable = False

    for layer in vgg_conv.layers:
        print(layer, layer.trainable)

    # Create the model
    model = models.Sequential()

    # Add the vgg convolutional base model
    model.add(vgg_conv)

    # Add new layers
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(sp_counted, activation='softmax'))

    # Show a summary of the model. Check the number of trainable parameters
    model.summary()

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Data Generator for Training data
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical'
    )

    # Data Generator for Validation data
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False)

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    # Train the Model
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=2 * train_generator.samples / train_generator.batch_size,
        epochs=config.EPOCH,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples / validation_generator.batch_size,
        verbose=1)

    # Save label and model
    label2index = train_generator.class_indices
    idx2label = dict((v, [k]) for k, v in label2index.items())
    with open(os.path.sep.join([config.MODEL_PATH, cluster + '.txt']), 'w') as f:
        json.dump(idx2label, f)

    model.save(os.path.sep.join([config.MODEL_PATH, cluster + '.h5']))

    score = model.evaluate_generator(
        validation_generator,
        steps=validation_generator.samples / validation_generator.batch_size
    )
    print('Loss \t\t:', score[0])
    print('Accuracy \t:', score[1] * 100, '%')
    eval_result = {
        'Loss': score[0],
        'Accuracy': str(score[1] * 100) + '%'
    }
    with open(os.path.sep.join([config.MODEL_PATH, cluster + '_eval.txt']), 'w') as f:
        json.dump(eval_result, f)

    return history


def plot_training_result(history, cluster):
    # Plot the accuracy and loss curves
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, config.EPOCH), loss, label="train_loss")
    plt.plot(np.arange(0, config.EPOCH), val_loss, label="val_loss")
    plt.plot(np.arange(0, config.EPOCH), acc, label="train_acc")
    plt.plot(np.arange(0, config.EPOCH), val_acc, label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(os.path.sep.join([config.MODEL_PATH, cluster + '.png']))


def execute(cluster):
    start = tm.time()

    history = train_data(cluster)
    plot_training_result(history, cluster)

    print('Make model done.....')
    end = tm.time()
    menit = (end - start) / 60
    print('Time spent => ', menit, ' minutes')


execute(config.CLUSTER[3])
