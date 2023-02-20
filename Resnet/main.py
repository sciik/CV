import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import urllib3
urllib3.disable_warnings()
# tfds.disable_progress_bar()


def normalize(image, label):
    return tf.cast(image, tf.float32) / 255., label


def apply_normalize_on_dataset(ds, is_test=False, batch_size=16):
    ds = ds.map(normalize, num_parallel_calls=1)
    ds = ds.batch(batch_size)

    if not is_test:
        ds = ds.repeat()
        ds = ds.shuffle(200)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def build_sequence_block(input_layer, channel):
    x = input_layer
    x = tf.keras.layers.Conv2D(channel, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(channel, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x


def build_bottelneck_block(input_layer, channel):
    x = input_layer
    x = tf.keras.layers.Conv2D(channel, (1, 1), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(channel, (3, 3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(channel*4, (1, 1), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x


def build_residual_block(input_layer, channel, channel_num, in_50=False):
    x = input_layer
    residual = input_layer

    if in_50:
        for i in range(channel_num):
            x = build_bottelneck_block(x, channel)
            if i < 3:
                x = tf.keras.layers.MaxPooling2D((2, 2))(x)
                residual = tf.keras.layers.Conv2D(channel, (1, 1), strides=2)(residual)
            else:
                residual = tf.keras.layers.Conv2D(channel*2, (1, 1), padding="same")(residual)
            residual = tf.keras.layers.BatchNormalization()(residual)
            residual = tf.keras.layers.Activation("relu")(residual)
            residual = tf.keras.layers.Conv2D(channel*4, (1, 1), padding="same")(residual)
            residual = tf.keras.layers.BatchNormalization()(residual)

            x = tf.keras.layers.add([x, residual])
            x = tf.keras.layers.Activation("relu")(x)
    else:
        for i in range(channel_num):
            x = build_sequence_block(x, channel)
            if i < 3:
                x = tf.keras.layers.MaxPooling2D((2, 2))(x)
                residual = tf.keras.layers.Conv2D(channel, (1, 1), strides=2)(residual)
            else:
                residual = tf.keras.layers.Conv2D(channel, (1, 1), padding="same")(residual)
            residual = tf.keras.layers.BatchNormalization()(residual)
            residual = tf.keras.layers.Activation("relu")(residual)
            residual = tf.keras.layers.Conv2D(channel, (1, 1), padding="same")(residual)
            residual = tf.keras.layers.BatchNormalization()(residual)

            x = tf.keras.layers.Add()([x, residual])
            x = tf.keras.layers.Activation("relu")(x)

    return x


def build_resnet(input_shape=(32, 32, 3), in_50=False):
    input_layer = tf.keras.Input(shape=input_shape)
    x = input_layer

    channel_list = [32, 64, 128]
    channel_num_list = [1, 1, 1]

    for channel, channel_num in zip(channel_list, channel_num_list):
        x = build_residual_block(x, channel, channel_num, in_50)

    output = tf.keras.layers.GlobalAveragePooling2D()(x)
    output = tf.keras.layers.Dense(10, activation="relu")(output)
    output = tf.keras.layers.Activation("softmax")(output)

    model = tf.keras.Model(inputs=input_layer, outputs=output)

    return model


def build_plainnet(input_shape=(32, 32, 3), in_50=False):
    input_layer = tf.keras.Input(shape=input_shape)
    x = input_layer

    channel_list = [32, 64, 128]
    channel_num_list = [1, 1, 1]

    for channel, channel_num in zip(channel_list, channel_num_list):
        if in_50:
            for i in range(channel_num):
                x = build_bottelneck_block(x, channel)
                x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        else:
            for i in range(channel_num):
                x = build_sequence_block(x, channel)
                x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        output = tf.keras.layers.GlobalAveragePooling2D()(x)
        output = tf.keras.layers.Dense(10, activation="relu")(output)
        output = tf.keras.layers.Activation("softmax")(output)

        model = tf.keras.Model(inputs=input_layer, outputs=output)
    return model


def train(model, ds_train, ds_test, ds_info, BATCH_SIZE, EPOCH):
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="SGD",
                  metrics=["accuracy"])
    history = model.fit(ds_train,
              steps_per_epoch=int(ds_info.splits["train"].num_examples/BATCH_SIZE),
              validation_steps=int(ds_info.splits["test"].num_examples/BATCH_SIZE),
              epochs=EPOCH,
              validation_data=ds_test,
              use_multiprocessing=True)

    return history


def draw_plot(histories):
    for history in histories:
        plt.plot(history.history["loss"])
    plt.title("model loss")
    plt.show()


if __name__ == '__main__':
    BATCH_SIZE = 256
    EPOCH = 30
    history = []

    (ds_train, ds_test), ds_info = tfds.load(
        "cifar10",
        split=["train", "test"],
        as_supervised=True,
        shuffle_files=True,
        with_info=True
    )

    ds_train = apply_normalize_on_dataset(ds_train, batch_size=BATCH_SIZE)
    ds_test = apply_normalize_on_dataset(ds_test, batch_size=BATCH_SIZE)

    resnet32 = build_resnet(in_50=False)
    resnet32.summary()

    resnet50 = build_resnet(in_50=True)
    resnet50.summary()

    plainnet32 = build_plainnet(in_50=False)
    plainnet32.summary()

    plainnet50 = build_plainnet(in_50=True)
    plainnet50.summary()

    history.append(train(resnet32, ds_train, ds_test, ds_info, BATCH_SIZE, EPOCH))
    history.append(train(resnet50, ds_train, ds_test, ds_info, BATCH_SIZE, EPOCH))
    history.append(train(plainnet32, ds_train, ds_test, ds_info, BATCH_SIZE, EPOCH))
    history.append(train(plainnet50, ds_train, ds_test, ds_info, BATCH_SIZE, EPOCH))

    draw_plot(history)
    #loss: 0.8556 - accuracy: 0.7143 - val_loss: 1.1405 - val_accuracy: 0.6126
    #loss: 0.6788 - accuracy: 0.7690 - val_loss: 0.9538 - val_accuracy: 0.6626
    #loss: 0.7724 - accuracy: 0.7351 - val_loss: 1.1030 - val_accuracy: 0.6127
    #loss: 1.1118 - accuracy: 0.6211 - val_loss: 1.3476 - val_accuracy: 0.5361