import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow_datasets as tfds


def get_clip_box(image_a, image_b):
    image_size_x = image_a.shape[1]
    image_size_y = image_a.shape[0]

    x = tf.cast(tf.random.uniform([], 0, image_size_x), tf.int32)
    y = tf.cast(tf.random.uniform([], 0, image_size_y), tf.int32)

    width = tf.cast(image_size_x * tf.math.sqrt(1 - tf.random.uniform([], 0, 1)), tf.int32)
    height = tf.cast(image_size_y * tf.math.sqrt(1 - tf.random.uniform([], 0, 1)), tf.int32)

    x_min = tf.math.maximum(0, x - width // 2)
    y_min = tf.math.maximum(0, y - height // 2)
    x_max = tf.math.minimum(image_size_x, x + width // 2)
    y_max = tf.math.minimum(image_size_y, y + height // 2)

    return x_min, y_min, x_max, y_max


def mix_2_images(image_a, image_b, x_min, y_min, x_max, y_max):
    image_size_x = image_a.shape[1]
    image_size_y = image_a.shape[0]

    middle_left = image_a[y_min:y_max, 0:x_min, :]
    middle_center = image_b[y_min:y_max, x_min:x_max, :]
    middle_right = image_a[y_min:y_max, x_max:image_size_x, :]
    middle = tf.concat([middle_left, middle_center, middle_right], axis=1)

    top = image_a[0:y_min, :, :]
    bottom = image_a[y_max:image_size_y, :, :]
    mixed_img = tf.concat([top, middle, bottom], axis=0)

    return mixed_img


def mix_2_labels(image_a, image_b, label_a, label_b, x_min, y_min, x_max, y_max, num_classes=120):
    image_size_x = image_a.shape[1]
    image_size_y = image_a.shape[0]

    mixed_area = (x_max - x_min) * (y_max - y_min)
    total_area = image_size_x * image_size_y

    ratio = tf.cast(mixed_area / total_area, tf.float32)

    if len(label_a.shape) == 0:
        label_a = tf.one_hot(label_a, num_classes)
    if len(label_b.shape) == 0:
        label_b = tf.one_hot(label_b, num_classes)

    mixed_label = (1 - ratio) * label_a + ratio * label_b

    return mixed_label


def cutmix(image, label, prob=1.0, batch_size=16, img_size=224, num_classes=120):
    mixed_imgs = []
    mixed_labels = []

    for i in range(batch_size):
        image_a = image[i]
        label_a = label[i]

        j = tf.cast(tf.random.uniform([], 0, batch_size), tf.int32)

        image_b = image[j]
        label_b = label[j]

        x_min, y_min, x_max, y_max = get_clip_box(image_a, image_b)

        mixed_imgs.append(mix_2_images(image_a, image_b, x_min, y_min, x_max, y_max))
        mixed_labels.append(mix_2_labels(image_a, image_b, label_a, label_b, x_min, y_min, x_max, y_max))

    mixed_imgs = tf.reshape(tf.stack(mixed_imgs), (batch_size, img_size, img_size, 3))
    mixed_labels = tf.reshape(tf.stack(mixed_labels), (batch_size, num_classes))

    return mixed_imgs, mixed_labels


def mixup_2_images(image_a, image_b, label_a, label_b):
    ratio = tf.random.uniform([], 0, 1)

    if len(label_a.shape) == 0:
        label_a = tf.one_hot(label_a, num_classes)
    if len(label_b.shape) == 0:
        label_b = tf.one_hot(label_b, num_classes)

    mixed_image = (1 - ratio) * image_a + ratio * image_b
    mixed_label = (1 - ratio) * label_a + ratio * label_b

    return mixed_image, mixed_label


def mixup(image, label, prob=1.0, batch_size=16, img_size=224, num_classes=120):
    mixed_imgs = []
    mixed_labels = []

    for i in range(batch_size):
        image_a = image[i]
        label_a = label[i]

        j = tf.cast(tf.random.uniform([], 0, batch_size), tf.int32)

        image_b = image[j]
        label_b = label[j]

        mixed_img, mixed_label = mixup_2_images(image_a, image_b, label_a, label_b)
        mixed_imgs.append(mixed_img)
        mixed_labels.append(mixed_label)

    mixed_imgs = tf.reshape(tf.stack(mixed_imgs), (batch_size, img_size, img_size, 3))
    mixed_labels = tf.reshape(tf.stack(mixed_labels), (batch_size, num_classes))

    return mixed_imgs, mixed_labels


def normalize_and_resize_img(image, label):
    image = tf.image.resize(image, [224, 224])
    return tf.cast(image, tf.float32) / 255., label


def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.clip_by_value(image, 0, 1)
    return image, label


def onehot(image, label):
    label = tf.one_hot(label, 120)
    return image, label


def apply_normalize_on_dataset(ds, is_test=False, batch_size=16, with_aug=False, with_cutmix=False, with_mixup=False,
                               one_hot=False):
    ds = ds.map(
        normalize_and_resize_img,
        num_parallel_calls=2
    )

    ds = ds.batch(batch_size)

    if with_aug:
        ds = ds.map(
            augment,
            num_parallel_calls=2
        )

    if with_cutmix:
        ds = ds.map(
            cutmix,
            num_parallel_calls=2
        )

    if with_mixup:
        ds = ds.map(
            mixup,
            num_parallel_calls=2
        )

    ds = ds.repeat()
    ds = ds.shuffle(200)

    if one_hot:
        ds = ds.map(
            onehot,
            num_parallel_calls=2
        )

    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def train(model, ds_train, ds_test, ds_info, EPOCH):
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.SGD(lr=0.01),
                  metrics=['accuracy'])

    history = model.fit(ds_train,
                        steps_per_epoch=int(ds_info.splits['train'].num_examples/16),
                        validation_steps=int(ds_info.splits['test'].num_examples/16),
                        epochs=EPOCH,
                        validation_data=ds_test,
                        use_multiprocessing=True)
    return history


if __name__ == '__main__':
    EPOCH = 20
    aug_list = ["no_aug", "aug", "cutmix", "mixup"]
    history = []

    (ds_train, ds_test), ds_info = tfds.load(
        'stanford_dogs',
        split=['train', 'test'],
        shuffle_files=True,
        with_info=True,
        as_supervised=True
    )

    num_classes = ds_info.features["label"].num_classes

    resnet50 = tf.keras.models.Sequential([
        tf.keras.applications.resnet.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3),
            pooling='avg',
        ),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    ds_trains = [apply_normalize_on_dataset(ds_train, one_hot=True),
                 apply_normalize_on_dataset(ds_train, with_aug=True, one_hot=True),
                 apply_normalize_on_dataset(ds_train, with_cutmix=True),
                 apply_normalize_on_dataset(ds_train, with_mixup=True)]
    ds_test = apply_normalize_on_dataset(ds_test, is_test=True, one_hot=True)

    for i in range(4):
        history.append(train(resnet50, ds_trains[i], ds_test, ds_info, EPOCH))
    #loss: 0.0040 - accuracy: 0.9998 - val_loss: 1.0317 - val_accuracy: 0.7369
    #loss: 0.0031 - accuracy: 0.9999 - val_loss: 1.0581 - val_accuracy: 0.7445
    #loss: 1.4911 - accuracy: 0.8323 - val_loss: 1.5550 - val_accuracy: 0.6566
    #loss: 1.5267 - accuracy: 0.8846 - val_loss: 1.5130 - val_accuracy: 0.6553
