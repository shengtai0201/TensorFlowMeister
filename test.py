import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow as tf
import tensorflow_datasets as tfds

#載入資料
tfds.disable_progress_bar()
dataset, metadata = tfds.load(
    'fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

#建立標籤
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#檢視資料
num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples: {}".format(num_test_examples))


def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


#預處理資料
train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)
train_dataset = train_dataset.cache()
test_dataset = test_dataset.cache()

#檢視圖片
for image, label in test_dataset.take(1):
    image = image.numpy().reshape((28, 28))

    plt.figure()
    plt.imshow(image, cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid(False)
    plt.show()

plt.figure(figsize=(10, 10))
i = 0
for (image, label) in test_dataset.take(25):
    image = image.numpy().reshape((28, 28))

    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_names[label])
    i += 1

plt.show()

#建立模型
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),  # 輸入層
    tf.keras.layers.Dense(128, activation=tf.nn.relu),  # 隱藏層
    tf.keras.layers.Dense(10)  # 輸出層
])

#編譯模型
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True), metrics=['accuracy'])

#訓練模型
BATCH_SIZE = 32
train_dataset = train_dataset.cache().repeat().shuffle(
    num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)
model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(
    num_train_examples/BATCH_SIZE))

#評估準確度
test_loss, test_accuracy = model.evaluate(
    test_dataset, steps=math.ceil(num_test_examples/32))
print('Accuracy on test dataset:', test_accuracy)

#進行預測
for test_images, test_labels in test_dataset.take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)

    predictions[0]
    np.argmax(predictions[0])
    test_labels[0]
