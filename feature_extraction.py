import cv2
import tensorflow as tf
from tensorflow import keras
from DLCVDatasets import get_dataset
from HogFeatures import compute_hog

if __name__ == '__main__':
    training_size = 60000
    test_size = 10000
    x_train, y_train, x_test, y_test, class_names = get_dataset(
        dataset='mnist', training_size=training_size, test_size=test_size)

# %% Extract hog features
    cell_size = (4,4)
    block_size = (2,2)
    nbins = 36
    hog_train = compute_hog(cell_size, block_size, nbins, x_train)
    hog_test = compute_hog(cell_size, block_size, nbins, x_test)
    # Need to normalize hog features?

    # Set up the model
    model = keras.models.Sequential([
        keras.layers.Dense(units=40, activation='relu', input_shape=[hog_train.shape[1]]),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(units=30, activation='relu'),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(units=20, activation='relu'),
        keras.layers.Dense(units=10, activation='relu'),
        keras.layers.Softmax()  # Softmax() has the same shape as its input
    ])

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', 
        optimizer=keras.optimizers.Adam(lr=0.001), 
        metrics=['acc'])

    # Train model
    # Use the callback to perform early stopping
    class MyCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs.get('val_acc') > 0.97:
                print('\nReached 97% accuracy so cancelling training!')
                self.model.stop_training = True

    myCallback = MyCallback()

    history = model.fit(hog_train, y_train, epochs=20, batch_size=64, callbacks=[myCallback], 
        validation_data=(hog_test, y_test))

# %%
    # Inspect result
    import numpy as np
    import matplotlib.pyplot as plt
    test_idx = 20
    hog_feature = np.array([hog_test[test_idx, :]]) # make it 2-d
    plt.imshow(x_test[test_idx], cmap='gray')
    plt.show()

    classes = model.predict(hog_feature)
    class_est = np.argmax(classes)
    print('Predicted classes: {}'.format(class_est))

    


# %%
