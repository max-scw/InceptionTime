# Inception time: ensemble of Inception models
import keras
from keras.layers import Conv1D, MaxPool1D, Concatenate, Activation, Add, Input, GlobalAveragePooling1D, Dense
from keras.layers.normalization.batch_normalization import BatchNormalization

from typing import Union

import pandas as pd
import numpy as np
import pathlib as pl

class InceptionTime1:

    def __init__(self, output_directory, input_shape, n_classes, verbose=False, build=True, batch_size=64,
                 n_filters=32, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41, n_epochs=1500):

        self.output_directory = output_directory

        self.n_filters = n_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.callbacks = None
        self.batch_size = batch_size
        self.bottleneck_size = 32
        self.n_epochs = n_epochs

        if build:
            self.model = self.build_model(input_shape, n_classes)
            if verbose:
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init.hdf5')

    def _inception_module(self, input_tensor, stride:int = 1, activation:str = 'linear'):
        # the inception module is a bottleneck operation followed by 3 parallel convolutions and a maximum pooling
        # operation followed by a convolution with kernel size 1
        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]
        # create a list of multiple, distinct convolutions on same input (that is the output of input_inception)
        conv_list = []
        for i in range(len(kernel_size_s)):
            conv_list.append(Conv1D(filters=self.n_filters, kernel_size=kernel_size_s[i],
                                    strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))
        # parallel path: add maximum pooling to same input (that is the output of input_inception)
        max_pool_1 = MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)
        # convolve the output of max-pooling with kernel size 1
        conv_6 = Conv1D(filters=self.n_filters, kernel_size=1,
                        padding='same', activation=activation, use_bias=False)(max_pool_1)
        # append to list of operations
        conv_list.append(conv_6)

        # create inception module: concatenate all operations that they run in parallel and add batch normalization for
        # better training (vanishing gradient problem)
        inception_block = Concatenate(axis=2)(conv_list)
        inception_block = BatchNormalization()(inception_block)
        # set activation functions to ReLU
        inception_block = Activation(activation='relu')(inception_block)
        return inception_block

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        shortcut_y = BatchNormalization()(shortcut_y)

        block = Add()([shortcut_y, out_tensor])
        block = Activation('relu')(block)
        return block

    def build_model(self, input_shape: tuple, n_classes: int) -> keras.Model:
        # define shape of the expected input
        input_layer = Input(input_shape)

        x = input_layer
        input_res = input_layer
        
        # stack inceltion modules / blocks
        for d in range(self.depth):
            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x
        # penultimate layer is a global average pooling layer
        gap_layer = GlobalAveragePooling1D()(x)
        # output layer is a dense softmax layer for classification
        output_layer = Dense(n_classes, activation='softmax')(gap_layer)
        # stack all layers together to a model
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        # compile setting loss function and optimizer
        model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        # construct / set callbacks
        #reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
        #                                              min_lr=0.0001)
        #
        #file_path = self.output_directory + 'best_model.hdf5'
        #
        #model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
        #                                                   save_best_only=True)
        self.callbacks = None  #[reduce_lr, model_checkpoint]

        return model

    def fit(self, 
            x_train: Union[np.ndarray, pd.Series, pd.DataFrame], 
            y_train: Union[np.ndarray, pd.Series],
            x_val: Union[np.ndarray, pd.Series, pd.DataFrame] = None, 
            y_val: Union[np.ndarray, pd.Series] = None) -> keras.Model:
        # x_val and y_val are only used to monitor the test loss and NOT for training

        if self.batch_size is None:
            mini_batch_size = int(min(x_train.shape[0] / 10, 16))
        else:
            mini_batch_size = self.batch_size
        mini_batch_size = None  # FIXME

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.n_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

        return self.model

    def predict(self, x: Union[np.ndarray, pd.Series, pd.DataFrame]) -> Union[np.ndarray, pd.Series]:
        y_prd = self.model.predict(x, batch_size=self.batch_size)
        return y_prd


if __name__ == '__main__':
    path_to_working_directory = pl.Path.cwd()
    print(f'current working directory: {path_to_working_directory}')
    path_to_data = pl.Path(r'archive/UCRArchive_2018/ChlorineConcentration/ChlorineConcentration_TRAIN.tsv')
    yx = np.loadtxt(path_to_data)
    x = yx[:, 1:]
    y = yx[:, 0] -1

    mdl = InceptionTime1(output_directory=path_to_working_directory.as_posix(), input_shape=(x.shape[1], 1),
                        n_classes=len(np.unique(y)), verbose=True, use_bottleneck=True, use_residual=False)
    mdl.fit(x, y)
