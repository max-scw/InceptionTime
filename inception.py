# Inception time: ensemble of Inception models
from types import new_class
import keras
from keras.layers import Conv1D, MaxPool1D, Concatenate, Activation, Add, Input, GlobalAveragePooling1D, Dense
from keras.layers.normalization.batch_normalization import BatchNormalization

import h5py

from typing import Union, List, Tuple

import pandas as pd
import numpy as np
import pathlib as pl


class InceptionTime1:
    _suffix_label_categories = '_label_categories.h5'
    _suffix_initial_weights = '_init.h5'
    _suffix_best_model  = '_best_model.h5'
    _h5_key_label_categories = 'label_categories'

    def __init__(
        self,
        output_directory: Union[str, pl.Path],  # TODO:make optional
        input_shape: Tuple[int],
        n_classes: int,
        verbose: bool = False,
        build: bool = True,
        batch_size: int = 64,
        n_filters: int = 32,
        use_residual: bool = True,
        use_bottleneck: bool = True,
        depth: int = 6,
        kernel_size: int = 41,
        n_epochs: int = 1500,
        model_name: str = None,
    ) -> None:
        # set required input parameters
        self.output_directory = pl.Path(output_directory)
        # set optional input parameters
        self.n_filters = n_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.callbacks = None
        self.batch_size = batch_size
        self.bottleneck_size = 32
        self.n_epochs = n_epochs
        self.y_categories = None
        if isinstance(model_name, str) and not model_name == "":
            self.model_name = model_name
        else:
            self.model_name = "InceptionTime1"

        if build:
            self.model = self.build_model(input_shape, n_classes)
            # if verbose:
            #    self.model.summary()
            self.verbose = verbose
            file_name = self.model_name + self._suffix_initial_weights
            self.model.save_weights(self.output_directory.joinpath(file_name))

    def _inception_module(
        self, input_tensor: keras.dtensor, stride: int = 1, activation: str = "linear"
    ) -> keras.dtensor:
        # the inception module is a bottleneck operation followed by 3 parallel convolutions and a maximum pooling
        # operation followed by a convolution with kernel size 1
        # only apply bottleneck operation if the input is multivariante data!
        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = Conv1D(
                filters=self.bottleneck_size,
                kernel_size=1,
                input_shape=input_tensor.shape[1:],
                padding="same",
                activation=activation,
                use_bias=False,
            )(input_tensor)
        else:
            input_inception = input_tensor

        # create list of kernel sizes of the convolutions (100%, 50%, 25% of the input kernel_size)
        kernel_size_s = [self.kernel_size // (2**i) for i in range(3)]
        # create a list of multiple, distinct convolutions on same input (that is the output of input_inception)
        conv_list = []
        for i in range(len(kernel_size_s)):
            conv_list.append(
                Conv1D(
                    filters=self.n_filters,
                    kernel_size=kernel_size_s[i],
                    strides=stride,
                    padding="same",
                    activation=activation,
                    use_bias=False,
                )(input_inception)
            )
        # parallel path: add maximum pooling to same input (that is the output of input_inception)
        max_pool_1 = MaxPool1D(pool_size=3, strides=stride, padding="same")(input_tensor)
        # convolve the output of max-pooling with kernel size 1 (this is basically a scaling)
        conv_6 = Conv1D(filters=self.n_filters, kernel_size=1, padding="same", activation=activation, use_bias=False)(
            max_pool_1
        )
        # append to list of operations
        conv_list.append(conv_6)

        # create inception module: concatenate all operations that they run in parallel and add batch normalization for
        # better training (vanishing gradient problem)
        inception_block = Concatenate(axis=2)(conv_list)
        inception_block = BatchNormalization()(inception_block)
        # set activation functions to ReLU
        inception_block = Activation(activation="relu")(inception_block)
        return inception_block

    def _shortcut_layer(self, input_tensor, out_tensor):
        # 1D convolution followed by batch normalization in parallel to "normal" input-output
        n_filters = int(out_tensor.shape[-1])
        shortcut_y = Conv1D(filters=n_filters, kernel_size=1, padding="same", use_bias=False)(input_tensor)
        shortcut_y = BatchNormalization()(shortcut_y)

        # put shortcut in parallel to the "normal" layer
        block = Add()([shortcut_y, out_tensor])
        block = Activation("relu")(block)
        return block

    def build_model(self, input_shape: Tuple[int], n_classes: int) -> keras.Model:
        # define shape of the expected input
        input_layer = Input(shape=input_shape)

        # initialize first layer as the input layer
        x = input_layer
        # initialize short-cut layer with input layer
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
        output_layer = Dense(n_classes, activation="softmax")(gap_layer)
        # stack all layers together to a model
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        # compile setting loss function and optimizer
        model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(), metrics=["accuracy"])

        # construct / set callbacks
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=50, min_lr=0.0001)
        # add checkpoints
        file_name = self.model_name + self._suffix_best_model
        file_path = self.output_directory.joinpath(file_name)
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor="loss", save_best_only=True)
        # set callbacks
        self.callbacks = [reduce_lr, model_checkpoint]

        return model

    def load_model(self, path_to_model: Union[str, pl.Path]) -> bool:
        # load model
        self.model = keras.models.load_model(path_to_model)
        # load label categories
        with h5py.File(path_to_model, 'r') as fl:
            self.y_categories = pd.Categorical(fl[self._h5_key_label_categories])

    def fit(
        self,
        x_train: Union[np.ndarray, pd.Series, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        x_val: Union[np.ndarray, pd.Series, pd.DataFrame] = None,
        y_val: Union[np.ndarray, pd.Series] = None,
    ) -> keras.Model:
        # x_val and y_val are only used to monitor the test loss and NOT for training

        # convert label input (y) to categoricals and store for backtransformation
        y_train = pd.Categorical(y_train)
        self.y_categories = y_train.categories
        # extract category codes because keras only handles increasing integers as classes starting at 0
        y_train = y_train.codes
        y_val = pd.Categorical(y_val, categories=self.y_categories).codes

        if self.batch_size is None:
            mini_batch_size = int(min(x_train.shape[0] / 10, 16))
        else:
            mini_batch_size = self.batch_size

        if x_val is not None and y_val is not None:
            validation_data = (x_val, y_val)
        else:
            validation_data = None

        hist = self.model.fit(
            x_train,
            y_train,
            batch_size=mini_batch_size,
            epochs=self.n_epochs,
            verbose=self.verbose,
            validation_data=validation_data,
            callbacks=self.callbacks,
        )

        # add label categories to output file
        file_name = self.model_name + self._suffix_best_model
        with h5py.File(file_name, 'a') as fl:
            fl.create_dataset(self._h5_key_label_categories, data=self.y_categories)

        return self.model

    def predict(self, x: Union[np.ndarray, pd.Series, pd.DataFrame]) -> Union[np.ndarray, pd.Series]:
        if self.y_categories is None:
            raise ValueError('No label categories found. Perhaps the model was not trained yet?')
        y_prd_codes = self.model.predict(x, batch_size=self.batch_size).argmax(axis=1)
        # transform back to categorical series
        y_prd = pd.Categorical.from_codes(y_prd_codes, categories=self.y_categories)
        return y_prd


class InceptionTimeEnsemble:
    """
    This is an ensemble of InceptionTime1 models. In the original paper the classifieres was named "NNE" (= Neural Network Ensemble) but relied on pretrained models loaded from disk. In contrast to this, this class actually traines the models out of the box.
    """

    def __init__(
        self,
        output_directory: Union[str, pl.Path],
        verbose: bool = False,
        n_ensemble_members: int = 5,
        n_epochs: int = 1500,
        model_name_prefix: str = ''
    ):

        # required input parameters
        self.output_directory = pl.Path(output_directory)

        # optional input parameters
        self.verbose = verbose
        # ensemble of n InceptionTime1 models
        self.n_ensemble_members = n_ensemble_members
        self.n_epochs = n_epochs
        self.model_name = '_'.join(filter(None, [model_name_prefix, "InceptionTime1", "Nr"]))

    def fit(
        self,
        x_train: Union[np.ndarray, pd.Series, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        x_val: Union[np.ndarray, pd.Series, pd.DataFrame] = None,
        y_val: Union[np.ndarray, pd.Series] = None,
    ):

        # determine input shape from input
        n_observations = x_train.shape[0]
        x_time_len = x_train.shape[1]
        x_signal_dim = tuple([1 if x_train.shape[2:] == () else x_train.shape[2:]])
        input_shape = (x_time_len,) + x_signal_dim
        # determine number of classes from input
        n_classes = len(pd.unique(y_train))

        for i in range(self.n_ensemble_members):
            print(f"Training InceptionTime1 model Nr. {i} ...")
            model_name = self.model_name + str(i)
            # initialize new instance of a single inception time
            # make sure that it is initialized differently thatn the others
            model = InceptionTime1(
                output_directory=self.output_directory,
                input_shape=input_shape,
                n_classes=n_classes,
                verbose=self.verbose,
                n_epochs=self.n_epochs,
                model_name=model_name,
            )
            # train / fit model
            model.fit(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
            # save model
            # keras.models.save_model(model.model, filepath=self.output_directory.joinpath(model_name + ".h5"))
            # already done with the checkpoints of the callback functions in InceptionTime1

            # free model instance
            keras.backend.clear_session()
        print(
            f"Done with training InceptionTime (an ensemble of {self.n_ensemble_members} InceptionTime1 models). All models saved to {self.output_directory}"
        )

        # calculate validation metrices of the ensemble
        if x_val is not None and y_val is not None:
            y_prd = self.predict(x_val)
            # TODO: calculate metrics

    def predict(self, x, detailed_output: bool = False):

        # dummy model to access internal variables
        mdl_dummy = InceptionTime1('',-1, -1, build=False)

        # loop through models for individual predictions
        y_prds = {}
        for i in range(self.n_ensemble_members):
            # load model
            file_name = self.model_name + str(i) + mdl_dummy._suffix_best_model 
            model = keras.models.load_model(self.output_directory.joinpath(file_name))
            # then compute the predictions
            y_prd_i = model.predict(x).argmax(axis=1)
            #keras.backend.clear_session()
            y_prds[file_name] = y_prd_i
        # create dataframe from stacked series
        y_prds = pd.DataFrame(y_prds)

        # load label categories
        with h5py.File(file_name, 'r') as fl:
            y_categories = pd.Categorical(fl[mdl_dummy._h5_key_label_categories])

        if detailed_output:
            # todo transform back
            return y_prds.apply(lambda x: pd.Categorical.from_codes(x, categories=y_categories))
        else:
            # average category codes
            y_prd = y_prds.mean(axis=1).apply(round)
            # transform back to categorical series
            # TODO: load categories
            return pd.Categorical.from_codes(y_prd, categories=y_categories)


if __name__ == "__main__":
    path_to_working_directory = pl.Path.cwd()
    print(f"current working directory: {path_to_working_directory}")
    path_to_archive = pl.Path(r"archive/UCRArchive_2018/")
    name_dataset = "ChlorineConcentration"

    path_to_dataset = path_to_archive.joinpath(name_dataset)
    yx = np.loadtxt(path_to_dataset.joinpath(name_dataset + "_TRAIN.tsv"))

    x_train = yx[:, 1:]
    y_train = yx[:, 0] - 10

    yx = np.loadtxt(path_to_dataset.joinpath(name_dataset + "_TEST.tsv"))

    x_test = yx[:, 1:]
    y_test = yx[:, 0] - 10
    """
    n_observations = x_train.shape[0]
    x_time_len = x_train.shape[1]
    x_signal_dim = tuple([1 if x_train.shape[2:] == () else x_train.shape[2:]])
    input_shape = (x_time_len,) + x_signal_dim

    
    mdl = InceptionTime1(
        output_directory=path_to_working_directory,
        input_shape=input_shape,
        n_classes=len(np.unique(y_train)),
        build=False,
        verbose=True,
        depth=6,
        use_bottleneck=True,
        use_residual=True,
        n_epochs=1,  # FIMXE: for testing only
    )
    # shape (observations, time-siwe signal length, signal dimensions): (467, 166) => dimension: (166, 1)
    #mdl.fit(x_train, y_train, x_test, y_test)
    mdl.load_model(path_to_working_directory.joinpath('InceptionTime1_best_model.h5'))
    mdl.predict(x_test)
    """
    ensemble = InceptionTimeEnsemble(output_directory=path_to_working_directory, n_epochs=1, verbose=True, n_ensemble_members=3)
    #ensemble.fit(x_train, y_train, x_test, y_test)

    ensemble.predict(x_test)

