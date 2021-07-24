import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import add
from keras.layers import Activation
from keras.layers import UpSampling2D
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import Lambda
from keras.layers import multiply
from keras.layers import concatenate

# Attention Unet
class Attention_UNet():
    def __init__(self, input_shape, num_classes=1, dropout=0, BatchNorm=True):
        super(Attention_UNet, self).__init__()
        self.max_pooling = True
        self.num_classes = num_classes
        self.input_shape = input_shape
        # number of basic filters for the first layer
        self.num_filters = 64
        # size of the convolutional filter
        self.filter_size = 3
        # size of upsampling filters
        self.upsampling_filter = 2


    def make_conv_block(self, input_layer, filter_size, num_filters, dropout=0, BatchNorm=False):
        conv_layer = Conv2D(num_filters, (filter_size, filter_size), padding="same")(input_layer)
        if BatchNorm is True:
            conv_layer = BatchNormalization(axis=3)(conv_layer)
        conv_layer = Activation("relu")(conv_layer)

        conv_layer = Conv2D(num_filters, (filter_size, filter_size), padding="same")(conv_layer)
        if BatchNorm is True:
            conv_layer = BatchNormalization(axis=3)(conv_layer)
        conv_layer = Activation("relu")(conv_layer)

        if dropout > 0:
            conv_layer = Dropout(dropout)(conv_layer)

        return conv_layer


    def make_repeat_elements(self, tensor, rep):
        """
           This function will repeat the elements of a tensor along an axis through a factor of rep using lambda function.
           For instance, if tensor has shape (None, 256,256,3), lambda will return a tensor of shape (None, 256,256,6),
           if specified axis=3 and rep=2
        """
        return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={"repnum":rep})(tensor)


    def _gate_signal(self, input_channels, output_channels, BatchNorm=False):
        """
            This function will resize the downsample layer feature map into the same dimension as the upsample
            layer feature map through 1x1 convolution

           return:
                the gating feature map with the same dimension of the up layer feature map
        """
        g = Conv2D(output_channels, (1,1), padding="same")(input_channels)
        if BatchNorm:
            g = BatchNormalization()(g)
        g = Activation("relu")(g)

        return g

    # We add attention block after shortcut connection in UNet
    def make_attention_block(self, input_layer, gating, num_filters):
        input_layer_shape = K.int_shape(input_layer)
        gating_shape = K.int_shape(gating)

        # Here, we should get the input_layer signal to the same shape as the gating signal
        input_layer_theta = Conv2D(num_filters, (2, 2), strides = (2, 2), padding = "same")(input_layer)
        input_layer_theta_shape = K.int_shape(input_layer_theta)

        # we should get the gating signal to the same number of filters as the num_filters
        gating_phi = Conv2D(num_filters, (1, 1), padding = "same")(gating)
        gating_upsample = Conv2DTranspose(num_filters,
                                          (3, 3),
                                          strides = (input_layer_theta_shape[1] // gating_shape[1],
                                                     input_layer_theta_shape[2] // gating_shape[2]),
                                          padding = "same")(gating_phi)
        concat_layer = add([gating_upsample, input_layer_theta])
        concat_layer = Activation("relu")(concat_layer)
        concat_layer = Conv2D(1, (1, 1), padding="same")(concat_layer)
        concat_layer = Activation("sigmoid")(concat_layer)   # To get weigth between 0 and 1
        concat_layer_shape = K.int_shape(concat_layer)
        concat_layer_upsampling = UpSampling2D(size = (input_layer_shape[1] // concat_layer_shape[1],
                                                       input_layer_shape[2] // concat_layer_shape[2]))(concat_layer)

        concat_layer_upsampling = self.make_repeat_elements(concat_layer_upsampling, input_layer_shape[3])

        y = multiply([concat_layer_upsampling, input_layer])

        # Final layer
        conv_result = Conv2D(input_layer_shape[3], (1, 1), padding="same")(y)
        conv_result_batchNorm = BatchNormalization()(conv_result)

        return conv_result_batchNorm

    def build_attention_unit(self, dropout=0, BatchNorm=True):
        input_layer = Input(self.input_shape, dtype=tf.float32)

        ############ Add downsampling layer ############
        # Block 1, 128
        encoder_128 = self.make_conv_block(input_layer, self.filter_size, self.num_filters, dropout=dropout, BatchNorm=BatchNorm)
        if self.max_pooling:
            encoder_pool_64 = MaxPooling2D(pool_size=(2, 2))(encoder_128)
        # Block 2, 64 layer
        encoder_64 = self.make_conv_block(encoder_pool_64, self.filter_size, 2 * self.num_filters, dropout=dropout, BatchNorm=BatchNorm)
        if self.max_pooling:
            encoder_pool_32 = MaxPooling2D(pool_size=(2, 2))(encoder_64)
        # Block 3, 32 layer
        encoder_32 = self.make_conv_block(encoder_pool_32, self.filter_size, 4 * self.num_filters, dropout=dropout, BatchNorm=BatchNorm)
        if self.max_pooling:
            encoder_pool_16 = MaxPooling2D(pool_size=(2, 2))(encoder_32)
        # Block 4, 8 layer
        encoder_16 = self.make_conv_block(encoder_pool_16, self.filter_size, 8 * self.num_filters, dropout=dropout, BatchNorm=BatchNorm)
        if self.max_pooling:
            encoder_pool_8 = MaxPooling2D(pool_size=(2, 2))(encoder_16)
        # Block 5, just convolutional block
        encoder_8 = self.make_conv_block(encoder_pool_8, self.filter_size, 16 * self.num_filters, dropout=dropout, BatchNorm=BatchNorm)

        ############ Upsampling layers #############
        # Block 6, attention gated concatenation + upsampling + double residual convolution
        gate_16 = self._gate_signal(encoder_8, 8 * self.num_filters, BatchNorm=BatchNorm)
        attention_block_16 = self.make_attention_block(encoder_16, gate_16, 8 * self.num_filters)
        decoder_16 = UpSampling2D(size=(self.upsampling_filter, self.upsampling_filter), data_format="channels_last")(
            encoder_8)
        decoder_16 = concatenate([decoder_16, attention_block_16], axis=3)
        decoder_conv_16 = self.make_conv_block(decoder_16, self.filter_size, 8 * self.num_filters, dropout=dropout, BatchNorm=BatchNorm)
        # Block 7
        gate_32 = self._gate_signal(decoder_conv_16, 4 * self.num_filters, BatchNorm=BatchNorm)
        attention_block_32 = self.make_attention_block(encoder_32, gate_32, 4 * self.num_filters)
        decoder_32 = UpSampling2D(size=(self.upsampling_filter, self.upsampling_filter), data_format="channels_last")(
            decoder_conv_16)
        decoder_32 = concatenate([decoder_32, attention_block_32], axis=3)
        decoder_conv_32 = self.make_conv_block(decoder_32, self.filter_size, 4 * self.num_filters, dropout=dropout, BatchNorm=BatchNorm)
        # Block 8
        gate_64 = self._gate_signal(decoder_conv_32, 2 * self.num_filters, BatchNorm=BatchNorm)
        attention_block_64 = self.make_attention_block(encoder_64, gate_64, 2 * self.num_filters)
        decoder_64 = UpSampling2D(size=(self.upsampling_filter, self.upsampling_filter), data_format="channels_last")(
            decoder_conv_32)
        decoder_64 = concatenate([decoder_64, attention_block_64], axis=3)
        decoder_conv_64 = self.make_conv_block(decoder_64, self.filter_size, 2 * self.num_filters, dropout=dropout, BatchNorm=BatchNorm)
        # Block 9
        gate_128 = self._gate_signal(decoder_conv_64, self.num_filters, BatchNorm=BatchNorm)
        attention_block_128 = self.make_attention_block(encoder_128, gate_128, self.num_filters)
        decoder_128 = UpSampling2D(size=(self.upsampling_filter, self.upsampling_filter), data_format="channels_last")(decoder_conv_64)
        decoder_128 = concatenate([decoder_128, attention_block_128], axis=3)
        decoder_conv_128 = self.make_conv_block(decoder_128, self.filter_size, self.num_filters, dropout=dropout, BatchNorm=BatchNorm)

        # Final convolutional layers (1 * 1)
        final_conv_lr = Conv2D(self.num_classes, kernel_size=1)(decoder_conv_128)
        final_conv_lr = BatchNormalization(axis=3)(final_conv_lr)
        # If a binary classification, we need to set "sigmoid" while for multichannel we should change to softmax
        final_conv_lr = Activation("sigmoid")(final_conv_lr)

        # Set the model
        model = Model(input_layer, final_conv_lr, name="Attention_UNet")
        print(model.summary())

        return model