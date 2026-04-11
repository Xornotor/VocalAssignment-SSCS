import pickle
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, Reduction
from tensorflow.keras.layers import Input, Resizing, Conv2D, BatchNormalization, Multiply, Lambda, Concatenate
import tensorflow.keras.backend as K

EPOCHS = 10
TRAINING_DTYPE = tf.float16
SPLIT_SIZE = 256
BATCH_SIZE = 24
LEARNING_RATE = 5e-3
RESIZING_FILTER = 'bilinear'

############################################################

def mask_voas_cnn_model(l_rate = LEARNING_RATE):
    x_in = Input(shape=(360, SPLIT_SIZE, 1))

    x = Resizing(90, int(SPLIT_SIZE/2), RESIZING_FILTER,
                 name="downscale")(x_in)
    
    x = BatchNormalization()(x)

    x = Conv2D(filters=32, kernel_size=(3, 3), padding="same",
        activation="relu", name="conv1")(x)

    x = BatchNormalization()(x)

    x = Conv2D(filters=32, kernel_size=(3, 3), padding="same",
        activation="relu", name="conv2")(x)

    x = BatchNormalization()(x)

    x = Conv2D(filters=16, kernel_size=(70, 3), padding="same",
        activation="relu", name="conv_harm_1")(x)

    x = BatchNormalization()(x)

    x = Conv2D(filters=16, kernel_size=(70, 3), padding="same",
        activation="relu", name="conv_harm_2")(x)
    
    x = BatchNormalization()(x)

    ## "masking" original input with trained data

    x = Resizing(360, SPLIT_SIZE, RESIZING_FILTER,
                 name="upscale")(x)

    x = Multiply(name="multiply_mask")([x, x_in])

    ## start four branches now

    ## branch 1
    x1a = Conv2D(filters=16, kernel_size=(3, 3), padding="same",
        activation="relu", name="conv1a")(x)

    x1a = BatchNormalization()(x1a)

    x1b = Conv2D(filters=16, kernel_size=(3, 3), padding="same",
        activation="relu", name="conv1b")(x1a)

    ## branch 2
    x2a = Conv2D(filters=16, kernel_size=(3, 3), padding="same",
        activation="relu", name="conv2a")(x)

    x2a = BatchNormalization()(x2a)

    x2b = Conv2D(filters=16, kernel_size=(3, 3), padding="same",
        activation="relu", name="conv2b")(x2a)

    ## branch 3

    x3a = Conv2D(filters=16, kernel_size=(3, 3), padding="same",
        activation="relu", name="conv3a")(x)

    x3a = BatchNormalization()(x3a)

    x3b = Conv2D(filters=16, kernel_size=(3, 3), padding="same",
        activation="relu", name="conv3b")(x3a)

    x4a = Conv2D(filters=16, kernel_size=(3, 3), padding="same",
        activation="relu", name="conv4a")(x)

    x4a = BatchNormalization()(x4a)

    x4b = Conv2D(filters=16, kernel_size=(3, 3), padding="same",
        activation="relu", name="conv4b"
    )(x4a)


    y1 = Conv2D(filters=1, kernel_size=1, name='conv_soprano',
                padding='same', activation='sigmoid')(x1b)
    y1 = tf.squeeze(y1, axis=-1, name='sop')

    y2 = Conv2D(filters=1, kernel_size=1, name='conv_alto',
                padding='same', activation='sigmoid')(x2b)
    y2 = tf.squeeze(y2, axis=-1, name='alt')

    y3 = Conv2D(filters=1, kernel_size=1, name='conv_tenor',
                padding='same', activation='sigmoid')(x3b)
    y3 = tf.squeeze(y3, axis=-1, name='ten')

    y4 = Conv2D(filters=1, kernel_size=1, name='conv_bass',
                padding='same', activation='sigmoid')(x4b)
    y4 = tf.squeeze(y4, axis=-1, name='bas')

    out = [y1, y2, y3, y4]

    model = Model(inputs=x_in, outputs=out, name='MaskVoasCNN')

    model.compile(optimizer=Adam(learning_rate=l_rate),
                 loss=BinaryCrossentropy(reduction=Reduction.SUM_OVER_BATCH_SIZE))
    
    model.load_weights('./Checkpoints/mask_voas.keras')

    return model

############################################################

def mask_voas_cnn_v2_model(l_rate = LEARNING_RATE):
    x_in = Input(shape=(360, SPLIT_SIZE, 1))

    x = Resizing(90, int(SPLIT_SIZE/2), RESIZING_FILTER,
                 name="downscale")(x_in)
    
    x = BatchNormalization()(x)

    x = Conv2D(filters=32, kernel_size=(3, 3), padding="same",
        activation="relu", name="conv1")(x)

    x = BatchNormalization()(x)

    x = Conv2D(filters=32, kernel_size=(3, 3), padding="same",
        activation="relu", name="conv2")(x)

    x = BatchNormalization()(x)

    x = Conv2D(filters=16, kernel_size=(48, 3), padding="same",
        activation="relu", name="conv_harm_1")(x)

    x = BatchNormalization()(x)

    x = Conv2D(filters=16, kernel_size=(48, 3), padding="same",
        activation="relu", name="conv_harm_2")(x)
    
    x = BatchNormalization()(x)

    x = Conv2D(filters=16, kernel_size=1, padding="same",
        activation="sigmoid", name="conv_sigmoid_before_mask")(x)

    ## "masking" original input with trained data

    x = Resizing(360, SPLIT_SIZE, RESIZING_FILTER,
                 name="upscale")(x)

    x = Multiply(name="multiply_mask")([x, x_in])

    x = BatchNormalization()(x)

    ## start four branches now

    ## branch 1
    x1a = Conv2D(filters=16, kernel_size=(3, 3), padding="same",
        activation="relu", name="conv1a")(x)

    x1a = BatchNormalization()(x1a)

    x1b = Conv2D(filters=16, kernel_size=(3, 3), padding="same",
        activation="relu", name="conv1b")(x1a)

    ## branch 2
    x2a = Conv2D(filters=16, kernel_size=(3, 3), padding="same",
        activation="relu", name="conv2a")(x)

    x2a = BatchNormalization()(x2a)

    x2b = Conv2D(filters=16, kernel_size=(3, 3), padding="same",
        activation="relu", name="conv2b")(x2a)

    ## branch 3

    x3a = Conv2D(filters=16, kernel_size=(3, 3), padding="same",
        activation="relu", name="conv3a")(x)

    x3a = BatchNormalization()(x3a)

    x3b = Conv2D(filters=16, kernel_size=(3, 3), padding="same",
        activation="relu", name="conv3b")(x3a)

    x4a = Conv2D(filters=16, kernel_size=(3, 3), padding="same",
        activation="relu", name="conv4a")(x)

    x4a = BatchNormalization()(x4a)

    x4b = Conv2D(filters=16, kernel_size=(3, 3), padding="same",
        activation="relu", name="conv4b"
    )(x4a)


    y1 = Conv2D(filters=1, kernel_size=1, name='conv_soprano',
                padding='same', activation='sigmoid')(x1b)
    y1 = tf.squeeze(y1, axis=-1, name='sop')

    y2 = Conv2D(filters=1, kernel_size=1, name='conv_alto',
                padding='same', activation='sigmoid')(x2b)
    y2 = tf.squeeze(y2, axis=-1, name='alt')

    y3 = Conv2D(filters=1, kernel_size=1, name='conv_tenor',
                padding='same', activation='sigmoid')(x3b)
    y3 = tf.squeeze(y3, axis=-1, name='ten')

    y4 = Conv2D(filters=1, kernel_size=1, name='conv_bass',
                padding='same', activation='sigmoid')(x4b)
    y4 = tf.squeeze(y4, axis=-1, name='bas')

    out = [y1, y2, y3, y4]

    model = Model(inputs=x_in, outputs=out, name='MaskVoasCNNv2')

    model.compile(optimizer=Adam(learning_rate=l_rate),
                 loss=BinaryCrossentropy(reduction=Reduction.SUM_OVER_BATCH_SIZE))

    model.load_weights('./Checkpoints/mask_voas_v2.keras')

    return model

############################################################

def __base_model(input, let):

    b1 = BatchNormalization()(input)

    # conv1
    y1 = Conv2D(16, (5, 5), padding='same', activation='relu', name='conv1{}'.format(let))(b1)
    y1a = BatchNormalization()(y1)

    # conv2
    y2 = Conv2D(32, (5, 5), padding='same', activation='relu', name='conv2{}'.format(let))(y1a)
    y2a = BatchNormalization()(y2)

    # conv3
    y3 = Conv2D(32, (5, 5), padding='same', activation='relu', name='conv3{}'.format(let))(y2a)
    y3a = BatchNormalization()(y3)

    # conv4 layer
    y4 = Conv2D(32, (5, 5), padding='same', activation='relu', name='conv4{}'.format(let))(y3a)
    y4a = BatchNormalization()(y4)

    # conv5 layer, harm1
    y5 = Conv2D(32, (70, 3), padding='same', activation='relu', name='harm1{}'.format(let))(y4a)
    y5a = BatchNormalization()(y5)

    # conv6 layer, harm2
    y6 = Conv2D(32, (70, 3), padding='same', activation='relu', name='harm2{}'.format(let))(y5a)
    y6a = BatchNormalization()(y6)

    return y6a, input


def late_deep_cnn_model():
    '''Late/Deep
    '''

    input_shape_1 = (None, None, 5) # HCQT input shape
    input_shape_2 = (None, None, 5)  # phase differentials input shape

    inputs1 = Input(shape=input_shape_1)
    inputs2 = Input(shape=input_shape_2)

    y6a, _ = __base_model(inputs1, 'a')
    y6b, _ = __base_model(inputs2, 'b')

    # concatenate features
    y6c = Concatenate()([y6a, y6b])

    # conv7 layer
    y7 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv7')(y6c)
    y7a = BatchNormalization()(y7)

    # conv8 layer
    y8 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv8')(y7a)
    y8a = BatchNormalization()(y8)

    y9 = Conv2D(8, (360, 1), padding='same', activation='relu', name='distribution')(y8a)
    y9a = BatchNormalization()(y9)

    y10 = Conv2D(1, (1, 1), padding='same', activation='sigmoid', name='squishy')(y9a)
    predictions = Lambda(lambda x: K.squeeze(x, axis=3))(y10)

    model = Model(inputs=[inputs1, inputs2], outputs=predictions)

    model.compile(
        loss=__bkld, metrics=['mse', __soft_binary_accuracy],
        optimizer='adam'
    )

    model.load_weights('./Checkpoints/exp3multif0.h5')

    return model

############################################################

def __bkld(y_true, y_pred):
    """Brian's KL Divergence implementation
    """
    y_true = K.clip(y_true, K.epsilon(), 1.0 - K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    return K.mean(K.mean(
        -1.0*y_true* K.log(y_pred) - (1.0 - y_true) * K.log(1.0 - y_pred),
        axis=-1), axis=-1)

############################################################

def __soft_binary_accuracy(y_true, y_pred):
    """Binary accuracy that works when inputs are probabilities
    """
    return K.mean(K.mean(
        K.equal(K.round(y_true), K.round(y_pred)), axis=-1), axis=-1)

############################################################