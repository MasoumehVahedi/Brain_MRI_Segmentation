from tensorflow.keras import backend as K


# loss function and metrics
def dice_coef(y_pred, Y):
    y_flatten = K.flatten(Y)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_flatten * y_pred_flatten)
    dice = (0.2 * intersection + 1.0) / (K.sum(y_flatten) + K.sum(y_pred_flatten) + 1.0)
    return dice

def jacard_coef(y_pred, Y):
    y_flatten = K.flatten(Y)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_flatten * y_pred_flatten)
    jacard = (intersection + 1.0) / (K.sum(y_flatten) + K.sum(y_pred_flatten) - intersection + 1.0)
    return jacard

def jacard_coef_loss(y_pred, Y):
    return -jacard_coef(y_pred, Y)


def dice_coef_loss(y_pred, Y):
    return -dice_coef(y_pred, Y)



