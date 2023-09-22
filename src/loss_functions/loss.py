import torch

import torch.nn.functional as F

# def huber_loss(y_pred, y_true, delta=1.0):
#     error = y_pred - y_true
#     abs_error = torch.abs(error)
#     quadratic = torch.min(abs_error, delta)
#     print("quadratic: \n", quadratic)
#     linear = (abs_error - quadratic)
#     return 0.5 * quadratic ** 2 + delta * linear

def huber_loss(y_pred, y_true, delta=1.0):
    error = y_pred - y_true
    abs_error = torch.abs(error)
    quadratic = torch.where(abs_error <= delta, 0.5 * abs_error ** 2, delta * (abs_error - 0.5 * delta))
    return quadratic.mean()
