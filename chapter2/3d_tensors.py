import os
import tensorflow as tf
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

tensor_3d = np.array([[[1, 2],[3, 4]], [[5, 6], [7, 8]]])
print tensor_3d
print tensor_3d.shape

