# Copyright (C) 2020 - Present, OPPO Mobile Comm Corp., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import tensorflow as tf
from keras import backend as k

dictionary_of_custom_variables = {
    # Need to add custom variables in form of key,value pair
     'scale_x': 2
}


def mix_loss(y_actual, y_pred):
    mloss = 0.84 * (1 - tf.reduce_mean(tf.image.ssim_multiscale(y_pred, y_actual, 1.0))) \
            + 0.16 * (tf.reduce_mean(tf.math.abs(y_pred - y_actual)))
    return mloss


def _hard_swish(x):
    return x * k.relu(x + 3.0, max_value=6.0) / 6.0


def _relu6(x):
    return k.relu(x, max_value=6.0)


def pixel_mse_loss(x, y):
    return tf.reduce_mean((x - y) ** 2)