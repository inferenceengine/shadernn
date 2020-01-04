# Custom Layer/ Operator

Custom layers/ operators need to be supported in shaderNN core and also need to be handled in the convertTool (h5/ onnx -> .json).

# How to implement a custom layer in ShaderNN core

In order to implement a custom layer in ShaderNN, there is a need to add support in the core.

The layers implemented in ShaderNN, inherit the parent class: `GenericModelLayer` defined in [dp.h](). In case the layer has a shader, the layer class can then also inherit `ShaderLayer` class, also defined in [dp.h]().

Once the layer's class is defined in [dp.h](), the implementation can be made in `core/src/ic2/your-custom-layer.cpp`. The required methods can be implemented as per the requirements of the layer. Refer to different layers implemented in `core/src/ic2/` to check the implementation examples.

# How to convert a model with custom layer

Some models make use of a custom layer or have some custom parameters involved. This section explains how the convertTool can be used to handle such models.

- Example of a model with custom parameters: 
  - ESPCN keras model uses a global variable 'scale_x' in the Python training script. It also makes use of a Lambda layer involving depth_to_space operator.
    ```
    scale_x = 2
    def SubpixelConv2D(name="subpixel"):
        def subpixel_shape(input_shape):
            ...
            ...
            return output_shape

        def subpixel(x):
            global scale_x
            return tf.compat.v1.depth_to_space(x, scale_x)

        return Lambda(subpixel, output_shape= subpixel_shape, name=name)

    class ESPCN:

        def __init__(self, scale):
            global scale_x
            scale_x = scale
            self.model = self.buildModel()

        def buildModel(self):
            global scale_x
            inputs = Input(shape=(None, None, 1), name= 'input')

            x = Conv2D(...
                       ...

            x = Conv2D(filters = scale_x**2*1, ... ,name='conv_3')(x)           

        x = SubpixelConv2D(name='subpixel')(x)
        
        model = Model(inputs=inputs, outputs=x)
        return model

    ```
    In order to load this model, a parameter scale_x needs to be provided to the Keras load_model api.
  - Some models have presence of some custom loss functions which need to be defined.
    ```
    def mix_loss(y_actual, y_pred):
        mloss = 0.84 * (1 - tf.reduce_mean(tf.image.ssim_multiscale(y_pred, y_actual, 1.0))) + 0.16 * (tf.reduce_mean(tf.math.abs(y_pred - y_actual)))
        return mloss
    ```

- In order to create json model using convertTool, we need to update [supportedCustomLayers.py]() file. The definition of custom variables can be added in a dictionary variable `dictionary_of_custom_variables`. The custom loss functions can be provided as part of python functions.
    ```
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
    ```

- Following this change in [supportedCustomLayers.py](), convertTool can now be used to generate json model file.
