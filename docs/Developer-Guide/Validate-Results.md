
#  How to validate results with NCNN

  

NCNN is a high-performance neural network framework developed by Tencent.

It can be found here: `https://github.com/Tencent/ncnn`

We use NCNN results as ground truth data.

The following steps explain the process of validation.

  

##  Step 1:

- Build the core for linux target:

```

./core/build.sh clean && ./core/build.sh linux

```

  

##  Step 2:

- The test source files are located in `demo/test/unittest`. The ShaderNN execution of models are located in the file [inferenceProcessorTest.cpp](../../demo/test/unittest/inferenceProcessorTest.cpp). The corresponding ncnn executions can be found in same dir with nomenclature format of `modelnameTest.cpp`. For example: [resnet18Test.cpp](../../demo/test/unittest/resnet18Test.cpp) represents source code for ncnn's execution of ResNet18.
- Install the NCNN library by running:

```

cd ./demo; ./install-deps.sh linux

```

- Build the tests by running:

```

./demo/build-tests.sh linux

```

- After successfully building the tests, the binaries are located at following path: `shaderneuralnetworkframework/demo/build-test/test/unittest`

  

##  Step 3:

- The tests can be run by executing the built binaries.

- First, the ShaderNN's model is executed.

```

./inferenceProcessorTest --use_compute 0 1

```

Here, first argument '0' represents model index. Case 0 is Resnet18. Second argument '1' represents dumping of model outputs. If set to '1', output is dumped, else it is not.

- The dumps can be located in `shaderneuralnetworkframework/inferenceCoreDump`

- The comparison with NCNN output is run.

```

// Run ./test_your_model

./resnet18Test --use_compute

```

This executes ncnn results and compares each layer's output with ShaderNN's result.

- If the results of a layer match, it shows a prompt on terminal:

`output res: 0`. If it doesn't match the terminal output reads: `output res: -1`.

- Example is as follows:

```

MRT_MODE SET TO: NULL
SHADER MODE: COMPUTE
getNCNNLayer:295, to get: input_1_blob
getSNNLayer:383, Dim: 32, 32, 3
CVMat2NCNNMat:128-------CV Mat-32--32--3-------
---------------------------------Conv_layer_1 layer input res: 0
getNCNNLayer:295, to get: activation_blob
getSNNLayer:404, Dim: 16, 16, 64, actual channels: 64
CVMat2NCNNMat:128-------CV Mat-16--16--64-------
-----------------------------Conv_layer_1 output res: 0
getNCNNLayer:295, to get: max_pooling2d_blob
getSNNLayer:404, Dim: 8, 8, 64, actual channels: 64
CVMat2NCNNMat:128-------CV Mat-8--8--64-------
-----------------------------Pooling_layer_1 output res: 0
getNCNNLayer:295, to get: conv2d_1_blob
getSNNLayer:404, Dim: 8, 8, 64, actual channels: 64
CVMat2NCNNMat:128-------CV Mat-8--8--64-------
-----------------------------1st Block Conv_Layer_2 output res: 0
getNCNNLayer:295, to get: batch_normalization_1_blob
getSNNLayer:404, Dim: 8, 8, 64, actual channels: 64
CVMat2NCNNMat:128-------CV Mat-8--8--64-------
-----------------------------1st Block batch_norm_layer_1 output res: 0
getNCNNLayer:295, to get: activation_1_blob
getSNNLayer:404, Dim: 8, 8, 64, actual channels: 64
CVMat2NCNNMat:128-------CV Mat-8--8--64-------
-----------------------------1st Block Add_layer_1 output res: 0
getNCNNLayer:295, to get: activation_2_blob
getSNNLayer:404, Dim: 8, 8, 64, actual channels: 64
CVMat2NCNNMat:128-------CV Mat-8--8--64-------
-----------------------------1st Conv2D_layer_2 output res: 0
...
getNCNNLayer:295, to get: activation_12_blob
getSNNLayer:404, Dim: 1, 1, 512, actual channels: 512
CVMat2NCNNMat:128-------CV Mat-1--1--512-------
-----------------------------activation 12 output res: 0
getNCNNLayer:295, to get: activation_13_blob
getSNNLayer:404, Dim: 1, 1, 512, actual channels: 512
CVMat2NCNNMat:128-------CV Mat-1--1--512-------
-----------------------------activation 13 output res: 0
getNCNNLayer:295, to get: activation_14_blob
getSNNLayer:404, Dim: 1, 1, 512, actual channels: 512
CVMat2NCNNMat:128-------CV Mat-1--1--512-------
-----------------------------activation 14 output res: 0
getNCNNLayer:295, to get: activation_15_blob
getSNNLayer:404, Dim: 1, 1, 512, actual channels: 512
CVMat2NCNNMat:128-------CV Mat-1--1--512-------
-----------------------------activation 15 output res: 0
getNCNNLayer:295, to get: average_pooling2d_blob
getSNNLayer:404, Dim: 1, 1, 512, actual channels: 512
CVMat2NCNNMat:128-------CV Mat-1--1--512-------
-----------------------------Final Average_Layer_1 output res: 0
getNCNNLayer:295, to get: dense_Softmax_blob
End of file stof
------NCNN Mat--10--1--1-------
0.011892, 0.001505, 0.052707, 0.365585, 0.187166, 0.079344, 0.260408, 0.026428, 0.012621, 0.002343, 

----------0--------------
------NCNN Mat--10--1--1-------
0.012049, 0.001527, 0.052777, 0.364945, 0.188186, 0.079162, 0.259715, 0.026491, 0.012773, 0.002375, 

----------0--------------
-----------------------------Output Blob output res: 0

```

Here, the output of one of the activation layer of ResNet 18 is shown. The last line reads output res:0 which indicates that the results of ShaderNN match with that of NCNN.

  

##  Step 4:

In order to implement a test for a model which is currently not supported, an implementation needs to be made in [inferenceProcessorTest.cpp](../../demo/test/unittest/inferenceProcessorTest.cpp) and a corresponding `your_modelTest.cpp` would be needed. Also, update the `CMakeLists.txt` in `demo/test/unittest` folder. Refer to existing tests for detailed implementation details.
