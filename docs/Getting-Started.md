
![ShaderNN logo](images/logo.png)

  
##  Getting Started:
 
##  Build ShaderNN Core


Core offers two broad build targets at the moment: Android, Linux

For default Android (64 bit, Debug) option:

```
export ANDROID_SDK_ROOT=/path/to/sdk/
% cd core
% ./build.sh android
```

**Note:** To switch targets, call the clean option prior to build.

For linux:

```
% cd core
% ./build.sh linux
```
  
To clean:

```
% cd core
% ./build.sh clean
```

##  Build and Run the demo-app

**Note:** Build the Core for Android first.

Install dependence at first build:
 
```
% cd demo/
% ./install-deps.sh android
```

Build:
```
% cd demo/android/
% ./build.sh
```

Flash:
```
% cd demo/android/
% // With Models
% ./flash.sh models
% // Without Models
% ./flash.sh
```

To run unit-tests:

```
% cd demo/android/
% ./run-unit-test.sh resnet18
```

  
  

##  Build and Run the tests

**Note:** Build the Core for Linux first.

  

Linux Tests:

Install dependence at first build:
 
```
% cd demo/
% ./install-deps.sh linux
``` 

Build:
```
% cd demo
% ./build-tests.sh linux
```

Run:

To test the inference on a model, we use NCNN as the basline. First, you need to run inferenceprocessorTest, which will run the model in ShaderNN, and dump the intermediate results into the local folder. Then you can run the modelTest to compare the ShaderNN intermediate results with NCNN results layer by layer. 

inferenceprocessorTest takes two more parameters, the second from last parameter is the model index, and the last parameter specifies if dump the output for test (0: no dump, 1: dump). You can also specify which shader option used to run: 

- --use_compute : using computer shader first. 
- --use_1ch_mrt : use fragment shader and render 1 channel (RGBA) per pass. 
- --use_2ch_mrt:  use fragment shader and render 2 channels (RGBA) per pass.
- --use_half: ShaderNN will use FP32 by default, and you can use this option to specify FP16

Model index:  
- 0: Resnet18
- 1: Yolov3Tiny
- 2: UNet 
- 3: MobilenetV2
- 4: SpatialDenoiser 
- 5: AIDenoiser
- 6: StyleTransfer

Below shows inference Resnet18 network with output dumped
```
% cd build-test/test/unittest/
% ./inferenceProcessorTest --use_compute 0 1
% // After above inference complete with output dumped, then compare the layer and end to end results with NCNN, below is to compare Resnet18
% ./resnet18Test --use_compute
```
