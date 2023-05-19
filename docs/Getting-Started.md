
![ShaderNN logo](images/logo.png)

  
##  Getting Started:
 
##  Build ShaderNN Core

Select OpenGL or Vulkan backend first:
```
% // Select OpenGL backend
% cd core
% ./config.sh gl
% // Select Vulkan backend
% cd core
% ./config.sh vulkan
```

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
% // flash with model files
% ./flash.sh models
% // flash without model files
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

inferenceprocessorTest takes multiple parameters, you can also specify one or more of those parameters used to run: 

- --use_vulkan : Use Vulkan backend. 
- --use_compute : Use compute shader (OpenGL only). 
- --use_1ch_mrt:  Use single plane MRT (OpenGL only).
- --use_2ch_mrt: Use double plane MRT (OpenGL only).
- --use_constants: Store weight as constants (OpenGL only).
- --use_use_half: Use half-precision floating point values (fp16).
- --use_finetuned: Use fine-tuned models.
- --dump_outputs: Dump outputs.
- --inner_loops: Number of inner loops (after model loading).
- --outer_loops: Number of outer loops (before model loading).
- model: use one of {resnet18 | yolov3tiny | unet | mobilenetv2 | spatialdenoise | aidenoise | styletransfer | espcn2x}

You can refer to unit test for resnet18 test_resnet18.sh as example. 

