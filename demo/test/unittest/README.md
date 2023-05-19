# ShaderNN unit tests

ShaderNN unit tests can be divided in 2 groups: ops test and model tests.  
Ops test usually test just one individual op.  
Model test test a particular model end to end and compare results layer by layer to the results, produced by NCNN engine, used as "ground truth".  
Binaries for all tests are produced under the directory: **{shadernn}/demo/build-test/test/unittest**  

## Ops tests

Ops tests are the following:  

| Test                   | Binary name            |
| ---------------------- | ---------------------- |
| Activation             | activationTest         |
| Batch normalization    | batchNormTest          |
| Binary op              | binaryOpTest           |
| Concatenation          | concatTest             |
| Convolution            | convolutionTest        |
| Depthwise convolution  | depthwiseConv2DTest    |
| Dense                  | denseTest              |
| Flatten                | flattenTest            |
| Image texture resize   | imageTextureResizeTest |
| Image texture general  | imageTextureTest       |
| Instance normalization | instanceNormTest       |
| Padding                | padTest                |
| Pooling                | poolingTest            |
| Upsampling             | upSampleTest           |

To run an op unit test just run the appropriate binary. Use _--help_ parameter to query the options that particular test accepts.  
All tests accepts the following options:  
_--use_vulkan_ - to run test against Vulkan platform. Otherwise it will run against OpenGL.  
_Note: If you have built ShaderNN against only one platform, you should run test against this platform. Otherwise the test will have no effect._  

_--print_mismatch_ - to print mismatch numbers if test fails.  

For convenience, you can run all ops tests through the script: **{shadernn}/demo/unittest.sh**  

## Model tests

Model tests run in 2 steps. In the 1st step run **inferenceProcessorTest** with the appropriate model as a parameter.  
**inferenceProcessorTest** will dump layer results in binary files.  
In the 1st step run the appropriate model result test. Model result test will run NCNN engine against the same model and compare its result with dumped ShaderNN results.  
The test for some models exist in 2 varuants: basic and fine-tuned (ft). To run the finetuned version of the model with **inferenceProcessorTest**, use the option: _--use_finetuned_.  

Model tests are the following:  

| Test            | Model parameter        | Model result test        |
| --------------- | ---------------------- | -------------------------|
| Mobilenet V2    | mobilenetv2            | mobilenetv2Test          |
| Mobilenet V2 ft | mobilenetv2            | mobilenetv2FinetunedTest |
| Resnet18        | resnet18               | resnet18Test             |
| Resnet18 ft     | resnet18               | resnet18FinetunedTest    |
| Spatial denoise | spatialdenoise         | N/A                      |
| Style transfer  | styletransfer          | styleTransferTest        |
| Unet            | unet                   | unetTest                 |
| Unet ft         | unet                   | unetFinetunedTest        |
| Yolo V3         | yolov3tiny             | yolov3TinyTest           |
| Yolo V3 ft      | yolov3tiny             | yolov3TinyFinetunedTest  |
| ESPCN           | espcn2x                | 1)                       |

1) ESPCN test is a special case. Use **{shadernn}/demo/test_espcn.sh** to test ESPCN.  

**./inferenceProcessorTest** accepts the following options:  
 ```
 -h,--help  Print this help message and exit  
 --use_vulkan         Use Vulkan  
 --use_compute        Use compute shader (OpenGL only)  
 --use_1ch_mrt        Use single plane MRT (OpenGL only)  
 --use_2ch_mrt        Use double plane MRT (OpenGL only)  
 --use_constants      Store weight as constants (OpenGL only)  
 --use_half           Use half-precision floating point values (fp16)  
 --use_finetuned      Use fine-tuned models  
 --dump_outputs       Dump outputs  
 --inner_loops UINT   Number of inner loops (after model loading)  
 --outer_loops UINT   Number of outer loops (before model loading)  
```

Always use -_-dump_outputs_ option when testing the model results.  
Don't use _--inner_loops_, _--outer_loops_ options when testing the model results.  
_--inner_loops_ option makes model run several times and used for benchmarking.  
_--outer_loops_ option used to test reattaching images for resnet18 model only.

Run **{model result test} --help** to find out the options for a particular model result test.  
All model results tests accept the following options:  
```
 -h,--help            Print this help message and exit  
 --use_vulkan         Use Vulkan  
 --use_1ch_mrt        Use single plane MRT (OpenGL only)  
 --use_2ch_mrt        Use double plane MRT (OpenGL only)  
 --use_half           Use half-precision floating point values (fp16)  
 --stop_on_mismatch   Stop on results mismatch  
 --print_mismatch     Print results mismatch  
 --print_last_layer   Print last layer  
```

The following parameters should match when you run **./inferenceProcessorTest** in a 1st step and **{model result test}** in the 2nd step:  
  --use_vulkan  
  --use_1ch_mrt  
  --use_2ch_mrt  
  --use_half  

Example to test resnet18 with Fragment Shader, 1 plane, FP16:
```
./inferenceProcessorTest --use_half --use_1ch_mrt --dump_outputs resnet18
./resnet18Test --use_1ch_mrt --use_half --stop_on_mismatch
```

_Note: If you have built ShaderNN against only one platform, you should run test against this platform. Otherwise the test will have no effect._  

For convenience, you can run all model tests through the following scripts:  
**{shadernn}/demo/test_espcn.sh**  
**{shadernn}/demo/test_mobilenetv2_ft.sh**  
**{shadernn}/demo/test_mobilenetv2.sh**  
**{shadernn}/demo/test_resnet18_ft.sh**  
**{shadernn}/demo/test_resnet18.sh**  
**{shadernn}/demo/test_styletransfer.sh**  
**{shadernn}/demo/test_unet_ft.sh**  
**{shadernn}/demo/test_unet.sh**  
**{shadernn}/demo/test_yolov3_ft.sh**  
**{shadernn}/demo/test_yolov3.sh**  
