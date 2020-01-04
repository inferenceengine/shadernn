# How to execute a Model not supported currently

In order to execute a model having an input shape/ output shape/ layer/ operator which is/ are currently not supported by ShaderNN, there is a need to implement a way to execute the model.

Before executing model, there is a need to add support of any custom layer/ operator used in model, in ShaderNN. Please refer to the guide: [Custom-Layer.md]() for more details.

In order to execute the model on android, supporting different types on inputs/ outputs shape/ format/ pre-post process, the following methods are used: 1. Unit-test, 2. Model Processor.

## Option 1: Unit-test (Preferred)
This can be implemented in `shaderneuralnetworkframework/demo/android`. The Java side handles test declarations: [NativeTests.java]() and cpp implementation of the test is in [native-lib.cpp](). 

The implementation of a test in `native-lib.cpp` has following basic steps:

- Create a render context and pointer to assetmanager:
    ```
    g_assetManager = AAssetManager_fromJava(env, java_am);
    SNN_CHK(g_assetManager);
    auto rc = gl::RenderContext(gl::RenderContext::STANDALONE);
    rc.makeCurrent();
    ```
- Create and initialize InferenceProcessor:
    ```
    // Specify Model name
    auto modelFilename = "your-model.json";
    
    // Specify Input Shape
    std::vector<std::pair<std::string, std::vector<uint32_t>>> inputList;
    inputList.push_back({"input", vector<uint32_t>{1920, 1080, 1, 1}});

    // Inference processor
    auto ip = new snn::InferenceProcessor();
    // half-precision: true, dumpOutputs: false in this case.
    ip->initialize(modelFilename, inputList, &rc, true, false); 
    ```
- Define input and create input texture:
    ```
    // Define Input
    snn::FixedSizeArray<snn::ImageTexture> inputTexs;
    auto input = snn::ManagedRawImage::loadFromAsset("images/bright_night_view_street_1080x1920.jpg");

    // Create Input texture
    auto input32f = snn::toR32f(input, -1.0, 1.0);
    gl::TextureObject scaleTex;
    scaleTex.allocate2D(input32f.format(), input32f.width(), input32f.height(), 1, 1);
    scaleTex.setPixels(0, 0, 0, 1920, 1080, 0, input32f.data());
    scaleTex.detach();
    inputTexs.allocate(1);
    inputTexs[0].texture(0)->attach(scaleTex.target(), scaleTex.id());

    ip->preProcess(inputTexs);
    ```
- Create output texture:
    ```
    snn::FixedSizeArray<snn::ImageTexture> outputTexs;
    outputTexs.allocate(1);
    outputTexs[0].texture(0)->allocate2D(snn::ColorFormat::RGBA32F, 1920, 1080);
    ```

- Run the inference core:
    ```
    ip->process(outputTexs);
    ip->finalize();
    glFinish();
    ```

For detailed implementation, refer to unit-tests present in `native-lib.cpp`

## Option 2: Model-Processor

Model-Processor approach works with the demo android app. The implementation is carried out such that processor files are created in shaderNN core. For example, please refer to: [spatialDenoiser.cpp]() & [spatialDenoiser.h](). Once the snn-core is built, demo app needs implementation of UI to select model on the android app.