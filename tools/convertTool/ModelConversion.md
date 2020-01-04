
![ShaderNN logo](../../docs/images/logo.png)

  

##  Model Conversion (Coming soon!):

- Conversion from TensorFlow: The .h5 model format is supported in this release.
- Conversion from PyTorch: Please export the model into ONNX model format (opset 11), simplify it and then convert it with convertTool.
- Conversion from ONNX: Opset version 11 is supported
  

#  ConvertTool

ConvertTool is a Python tool for converting pre-trained TensorFlow/ONNX models to JSON format.


##  Usage 

Use the convertTool.py file from tools/convertTool with -filepath (-f) option for converting from .h5/.onnx file formats to JSON format used by ShaderNN.

```bash
python3 convertTool.py -f sample.h5
python3 convertTool.py -f sample.onnx
```

Use the command with -decoupleWeights (-d) option for generating separate files for layers and decoupled weights.

```bash
python3 convertTool.py -f sample.h5 -d
python3 convertTool.py -f sample.onnx -d
```
