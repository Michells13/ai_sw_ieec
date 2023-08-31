# TensorFlow vs TensorFlow Lite Model Specifications

This readme provides a comparison of model specifications between TensorFlow (tf) and TensorFlow Lite (tflite) versions for different architecture models. This information outlines the input and output specifications for both TensorFlow and TensorFlow Lite versions of the mentioned model architectures. It helps in understanding the differences and optimizations made in TensorFlow Lite models.

## Sizes

| Model Name   | Original Size (MB) | Optimized Size (MB)  |
|--------------|--------------------|----------------------|
| InceptionV3  | 86.05              | 21.7                 |
| MobileNet    | 16.8               |  4.4                 |
| ResNet50     | 92.6               | 23.6                 |
| SqueezeNet   | ----               | ----                 |
| VGG          | ----               | ----                 |
| U-Net        | 52.6               | 6.62                 |


## InceptionV3

| Parameter            | TensorFlow (tf)          | TensorFlow Lite (tflite) |
|--------------------- |------------------------- |-------------------------- |
| Input name           | input_2                  | serving_default_input_2:0 |
| Input shape          | (None, 299, 299, 3)      | (1, 299, 299, 3)          |
| Input data type      | float32                  | float32                   |
| Output name          | predictions/Softmax:0    | StatefulPartitionedCall:0 |
| Output shape         | (None, 4)                | (1, 4)                    |
| Output data type     | float32                  | float32                   |


## MobileNet

| Parameter            | TensorFlow (tf)          | TensorFlow Lite (tflite) |
|--------------------- |------------------------- |-------------------------- |
| Input name           | input_50                 | serving_default_input_50:0|
| Input shape          | (None, 224, 224, 3)      | (1, 224, 224, 3)          |
| Input data type      | float32                  | float32                   |
| Output name          | predictions/Softmax:0    | StatefulPartitionedCall:0 |
| Output shape         | (None, 4)                | (1, 4)                    |
| Output data type     | float32                  | float32                   |


## ResNet50

| Parameter            | TensorFlow (tf)          | TensorFlow Lite (tflite)  |
|--------------------- |------------------------- |-------------------------- |
| Input name           | input_14                 | serving_default_input_14:0|
| Input shape          | (None, 224, 224, 3)      | (1, 224, 224, 3)          |
| Input data type      | float32                  | float32                   |
| Output name          | predictions/Softmax:0    | StatefulPartitionedCall:0 |
| Output shape         | (None, 4)                | (1, 4)                    |
| Output data type     | float32                  | float32                   |


## Unet

| Parameter            | TensorFlow (tf)          | TensorFlow Lite (tflite) |
|--------------------- |------------------------- |-------------------------- |
| Input name           | input_3                  | serving_default_input_3:0 |
| Input shape          | (None, 256, 256, 3)      | (1, 256, 256, 3)          |
| Input data type      | float32                  | float32                   |
| Output name          | conv2d_80/Sigmoid:0      | StatefulPartitionedCall:0 |
| Output shape         | (None, 256, 256, 1)      | (1, 256, 256, 1)          |
| Output data type     | float32                  | float32                   |



## Model Download

You can download the trained and optimized models using the following Google Drive links:

### Classification
- [InceptionV3 tf](https://drive.google.com/uc?id=1_cHQmAq6e5qKFZ1Hx3uXK4R0FuBi-BSD&export=download)
- [InceptionV3 tflite](https://drive.google.com/uc?id=1qY6pLCKIohsYRx3ojvbwsB0DGgg7wLtS&export=download)
- [MobileNet tf](https://drive.google.com/uc?id=1GffvQXmlWF7B81yoeJ_-RphEVkQelm4K&export=download)
- [MobileNet tflite](https://drive.google.com/uc?id=1V5HWiGofsd2NZ7R3887s0Ve0hJZYWYW9&export=download)
- [ResNet50 tf](https://drive.google.com/uc?id=1HZn7Fu6yeRpiWQkuYWSKp6D5D21ykq5e&export=download)
- [ResNet50 tflite](https://drive.google.com/uc?id=1Wkk2-cBZdxEb_KlPny6M-Ra7CNQUhMmB&export=download)
- [SqueezeNet tf/pending to update](link_to_squeezenet_model)
- [SqueezeNet tflite/pending to update](link_to_squeezenet_model)
- [VGG16 tf/pending to update](link_to_vgg_model)
- [VGG16 tflite/pending to update](link_to_vgg_model)

### Segmentation
- [U-Net tf](https://drive.google.com/uc?id=1URpDD_kQrFuyNzzC6SzbQciT--HwzzGm&export=download)
- [U-Net tflite](ttps://drive.google.com/uc?id=16enrvngdk_NVKXjKEMI-fJh_tcJK1Hut&export=download)

To download the models using the command line, you can use the `wget` utility:

```bash
wget   'https://drive.google.com/uc?id=1_cHQmAq6e5qKFZ1Hx3uXK4R0FuBi-BSD&export=download'  -O InceptionV3.h5
wget   'https://drive.google.com/uc?id=1qY6pLCKIohsYRx3ojvbwsB0DGgg7wLtS&export=download'  -O Optimized_InceptionV3.tflite
wget   'https://drive.google.com/uc?id=1GffvQXmlWF7B81yoeJ_-RphEVkQelm4K&export=download'  -O MobileNet.h5
wget   'https://drive.google.com/uc?id=1V5HWiGofsd2NZ7R3887s0Ve0hJZYWYW9&export=download'  -O Optimized_MobileNet.tflite
wget   'https://drive.google.com/uc?id=1HZn7Fu6yeRpiWQkuYWSKp6D5D21ykq5e&export=download'  -O Resnet50.h5
wget   'https://drive.google.com/uc?id=1Wkk2-cBZdxEb_KlPny6M-Ra7CNQUhMmB&export=download'  -O Optimized_Resnet50.tflite
wget   'https://drive.google.com/uc?id=1URpDD_kQrFuyNzzC6SzbQciT--HwzzGm&export=download'  -O unet.h5
wget   'https://drive.google.com/uc?id=16enrvngdk_NVKXjKEMI-fJh_tcJK1Hut&export=download'  -O Optimized_unet.tflite

No optimized tflite models:

wget 'https://drive.google.com/uc?id=1PEMG_QqW5f0ouhbPTwdh_e_7RhYUZ_tw&export=download'  -O converted_NoO_Inceptionv3.tflite
wget 'https://drive.google.com/uc?id=1lfqdsJFYp7mW1q_6iHb8eT8cpvM-CJkS&export=download'  -O converted_NoO_MobileNet.tflite
wget 'https://drive.google.com/uc?id=1Dzt6av14YlXCo6VUySrp3zJ7iwVAzC4G&export=download'  -O converted_NoO_resnet50.tflite
wget 'https://drive.google.com/uc?id=1Z8cta8FDK7aejD_w6Cdd2AXERDBRCwG7&export=download'  -O converted_NoO_unet.tflite
```

