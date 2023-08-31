# TFG_Alex

## Introduction



In this repository you'll find two main files, mainprova.cpp for testing TFLite functions and mainNoOpencv.cpp for inference, for three pre-trained models with [LSVRC2012](https://image-net.org/challenges/LSVRC/2012/) database (only mobilenet_v1 and v2 working ok).

"ujpeg" files are provided in order to avoid using OpenCV or Pillow to read images, but you'll need to make TFL static library for your own architecture, ending up with libtensorflow-lite.a library and C API files (such as common and c_api) in /tensorflow/lite/c/ folder. Instructions [here](https://docs.google.com/document/d/1YPtUKOZrmr-ISzwUWAtvIcHHQDuf9tASxwWozw0Kcs0/edit?usp=drive_link). 

## Getting images

To make it easy, you can download pre-processed images with 224x224 pixels [here](https://www.kaggle.com/datasets/abhinavnayak/catsvdogs-transformed) or resize your own images to fit into your model. Just download your images and make sure they are in ".jpeg" format, if you need to chage their extension (for example from .jpg to .jpeg) open a terminal and move to /images folder:

```
for f in *jpg; do
> mv -- "$f" "${f%.jpg}.jpeg"
> done
```

Save your images in /images folder, you doesn't need to rename them, mainNoOpencv will read any ".jpeg" file contained in.

Modify NSAMPLES value (line 44 from mainNoOpencv.cpp) accordingly to the number of files in /images.


## Verifiying paths

Once you cloned our repository, you'll need to change labels, model, and image's path (lines 46 to 48 from mainNoOpencv.cpp) to your own. Do the same with your included files from TensorFlow Lite C API and "ujpeg" files (lines 11 to 14 from mainNoOpencv.cpp).

```
#include </home/pi/tensorflow-2.4.0/tensorflow/lite/c/common.h>
#include </home/pi/tensorflow-2.4.0/tensorflow/lite/c/c_api.h>
#include "/home/pi/Mobilenet_v1/c_api_TF_files/ujpeg.h"
#include "/home/pi/Mobilenet_v1/c_api_TF_files/ujpeg.c"

[...]

const char *LABELS_FILE = "/home/pi/Mobilenet_v1/labels/labels_mobilenet.txt"; 
const char *MODEL_FILE = "/home/pi/Mobilenet_v1/models/mobilenet_v1_1.0_224.tflite"; 
char directori[] = "/home/pi/Mobilenet_v1/images"; //CANVIAR pel path corresponent 
```

## Compiling 

If you don't have some package installed for the command provided below, install it with apt-get.
You need to provide your own TFL static library when compiling so make sure its path is ok.

Some architectures may need different parameters, for example:

```
// linux_aarch64 and armv6
 g++ mainNoOpencv.cpp -lstdc++ -Wl,--no-as-needed -ldl -lpthread -fpermissive -Wregister -Wreturn-type /home/pi/tensorflow-2.4.0/tensorflow/lite/tools/make/gen/linux_aarch64/lib/libtensorflow-lite.a -o mobilenet1NoOpencv_exec


 // linux_riscv64
 g++ mainNoOpencv.cpp -lstdc++ -Wl,--no-as-needed -ldl -latomic -fpermissive -Wregister -Wreturn-type /home/pi/tensorflow-2.4.0/tensorflow/lite/tools/make/gen/linux_riscv64/lib/libtensorflow-lite.a -o mobilenet1NoOpencv_exec
```
