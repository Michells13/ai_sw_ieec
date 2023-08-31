#include <iostream>
#include <fstream>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <dirent.h>
#include <vector>
#include <time.h>

#include </home/pi/tensorflow-2.4.0/tensorflow/lite/c/common.h>
#include </home/pi/tensorflow-2.4.0/tensorflow/lite/c/c_api.h>
#include "/home/pi/ai_sw/Risc-V/c_api_TF_files/ujpeg.h"
#include "/home/pi/ai_sw/Risc-V/c_api_TF_files/ujpeg.c"


// Dispose of the model and interpreter objects.
int disposeTfLiteObjects(TfLiteModel* pModel, TfLiteInterpreter* pInterpreter)
{
    if(pModel != NULL)
    {
      TfLiteModelDelete(pModel);
    }

    if(pInterpreter)
    {
      TfLiteInterpreterDelete(pInterpreter);
    }
}

//read labels
void readLabels(const char *labelsFile, std::vector<std::string> &labels)
{
    std::string line;
    std::ifstream fin(labelsFile);
    while (getline(fin, line)) {
        labels.push_back(line);
    }
}


// The main function.
int main(void) 
{
    #define NSAMPLES 1	  //nombre imatges a classificar
    const int NUM_CLASSES =  4; //nombre de categoríes de la NN
    const char *LABELS_FILE = "/home/pi/ai_sw/Risc-V/labels/labels_intr.txt";  //fitxer amb les categoríes
    const char *MODEL_FILE =  "/home/pi/ai_sw/Risc-V/models/NoOptimized_unet.tflite";    //model de la NN
    char directori[] = "/home/pi/ai_sw/Risc-V/images_256"; //CANVIAR pel path corresponent
    
    //reading labels
    std::vector<std::string> labels;
    readLabels(LABELS_FILE, labels);
    
    double outputAcum = 0.0; //accuracy acumulat
    
    TfLiteStatus tflStatus;
    
    //variables per trobar les imatges
    struct dirent *ent;
    DIR *dir;
    char nom[251];
    
    //Busquem arxius .jpeg al path indicat a directori[]
    if ((dir = opendir((const char*) directori)) != NULL) {
     while ((ent = readdir(dir)) != NULL) {
      char *punt = NULL;
      if ((punt = strrchr(ent->d_name, '.')) != NULL) {
      //Fem una inferència per cada fitxer .jpeg que trobi
       if(!strcmp (punt, ".jpeg")) {  
    	
    	    strcpy(nom, directori);
    		
    	    strcat(nom, "/");
    		
    	    strcat(nom, ent->d_name); //guardem el path a nom
    
	    // Create JPEG image object.
	    ujImage img = ujCreate();

	    // Decode the JPEG file.
	    ujDecodeFile(img, (const char*) nom);

	    // Check if decoding was successful.
	    if(ujIsValid(img) == 0){
		return 1;
	    }
	    
	    // There will always be 3 channels.
	    int channel = 3;

	    // Height will always be 224, no need for resizing.
	    int height = ujGetHeight(img);

	    // Width will always be 224, no need for resizing.
	    int width = ujGetWidth(img);

	    // The image size is channel * height * width.
	    int imageSize = ujGetImageSize(img);

	    // Fetch RGB data from the decoded JPEG image input file.
	    uint8_t* pImage = (uint8_t*)ujGetImage(img, NULL);

	    // The array that will collect the JPEG RGB values.
	    float imageDataBuffer[imageSize];

	    // RGB range is 0-255. Scale it to 0-1.
	    int j=0;
	    for(int i = 0; i < imageSize; i++){
		imageDataBuffer[i] = (float)pImage[i] / 255.0;
	    }

	    // Load model.
	    TfLiteModel* model = TfLiteModelCreateFromFile(MODEL_FILE);
	    TfLiteInterpreterOptions *options = TfLiteInterpreterOptionsCreate();
	    
	    // Create the interpreter.
	    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

	    // Allocate tensors.
	    tflStatus = TfLiteInterpreterAllocateTensors(interpreter);

	    // Log and exit in case of error.
	    if(tflStatus != kTfLiteOk)
	    {
	      printf("Error allocating tensors.\n");
	      disposeTfLiteObjects(model, interpreter);
	      return 1;
	    }

	    // The input tensor.
	    TfLiteTensor* inputTensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
		

	    // Copy the JPEG image data into into the input tensor.
	    // Invoke the U-Net model.
	    tflStatus = TfLiteTensorCopyFromBuffer(inputTensor, imageDataBuffer, imageSize * sizeof(float)); 
		
	
	    // Log and exit in case of error.
	    if(tflStatus != kTfLiteOk)
	    {
	      printf("Error copying input from buffer.\n");
	      disposeTfLiteObjects(model, interpreter);
	      return 1;
	    }

	    // Invoke interpreter.
	    tflStatus = TfLiteInterpreterInvoke(interpreter);

	    // Log and exit in case of error.
	    if(tflStatus != kTfLiteOk)
	    {
	      printf("Error invoking interpreter.\n");
	      disposeTfLiteObjects(model, interpreter);
	      return 1;
	    }

	    // Extract the output tensor data.
	    const TfLiteTensor* outputTensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
	    
		// Get the dimensions of the output tensor.
		int outputHeight = TfLiteTensorDim(outputTensor, 1);
		int outputWidth = TfLiteTensorDim(outputTensor, 2);

		// Ensure that the output tensor dimensions are 256x256.
		if (outputHeight != 256 || outputWidth != 256) {
    		printf("Error: Output tensor dimensions are not 256x256.\n");
    		disposeTfLiteObjects(model, interpreter);
    		return 1;
		}

		// Copy the output tensor data (256x256 segmentation mask) to a buffer.
		float segmentationMask[256 * 256];
		tflStatus = TfLiteTensorCopyToBuffer(outputTensor, segmentationMask, sizeof(segmentationMask));

		// Print the segmentation mask values to the terminal.
		for (int i = 0; i < 256; ++i) {
   		 for (int j = 0; j < 256; ++j) {
       		 // Print the value of the segmentation mask at (i, j).
        		std::cout << segmentationMask[i * 256 + j] << " ";
   		 }
    		std::cout << std::endl; // Move to the next line in the terminal.
}

	 

	    // Dispose of the TensorFlow objects.
	    disposeTfLiteObjects(model, interpreter);
	    
	    // Dispoice of the image object.
	    ujFree(img);
	}
       }
      } closedir(dir); //tanquem directori de les imatges
     }else
     	{ printf("No s'ha pogut trobar el directori.");
     	  return 1;
     	}
     
    
    return 0;
}
