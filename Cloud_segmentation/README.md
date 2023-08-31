# Cloud Segmentation Project

This project focuses on cloud segmentation using the "CloudSen12" dataset. The goal is to develop a cloud segmentation model using the U-Net architecture. The project is divided into several files, each serving a specific purpose.

## Files

1. **split_cloud_segmentation.py**

   This file, `split_cloud_segmentation.py`, is responsible for processing the batch structure of the "CloudSen12" dataset and organizing it into the required format for training the segmentation task. The script preprocesses the dataset and prepares the input data for training.

2. **utils_segmentation.py**

   `utils_segmentation.py` contains essential resources and functions required for the project. This file houses utility functions that aid in data loading, preprocessing, augmentation, and visualization. It provides a set of tools that contribute to the smooth execution of the segmentation project.

3. **unet.py**

   The file `unet.py` implements the U-Net model architecture, which is used for training the cloud segmentation model. This module includes the model definition, layers, and configurations necessary to create and train the U-Net neural network.

4. **test_label.py** 

   `test_label.py` Is responsible for orchestrating the testing process of some functions in the server.

## Usage

1. Run `split_cloud_segmentation.py` to preprocess and structure the dataset.

2. Utilize functions and resources from `utils_segmentation.py` for data handling and visualization.

3. Use the `unet.py` file to define and train the U-Net model architecture.



## Dependencies

Make sure to have the required dependencies installed. You can usually install them using package managers like pip or conda. Required dependencies might include:

- TensorFlow
- NumPy
- OpenCV
- Other project-specific dependencies

## Acknowledgments

This project is a part of a broader effort to explore cloud segmentation techniques using the "CloudSen12" dataset over risc-v architectures. Contributions and feedback are welcome.

## License

This project is licensed under the [MIT License](LICENSE).

---

