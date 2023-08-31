# Cloud Classification System



## Description
The objective of Task 1 is to build a robust deep learning system that can accurately classify images based on the percentage of cloud coverage. The system will be trained on a large dataset of labeled images, where each image is assigned a class based on the cloud coverage percentage it represents. The classes could include categories such as "Clear Sky," "Partly Cloudy,",  "Overcast," and so on (or 0 to 25, 25 to 50, 50 to 75 and 75 t0 100 of percentage)

The complete process for Task 1 includes the following steps:

1. **Dataset Preparation:**

-***Partial dataset download:*** A portion of the "cloudsen12" dataset was obtained from a reliable source. This subset includes a selection of images representing various cloud coverage scenarios.

-***Dataset annotation and labeling***: The dataset is cloudsen12. Since the dataset does not contain labels in txt format, a script was made to inspectionate each mask corresponding to each image to provide the cloud coverage, then the images were categorized into appropriate classes according to the predetermined criteria. 

-***Dataset structuring:*** The dataset was organized in a structured format suitable for model training and evaluation. Directories were created to store images belonging to different classes, ensuring easy access and management of the data.

-***Data Preprocessing:*** Perform necessary preprocessing steps on the collected data, such as resizing, normalizing, and augmenting the images. Split the dataset into training, validation, and testing sets.

2. **Model Selection and Architecture Design**: The most popular models for classification were proposed due their performans in the same or similar tasks(more documentation about the sources will be added here) as : 

- Resnet 50 
- Mobilnet
- VGG16
- InceptionV3

3. **Model Training**: To train each model and to find the best conbination of hyperparameter an optuna study was implemented where the goal was to utilize the appropriate loss functions, optimization algorithms (e.g., Stochastic Gradient Descent), and regularization techniques to improve the model's performance.

4. **Model Evaluation**: Once the hyperparameters were found, some evaluation metrics such as accuracy, precision, and recall were used.

5. **Inference and Deployment**: Once the model achieves satisfactory performance on the validation set, evaluate its performance on the test set to obtain unbiased performance metrics. Save the trained model for future use and deploy it in a production environment for inference on new, unseen images.The best weights for all models will be update in the following drive link:(link)

The completion of Task 1 will result in a deep learning system capable of accurately classifying images based on the percentage of cloud coverage. This system can be used for various applications, such as Risc-V implementation, satellite image analysis, and climate studies.

Remember to explore the `Task1` directory in the repository for detailed code implementation, documentation, and any additional resources related to this task.


## Files

1. **dd.py**

   This script serves the purpose of automating the download and extraction of subsets from the CloudSen12 datasets, which are hosted remotely. This script offers a streamlined way to acquire a specified number of subsets from the CloudSen12 dataset, saving you time and effort. The script is designed to be user-friendly, utilizing command-line arguments to enable customization according to your needs

###Running the Script

- **Run the Script:** Open your Command Prompt or Terminal and navigate to the directory where your `download_script.py` and `urls.txt` files are located using the `cd` command. Then, execute the following command to run the script, providing the necessary command-line arguments:

   ```sh
   python download_script.py --txt urls.txt --n <number_of_subsets> --outputPath <output_directory>
   ```

   Replace `<number_of_subsets>` with the desired number of subsets you want to download and `<output_directory>` with the path where you want the downloaded and extracted subsets to be saved.

- **Monitor the Output:** As the script runs, it will display output indicating its progress and the actions it's taking. This output will include information about the downloaded files and the extraction process.

- **Wait for Completion:** The script will begin downloading and extracting the specified number of subsets based on the URLs in the `urls.txt` file. Depending on your system's performance and your network speed, the process may take some time. Please be patient and allow the script to finish.

---




2. **split_cloudsen.py**


 Python script to process and organize the CloudSen12 dataset. The script performs various operations, including splitting data into training and testing subsets, converting labels, and converting TIF images to JPG (it will have the option to use .tif or .png in the future)format. Follow the steps below to effectively use the script:

### Running the Script

- **Customize Paths:** Modify the `path_in` and `path_out` variables to match your file system paths. The `path_in` variable should point to the directory containing the original CloudSen12 subsets, while the `path_out` variable should specify where the processed and split subsets will be stored.

- **Specify Label Type:** Set the `type_of_label` variable according to the desired type of label to use. Options include `"manual_hq"`.

- **Create Necessary Folders:** The script includes a function to create necessary folders if they don't already exist. You can keep or customize this functionality based on your needs.

- **Split Data:** The script divides the data into training and testing subsets. The default split factor is 80% for training and 20% for testing. You can adjust this split ratio as needed.

- **Running the Script:** Open a terminal or command prompt and navigate to the directory containing the script using the `cd` command. Run the script using the following command:

   ```sh
   python split_cloudsen.py
   ```

   The script will perform the defined operations on the CloudSen12 dataset data downloaded from  the **dd.py** script.

- **Output:**

   - The script will split the data into training and testing subsets, organizing them under the `path_out` directory in subdirectories named "train" and "test" respectively.
   - It will also convert TIF images to JPG format for compatibility and easier handling.



3. **test_classifier.py**

 Script for conducting hyperparameter optimization on the CloudSen12 classifier using the Optuna library. This script encompasses various operations, including model creation, optimization, and result analysis. Follow the steps below to effectively utilize the script:


### Running the Script

- **Customize Paths:** Modify the `train_data_dir` and `test_data_dir` variables to point to the respective directories containing your training and testing data subsets.

- **Run the Script:** Open a terminal or command prompt and navigate to the directory containing the script using the `cd` command. Run the script using the following command:

   ```sh
   python cloudsen12_optimization.py
   ```

   The script will begin the hyperparameter optimization process using Optuna.

### Output

- The script performs hyperparameter optimization on the CloudSen12 classifier using the Optuna library.
- Optimization progress and results will be displayed in the terminal as the script runs.

### Notes

- The script includes functions to create a model architecture and an objective function for optimization. The model architecture uses a ResNet50 base model with additional layers, you can change the model manualy in order to create trainings for other kind of architectures.
- Hyperparameter tuning includes optimizer selection, data augmentation parameters, and other configuration settings.
- The script will generate several plots to visualize the optimization history, parameter importances, slices, and contours. These plots are saved as image files.
- The best model will be saved as  a .h5 file 

Feel free to adapt the script according to your needs. It's important to review the script, especially the paths and hyperparameter options, to ensure they match your dataset and objectives. If you encounter any issues or have questions, don't hesitate to reach out for assistance.





4. **ROC_and_conf_mat.py** 

 Script to analyze the performance of the CloudSen12 classifier using ROC curves and confusion matrix visualization. This script calculates and visualizes these metrics to provide insights into the model's classification performance.



 
- **Customize Paths:** Modify the `train_data_dir` and `test_data_dir` variables to point to the respective directories containing your training and testing data subsets. Additionally, you can customize the `initial_model` variable to point to your trained model file.

- **Run the Script:** Open a terminal or command prompt and navigate to the directory containing the script using the `cd` command. Run the script using the following command:

   ```sh
   python cloudsen12_evaluation.py
   ```

   The script will calculate and visualize the ROC curves and confusion matrix.

### Output

- The script calculates and displays the average ROC curve along with individual ROC curves for each class, as well as their respective AUC scores.
- Additionally, the script creates a confusion matrix for classification analysis and displays it using a heatmap.

### Notes

- The script uses a trained model loaded from the specified `initial_model` path.
- ROC curves are generated to visualize the true positive rate against the false positive rate, helping to assess the model's performance across different thresholds.
- The confusion matrix provides an overview of predicted class labels against actual class labels, highlighting areas of correct and incorrect classification.
- Make sure to review the script and customize the paths according to your dataset and setup. 

## Usage

1. Run `dd.py` to download the dataset.

2. Run `split_cloudsen.py` to preprocess and structure the dataset.

3. Run `test_classifier.py` to perform training and hyperparameter search.

4. Get the best model and test it with `ROC_and_conf_mat.py`




