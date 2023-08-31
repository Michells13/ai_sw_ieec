# AI_SW Repository

Welcome to the AI_SW repository! This repository is dedicated to the development and implementation of deep learning systems on RISC-V based platforms. Our team is focused on leveraging the power of AI to enhance the capabilities of RISC-V systems. the work is part of  a TFM that is described in the following article https://www.overleaf.com/read/hbgypqwfgxnw.

## Objective
The main objective of our AI team is to advance the field of deep learning and facilitate its integration with RISC-V architectures. We aim to develop and implement cutting-edge AI models and algorithms that can be efficiently deployed on RISC-V-based systems. By bridging the gap between AI and RISC-V, we strive to unlock new opportunities for intelligent applications in various domains.

## Repository Structure
This repository is organized into different tasks, each addressing specific aspects of AI implementation on RISC-V. Here's an overview of the repository structure:

- **Cloud detection/classification**: The first task in the AI_SW repository focuses on creating a deep learning system capable of classifying images based on the percentage of clouds present in each image. The objective is to develop an accurate and efficient system that can classify images into different classes depending on the cloud coverage. This task encompasses the entire pipeline, including training the model, evaluating its performance, and deploying it for inference. 
- **Cloud Segmentation**: The second task of this project involves semantic segmentation. What we aim to achieve in this task is to identify clouds at a pixel level and then create a mask that enables us to visualize the areas of the image that are covered by clouds. Additionally, the project contains a script that generates the necessary data structure for training the system, taking into account the structure of cloud data from the Sen12 dataset.

- **Object Detection **: (pending)

- **Risc-V implementation - QEMU Emulation **: In order to empirically evaluate the efficacy of the models and deep learning architectures within a RISC-V analogous framework, our methodology necessitated the utilization of the QEMU virtualization software alongside the TensorFlow Lite libraries. Subsequently, it was imperative to undertake a comprehensive process involving code compilation and intricate modifications to the existing Alex (TFG Student project) implementation. This was executed with the paramount objective of ensuring its seamless functionality within the established experimental setup.


Please navigate to the relevant task directories to explore the specific implementation details and code related to each task.

## Installation
To set up the necessary environment for running the code in this repository, we provide an `environment.yml` file. The `environment.yml` file contains a list of dependencies and their versions required for the system. 

To install the requirements, follow these steps:

1. Clone this repository: `git clone https://gitlab.ieec.cat/Vibria/AI_SW.git`
2. Navigate to the repository directory: `cd AI_SW`
3. Create a new virtual environment (optional but recommended): `python -m venv env`
4. Activate the virtual environment:
   - On Windows: `.\env\Scripts\activate`
   - On macOS and Linux: `source env/bin/activate`
   We strongly encourage the utilization of the Linux operating system.
5. Install the dependencies: `conda env update --file environment.yml`

Once the installation is complete, you'll have all the necessary packages and libraries set up in your environment to run the code smoothly.

## Contributing
We welcome contributions to the AI_SW repository! If you'd like to contribute, please follow these guidelines:

1. Fork the repository and create a new branch for your contribution.
2. Make your changes and ensure that the code is functioning properly.
3. Write clear and concise commit messages.
4. Submit a pull request, describing the changes you've made.

Please note that all contributions are subject to review, and we appreciate your understanding and cooperation in maintaining the quality of the repository.

## Contact
If you have any questions, suggestions, or feedback related to this repository or our AI team, please feel free to reach out to us. You can contact us at vargas@ieec.cat

We look forward to collaborating with you on AI and RISC-V!

Happy coding!

