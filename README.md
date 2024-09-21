# Spatio-Temporal Convolutional Graph Network for Beach Water Quality Forecasting (STGCN-WQ)

This study developed a new prediction model: a deep learning model based on Spatio-Temporal Graph Convolutional Networks for beach Water Quality forecasting (**STGCN-WQ**). Additionally, the study proposed a Spatio-Then-Temporal imputation strategy to effectively handle missing data in the dataset, enhancing the robustness of the model.
Experimental results show that the STGCN-WQ model improves the F1 score and AUC value by 32\% and 23\%, respectively, compared to the baseline model, validating the effectiveness and superiority of this model. 

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Features](#features)
- [License](#license)

## Requirements

The following libraries are required to run the project:

- Python 3.x
- pandas
- numpy
- matplotlib
- torch
- scikit-learn

See the `requirements.txt` for a full list of dependencies.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/lps20/STGCN-WQ
   cd STGCN-WQ
2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To run the neural network classifier, use the command-line arguments to specify the dataset, model parameters, and other options.
```bash
python run.py
```
## Project Structure
```bash
.
├── main.py            # Main script to run the training and evaluation
├── model.py           # Neural network model definition
├── utils.py           # Utility functions for data processing
├── requirements.txt   # Python dependencies
├── README.md          # Project documentation
└── data/              # Directory to store datasets
```

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
### Explanation of Key Sections:
1. **Requirements**: Lists the required Python packages.
2. **Installation**: Explains how to clone the project and install dependencies.
3. **Usage**: Provides an example command for running the script and describes the parameters.
4. **Project Structure**: Outlines the main files and their purposes.
5. **Features**: Summarizes the key functionality of the project.

You can update the repository URL, file paths, and any other specific details based on your actual project setup. Let me know if you need more detailed instructions or examples!
