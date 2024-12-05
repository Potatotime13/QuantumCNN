# Quantum Convolutional Neural Network (QuantumCNN)

This project implements a Quantum Convolutional Neural Network (QuantumCNN) using PyTorch. The QuantumCNN is designed to process 2D input data, such as images, by leveraging quantum computing principles. The network includes both classical and quantum convolutional layers to perform image classification tasks.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Quantum computing has the potential to revolutionize machine learning by providing new ways to process and analyze data. This project explores the integration of quantum computing with classical convolutional neural networks (CNNs) to create a hybrid QuantumCNN. The QuantumCNN can be used for image classification tasks, such as recognizing handwritten digits from the MNIST dataset.

## Installation

To run this project, you need to have Python and PyTorch installed. You can install the required dependencies using the following commands:

```bash
pip install numpy torch torchvision tqdm
```

## Usage

1. Clone the repository:
    
```bash
git clone https://github.com/yourusername/QuantumCNN.git
cd QuantumCNN
```

2. Run the training script:

```bash
python QuantumCNN.py
```

The script will download the MNIST dataset, train the QuantumCNN model, and evaluate its performance on the validation set.

## Model Architecture
The QuantumCNN model consists of the following components:

In the original paper the have the following architecture:

28x28x1 --(QCONV)-> 7x7x4 --(FC)-> 11
            |                | 
Params:     4               8635

- Classical Convolutional Layers: Two classical convolutional layers are used to extract features from the input images.
- Quantum Convolutional Layer: A custom quantum convolutional layer (QuantumConv2d) is implemented to process the features using quantum operations.
- Fully Connected Layer: A fully connected layer is used to map the extracted features to the output classes.

QuantumConv2d Layer

The QuantumConv2d layer simulates a quantum circuit for 2D inputs. It includes the following quantum operations:

- Hadamard Gates: Applied to all qubits to create superposition states.
- RZ Gates: Rotation gates applied to each qubit based on the input data.
- RZZ Gates: Two-qubit rotation gates to create entanglement between qubits.
- RX Gates: Rotation gates applied to each qubit based on learnable parameters.
- CNOT Gates: Controlled-NOT gates to create entanglement between qubits.

QuantumConvNet
The QuantumConvNet class defines the overall architecture of the QuantumCNN model. It includes the classical convolutional layers, the quantum convolutional layer, and the fully connected layer.

Training and Evaluation
The training script (QuantumCNN.py) performs the following steps:

- Data Loading: The MNIST dataset is downloaded and preprocessed.
- Model Initialization: The QuantumConvNet model is initialized.
- Training Loop: The model is trained for a specified number of epochs using the training data.
- Validation: The model's performance is evaluated on the validation set after each epoch.

The training progress and validation accuracy are printed to the console.

## Contributing
Contributions to this project are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for details. ```