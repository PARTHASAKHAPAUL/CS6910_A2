# CS6910_A2
# ```PART: A```

# CNN Model for Image Classification
This part contains instructions on how to train and evaluate a Convolutional Neural Network (CNN) model designed for image classification. The model is implemented using PyTorch and is adaptable for various image classification datasets.

## Requirements
    Python 3.6 or higher
    PyTorch
    torchvision
    numpy

## Dataset
[iNaturalist-12K](https://www.kaggle.com/datasets/aryanpandey1109/inaturalist12k)


## Data Preparation
Before training the model, the data needs to be prepared and loaded. The ```prepare_data_loaders``` function is designed to ```augment```, ```transform```, and ```load``` the training and validation data from ```train folder``` of the ```iNaturalist dataset```. It splits the training data into ```training and validation sets (80:20)``` and applies transformations.

## Model Configuration
The my_CNN model is customized with the following parameters:

- ```num_classes```: The number of ```output classes```.
- ```num_filters```: A list specifying the ```number of filters``` for each convolutional layer.
- ```filter_sizes```: A list specifying the ```kernel size``` for each convolutional layer.
- ```activation_fn```: The activation function to use (```relu```, ```gelu```, ```silu```, ```mish```).
- ```num_neurons_dense```: The number of neurons in the ```dense layer```.
- ```dropout_rate```: The dropout rate for ```regularization```.
- ```use_batchnorm```: Whether to use batch normalization (```yes``` or ```no```).

Example of initializing the model with custom parameters:
    
    model = my_CNN(num_classes=10,
                   num_filters=[32, 64, 128, 256, 512],
                   filter_sizes=[3, 3, 3, 3, 3],
                   activation_fn='relu',
                   num_neurons_dense=512,
                   dropout_rate=0.5,
                   use_batchnorm='yes')
                   
## Training and Evaluation
To train and evaluate the model, the ```train_and_evaluate``` method of the ```my_CNN``` class is designed. This method requires the training and validation DataLoaders, the number of epochs, and the learning rate.

Here is one example of the above:

    train_loader, validation_loader = prepare_data_loaders(data_augment=True, batch_size=64)
    model.train_and_evaluate(train_loader, validation_loader, n_epochs=10, lr=0.001)
    
This function trains the model for a specified number of epochs and evaluates it on the validation set after each epoch, printing the training loss, validation loss, and validation accuracy.

## Customization
Anyone can customize the model further by modifying the parameters or by altering the ```my_CNN``` class definition to adjust the architecture as per their requirements.

## Note
Here a ```CUDA-enabled GPU```(on ```Kaggle```) is used for training the model, as ```CNN``` training is computationally intensive.





# ```PART: B```
# Fine-tuning a pre-trained model on the iNaturalist Dataset

This part describes how to fine-tune a pre-trained ```ResNet50``` model on a new dataset(```iNaturalist_12K```) using ```PyTorch```.
The process is divided into three main parts: 

    loading the pre-trained model, 
    setting up the W&B sweep configuration for hyperparameter optimization, 
    and implementing the training procedure along with data preparation.

## Goal
This part focuses on fine-tuning model(```ResNet50```) pre-trained on the ```ImageNet``` dataset to classify images from the ```iNaturalist``` dataset, which contains images of various animal species.

## Worklog
- **```Step 1:```** Pre-trained Model Loading
Models like ```ResNet50```, which have been pre-trained on the ```ImageNet``` dataset, are utilized. These models have seen a variety of animal images, making them a suitable choice for the ```iNaturalist``` dataset.

Importing the model:

    import torch
    import torch.nn as nn
    from torchvision import models
    
    # Load pre-trained ResNet50
    model = models.resnet50(pretrained=True)


- **```Step 2:```** Addressing Image Dimension Differences
The ImageNet dataset images have a different dimension than the iNaturalist dataset. To address this, we resize the images in the iNaturalist dataset to match the dimensions expected by the pre-trained models (```224x224``` pixels for ResNet50 model).

- **```Step 3:```** Modifying the Model for the New Task
Since ImageNet has ```1,000 classes``` and the iNaturalist dataset has only ```10```, the last layer of the pre-trained model, which corresponds to the class predictions, is replaced with a new layer with 10 outputs. This step is crucial for running the model to the specific classification task.

- **```Step 4:```** Fine-Tuning Strategy
To make training tractable on smaller datasets, we freezing the weights of all layers except the last (or last few). This means that during training, the gradients are not computed for the frozen layers, and only the weights of the unfrozen layers are updated. This approach significantly reduces the computational cost and allows for quicker experimentation.

- **```Step 5:```** Experimentation with Weights & Biases
A sweep configuration is defined for testing different hyperparameters, such as the extent of freezing layers, batch sizes, and the number of epochs. This process helps in identifying the best configuration for the model.

        'parameters': {
                'freeze': {'values': [0.7, 0.8, 0.9]},
                'epochs': {'values': [5, 7]},
                'batch_size': {'values': [32, 64]},
            }

- **```Step 6:```** Training and Evaluation
The model is then trained on the iNaturalist dataset, and its performance is evaluated.

- **```Conclusion```**
The experiments conducted demonstrate the effectiveness of fine-tuning pre-trained models on a specialized dataset
