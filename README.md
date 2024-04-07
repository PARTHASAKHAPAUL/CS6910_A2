# CS6910_A2

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
