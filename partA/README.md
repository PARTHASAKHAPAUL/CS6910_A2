
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
