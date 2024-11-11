# Computer Vision Experiments

Computer Vision classification experiments training and testing Convolutional Neural Networks (CNNs) to analyze the impact of different parameters and the use of techniques such as feature extraction and transfer learning.

# Results


## Impact of Different Parameters meanwhile Training

Experiments to check how much the leraning rate and the activation function affect the model's performance.

### Experiment 1 - CNN with LeakyReLU activation function, SGD optimizer with lr=0.0001

With such a low learning rate, the model is not able to learn anything.

![](docs/img/tensorboard_custom_cnn1_sgd.png)

- Test accuracy: 0.0994
- Test loss: 2.30374

All metrics in one plot:

![](docs/plots/metrics_custom_cnn1_lr0001.png)

This experiment has a very low accuracy and high loss, which is expected since the learning rate is very low.

### Experiment 2 - CNN with Adam optimizer and LeakyReLU activation function

![](docs/img/tensorboard_custom_cnn2_adam.png)

- Test accuracy: 0.4146
- Test loss: 1.605

This experiment has a higher accuracy and lower loss than the previous one, which is expected since the Adam optimizer
 adapts the learning rate over time. The model is still underfitting, as the accuracy is low and the loss is high, but
 it is an improvement over the previous experiment

All metrics in one plot:

![](docs/plots/metrics_custom_cnn2_adam.png)

### Experiment 3 - CNN with Tanh as activation function and Adam optimizer

![](docs/img/tensorboard_custom_cnn3_tanh.png)

- Test accuracy: 0.4117
- Test loss: 1.6747

This experiment has a similar accuracy and loss to the previous one, which is expected since the activation function
 does not have a significant impact on the model's performance.

All metrics in one plot:

![](docs/plots/metrics_custom_cnn3_tanh.png)

## Transfer Learning and Feature Extraction using pre-trained models

Experiments to leverage transfer learning and fine-tuning to improve the model's performance.

### 2.1 - Transfer Learning from ImageNet

#### Experiment 1 - Fine tuning

![](docs/img/tensorboard_alexnet_fine_tuning.png)

- Test loss: 0.779.
- Test accuracy: 0.7299.

All metrics in one plot:

![](docs/plots/metrics_alexnet_fine_tuning.png)

#### Experiment 2 - Feature Extraction

- Test loss: 2.41
- Test accuracy: 0.1099

### Differences between fine-tuning and feature extraction

Fine-tuning has a higher accuracy and lower loss than feature extraction. This is because fine-tuning trains the whole
 model, while feature extraction only trains the last layer.

### 2.2 - Transfer Learning from MNIST

#### Experiment 1 - CNN for MNIST

- Test loss: 0.1524
- Test accuracy: 0.9588.

![](docs/img/tensorboard_mnist_cnn.png)

The experiment has a high accuracy and low loss, which is expected since the MNIST dataset is a simple dataset.

All metrics in one plot:

![](docs/plots/metrics_mnist_cnn_train.png)


#### Experiment 2 - Pretrained MNIST CNN for SVHN dataset

- Test accuracy: 0.1856
- Test loss: 2.628

The experiment has a low accuracy and high loss, which is expected since the model was trained on a different dataset, 
but it performs better than if the model wasn't trained at all.

#### Experiment 3 - Fine-tuning MNIST CNN for SVHN dataset

- Test loss: 0.8541
- Test accuracy: 0.767

![](docs/img/tensorboard_svhn_fine_tuning.png)

The experiment has a higher accuracy and lower loss than the previous one, which is expected since the model is being
 fine-tuned on the SVHN dataset.

All metrics in one plot:

![](docs/plots/metrics_svhn_fine_tuning.png)
