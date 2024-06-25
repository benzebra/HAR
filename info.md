# MNIST
Image Dataset -> Number recognition from pictures 

### Centralized Learning 
- PyTorch **(to dev)**
- Tensorflow **(done)**

### Federated Learning 
- Flower + PyTorch (FEMNIST, use `Tao Chan gitrepo: https://github.com/tao-shen/FEMNIST_pytorch/blob/master/femnist.py` ) **(to lookup)**
- Flower + Tensorflow **(to imp)**


# UCI HAR
Tabular Dataset -> Activity recognition from data

### Centralized Learning 
- Scikit-Learn **(to imp)**
- XGBoost **(to dev)**
- Tensorflow **(to dev)**

### Federated Learning
- Flower + Scikit-Learn **(to imp)**
- Flower + XGBoost **(to dev)**
- Flower + Tensorflow **(to dev)**

---

To implement the model used by tao chan in pytorch femnist to scikit-learn

--- 

I think that tensorflow could be the most suitable solution for all of the problems. It allows to use `keras.datasets.mnist` for MNIST centralized and federated solution and I'll see its behaviour with tabular dataset such as `UCI HAR`

---

| PyTorch                           | Keras                                          |
|-----------------------------------|------------------------------------------------|
|  Conv2d(1, 32, kernel_size=3)     |  Conv2D(32, kernel_size=3)                     |
|  MaxPool2d(2, stride=2)           |  MaxPooling2D(pool_size=(2, 2), strides=2)     |
|  Conv2d(32, 64, kernel_size=3)    |  Conv2D(64, kernel_size=3)                     |
|  Dropout(0.25)                    |  Dropout(0.25)                                 |
|  Flatten()                        |  Flatten()                                     |
|  Linear(9216, 128)                |  Dense(128, activation="linear")               |
|  Dropout(0.5)                     |  Dropout(0.5)                                  |
|  Linear(128, num_classes)         |  Dense(num_classes, activation="linear")       |
|  ReLU()                           |  ReLU                                          |