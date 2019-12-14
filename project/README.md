# CS221-Project/project

This project folder contains all the data, code, and experimental results from our testing.

Our final model and a complete script to train and test it on the dataset is in [final-CNN.py](final-CNN.py).

The [data](data) folder contains the datasets we used along with scripts used to preprocess them, [experiments](experiments) contains training logs of the performance of all the models we experimented with, and [models](models) contains the pre-trained weights with the highest validation accuracy of each of our models saved in PyTorch using `torch.save(model.state_dict(), model_path)`.