# CS221-Project

Detection of photorealistic CGI from photographic images

## Documents

You can find compiled documentation of this research project in the [documents](documents/) folder. This included a project proposal, a mid-progress report, a final poster, and a research paper.

## Project

The [project](project/) folder contains all the data, code, and experimental results from our testing. Our final model and a complete script to train and test it on the dataset is in [project/final-CNN.py](project/final-CNN.py).

The [project/data](project/data) folder contains the datasets we used along with scripts used to preprocess them, [project/experiments](project/experiments) contains training logs of the performance of all the models we experimented with, and [project/models](project/models) contains the pre-trained weights with the highest validation accuracy of each of our models saved in PyTorch using `torch.save(model.state_dict(), model_path)`.

## Authors

This project was made by

* **Shawn Zhang** - Stanford University '22 - *Initial work* - [shawnbzhang](https://github.com/shawnbzhang)
* **Gabriel Mukobi** - Stanford University '22 - *Initial work* - [mukobi](https://github.com/mukobi)
