# CS221-Project

Detection of photorealistic CGI from photographic images

## Documents

You can find compiled documentation of this research project in the [documents](documents/) folder. This included a [project proposal](documents/0.%20proposal.pdf), a [mid-progress report](documents/1.%20progress-report.pdf), a [final poster](documents/2.%20poster.pdf), and a [final research paper](documents/3.%20final-paper.pdf).

## Project

The [project](project/) folder contains all the data, code, and experimental results from our testing. Our final model and a complete script to train and test it on the dataset is in [project/final-CNN.py](project/final-CNN.py).

The [project/data](project/data) folder contains the datasets we used along with scripts used to preprocess them, [project/experiments](project/experiments) contains training logs of the performance of all the models we experimented with, and [project/models](project/models) contains the pre-trained weights with the highest validation accuracy of each of our models saved in PyTorch using `torch.save(model.state_dict(), model_path)`.

## Authors

This project was made by

- **Shawn Zhang** - Stanford University '22 - _Initial work_ - [shawnbzhang](https://github.com/shawnbzhang)
- **Gabriel Mukobi** - Stanford University '22 - _Initial work_ - [mukobi](https://github.com/mukobi)
