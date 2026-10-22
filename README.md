<div align="center">

# Keras & TensorFlow learning project

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3-D00000?logo=keras&logoColor=white)
![uv](https://img.shields.io/badge/package%20manager-uv-DE5FE9?logo=uv&logoColor=white)
![Task](https://img.shields.io/badge/task-sentiment%20analysis-blue)
![Dataset](https://img.shields.io/badge/dataset-IMDB%20Large%20Movie%20Review-lightgrey)
![Status](https://img.shields.io/badge/purpose-learning%20project-yellow)

</div>

A small project to refresh my Keras and TensorFlow knowledge: binary
sentiment classification (positive / negative) of IMDB movie reviews.

All training and inference logic lives in a single file, `src/tf_keras_learning/main.py`.

## Model

`TextVectorization` -> `Embedding` -> `Dropout` -> 2x `Conv1D` -> `GlobalMaxPool1D`
-> `Dense` -> `Dropout` -> `Dense(sigmoid)`.

The text vectorization layer is part of the exported `.keras` model, so it
accepts raw strings directly.

Test accuracy: _fill in after training_ 

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/)

```sh
uv sync
```

## References

- [Text classification from scratch](https://keras.io/examples/nlp/text_classification_from_scratch/)
