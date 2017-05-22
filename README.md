# Bidirectional Character LSTM for Sentiment Analysis 

### Requirements
- Python 2.7
- Tensorflow
- NLTK

### Setup
- Download the [datasets](http://help.sentiment140.com/for-students/) in `/datasets`
- Download the Tokenize module of NLTK using `nltk.download()`
- Change the `PATH` variable in [data_utils.py](https://github.com/charlesashby/CharLSTM/blob/master/lib/data_utils.py) and in the model files
- Train the model you want with `python main.py <MODEL_NAME> --train` 

### Using a Pretrained Model
This repository provides a pretrained model for the unidirectional model you can test your own sentences using:

```
python main.py lstm --sentences 'sentence 1' 'sentence 2' 'etc...'
```

### Model

![](charlstm_diagram.png)



