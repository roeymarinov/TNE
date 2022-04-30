## The Model
Our model is based on the coupled variant in the original TNE paper, with necessary changes to the pretrained MLM, NP representation, and the architecture. Our dataset also originates in the same paper and we use the 3,988 document training set for training, and the 500 validation set for validation and testing.

## Getting Started

Install dependencies
```shell
conda create -n tne python==3.7 anaconda
conda activate tne

pip install -r requirements.txt
```

## train
To train our models on the first set of experiments, run:

```bash
allennlp train tne/modeling/configs/experiments/first_set/<name_of_experiment>.jsonnet \
         --include-package tne \
         -s models/<name_of_experiment>
```

To train our models on the second set of experiments, run the same command but with "second_set" instead of "first_set"

To train our model experiment with np recognition, run:

```bash
allennlp train tne/modeling/configs/experiments/np_recognition.jsonnet \
         --include-package tne \
         -s models/np_recognition
```


