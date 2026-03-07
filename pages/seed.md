# Seed:

Participants must submit a `submission.py` file defining a `get_model()` function, which returns an model object. This object must implement two methods:

* `fit(train_features, train_labels, data_dir)`
* `predict(audio_path)`


```python
class Model:
    def __init__(self):
        pass

    def fit(self, train_features, train_labels, data_dir):
        pass

    def predict(self, audio_path):
        pass


def get_model():
    return Model()
```

See `solution/submission.py` in the given bundle for a baseline model.
