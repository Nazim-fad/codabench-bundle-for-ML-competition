# How to participate

Participants must submit a `submission.py` file defining a `get_model()` function, which returns an model object. This object must implement two methods:

* `fit(train_features, train_labels, data_dir)`
* `predict(audio_path)`

The `predict` method must return a list of detected emergency segments, where each segment is a dictionary with `start` and `end`. For example:

```python
[
    {"start": 0.5, "end": 1.7},
    {"start": 3.2, "end": 4.0},
]
```

If no emergency is detected, the model should return an empty list:

```python
[]
```

See the "Seed" page for the outline of a `Model` class, with the expected
function names.

See the "Timeline" page for additional information about the phases of this
competition
