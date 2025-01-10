## Criterions

A criterion defines the details of our loss function.

Key files for our project are the following:

- `ce_criterion.py`: cross entropy loss, used in classification
- `reconstruction_norm.py`: computes L1, L2, or Smooth L1  loss used for evaluating the quality of a reconstruction (waveform or spectrogram). Depending on given parameters, it can compute the loss over masked patches only, or over visible patches as well, with a normalizing factor.

**How to add a new criterion**
1. Add the code for the task to this subdirectory.
2. Add the configuration file of the model to [`./config/criterion`](https://github.com/ofsoundof/TimeFM/tree/split_attention_foundation_model/config/criterion).
