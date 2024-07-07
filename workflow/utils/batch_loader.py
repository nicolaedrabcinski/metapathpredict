from tensorflow import keras

from sklearn.utils import shuffle

import numpy as np

class BatchLoader(keras.utils.Sequence):
    """
    Helper to iterate over the data 
    """

    def __init__(self, input_seqs, input_seqs_rc, input_labs, batches, rc=True, random_seed=1):
        self.input_seqs = input_seqs
        self.input_seqs_rc = input_seqs_rc
        self.input_labs = input_labs
        self.batches = batches
        self.rc = rc
        self.random_seed = random_seed

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        batch = sorted(self.batches[idx])
        batch_seqs, batch_seqs_rc, batch_labs = shuffle(
            np.array(self.input_seqs[batch, ...]),
            np.array(self.input_seqs_rc[batch, ...]),
            np.array(self.input_labs[batch, ...]),
            random_state=self.random_seed
        )

        if self.rc:
            return (batch_seqs, batch_seqs_rc), batch_labs
        else:
            return batch_seqs, batch_labs


class BatchGenerator:
    """
    Helper to iterate over the data
    """
    def __init__(
            self,
            input_seqs,
            input_seqs_rc,
            input_labs,
            batches,
            random_seed=1
    ):
        self.input_seqs = input_seqs
        self.input_seqs_rc = input_seqs_rc
        self.input_labs = input_labs
        self.batches = batches
        self.random_seed = random_seed

    def __call__(self):
        for batch in self.batches:
            batch = sorted(batch)
            batch_seqs, batch_seqs_rc, batch_labs = shuffle(
                np.array(self.input_seqs[batch, ...]),
                np.array(self.input_seqs_rc[batch, ...]),
                np.array(self.input_labs[batch, ...]),
                random_state=self.random_seed
            )
            yield (batch_seqs, batch_seqs_rc), batch_labs

