import pandas as pd
import numpy as np


# ---- Convert to numpy spike trains tailored to GPFA ---- #
def dataframe_to_spike_trains(df, channels):
    """Convert trial x (variable) time x neuron spiking dataframe to list of numpy spike trains"""
    seqs = []
    
    
    # collect amount of bins as well as spike train for each trial:
    for trial in df['trial_id'].unique():
        n_bins = df[df['trial_id']==trial].shape[0]
        y = np.sqrt(df[df['trial_id']==trial][channels].T.values)
        seqs.append(
        (n_bins, y)
        )
    
    seqs = np.array(seqs, dtype=[('T', int), ('y', 'O')])
    
    # does any neuron never spike?
    # keep neurons that spike at least once only
    
    has_spikes_bool = np.hstack(seqs['y']).any(axis=1)
    for seq in seqs:
        seq['y'] = seq['y'][has_spikes_bool, :]
    
    return seqs