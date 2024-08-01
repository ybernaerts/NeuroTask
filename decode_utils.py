import torch
import pandas as df
import numpy as np

def estimate_readout_matrix(df, channels, m, delta, n_iter=2500, dtype=torch.float32):
    """
    Estimate readout matrix for variable lengths across trials
    """
    n_trials = df['trial_id'].unique().shape[0]
    n_neuron = len(channels)
    n_latent = m[0].shape[0]
    
    M = []
    #M = torch.zeros((n_trials, n_time_bins, n_latent))
    C_hat = torch.nn.Linear(n_latent, n_neuron, bias=True)

    for n in range(n_trials):
        if (torch.is_tensor(m[0])):
            M.append(torch.tensor(m[n].detach().clone(),dtype=dtype).T) # time x latent
        else:
            M.append(torch.tensor(m[n],dtype=dtype).T) # time x latent

    opt = torch.optim.Adam(C_hat.parameters(), lr=1e-2)
    loss_log = []
    loss_trials = torch.zeros((n_trials,))
    for i in range(n_iter):
        for n, trial_id in enumerate(df['trial_id'].unique()):
            Y = torch.as_tensor(df[df['trial_id']==trial_id][channels].values) # time x neuron
            log_r = C_hat(M[n]) # time x neuron = (time x latent) x (latent x neuron) + (neuron)
            ell = Y * log_r - delta * torch.exp(log_r)
            loss = -1*torch.sum(ell)
            
            loss.backward()
            opt.step()
            opt.zero_grad()
            
            loss_trials[n]=loss.item()
            
        loss_log.append(torch.mean(loss_trials).item())

    return C_hat, loss_log

def expected_ll_poisson(df, channels, m, p, C, delta, dtype=torch.float32):
    """
    Expected Poisson log-likelihood for variable lengths across trials.
    Expects raw spikes df of size time x neuron (how raw parquet files are set up)
    Expects latents m of size trials x latent x time (how GPFA returns them)
    Expects covariances p of size trials x time x time x latent (how GPFA returns them)
    """
    n_trials = df['trial_id'].unique().shape[0]
    n_neuron = len(channels)
    n_latent = m[0].shape[0]
    
    M = []
    P = []
    
    for n in range(n_trials):
        if (torch.is_tensor(m[0])):
            M.append(torch.tensor(m[n].detach().clone(),dtype=dtype).T) # convert to time x latent

        else:
            M.append(torch.tensor(m[n],dtype=dtype).T) # convert to time x latent
        
        if (torch.is_tensor(p[0])):
            P.append(torch.tensor(p[n].detach().clone(),dtype=dtype)) # should be in time x time x latent

        else:
            P.append(torch.tensor(p[n],dtype=dtype)) # should be time x time x latent
    
    nats=torch.zeros((n_trials,))
    
    for n, trial_id in enumerate(df['trial_id'].unique()):
        Y = torch.as_tensor(df[df['trial_id']==trial_id][channels].values) # should be in time x neuron    
        spk_count_per_trial = Y.sum(dim=0)

        log_rate = C(M[n]) + 0.5 * torch.einsum('nl, ttl, nl -> tn', C.weight, P[n], C.weight) # time x neuron
        likelihood_pdf = torch.distributions.Poisson(delta * torch.exp(log_rate))
        log_prob = likelihood_pdf.log_prob(Y)
        log_prob = log_prob.sum(dim=0) # sum over time

        null_likelihood_pdf = torch.distributions.Poisson(delta * torch.exp(C.bias) * torch.ones_like(log_rate))
        null_likelihood_log_prob = null_likelihood_pdf.log_prob(Y)
        null_likelihood_log_prob = null_likelihood_log_prob.sum(dim=0)

        bidx = spk_count_per_trial != 0 # exclude (neurons x trials) with no spikes
        nats_array = torch.mean((log_prob[bidx] - null_likelihood_log_prob[bidx]) * (1 / spk_count_per_trial[bidx]))
        nats[n]=nats_array
    
    return torch.mean(nats) / np.log(2.0) # bits/spike/neuron