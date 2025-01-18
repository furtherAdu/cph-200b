import torch


def cox_partial_likelihood(risk_scores, c, t):
    event = (c.squeeze() == 1).nonzero()

    # Compute the log-likelihood
    log_likelihood = torch.tensor([0.0], requires_grad=True)
    for i in event:
        at_risk = t >= i  # Breslow's method for ties
        log_likelihood = log_likelihood + risk_scores[i] - torch.logsumexp(risk_scores[at_risk], dim=0)

    return -log_likelihood / c.sum()  # Return negative log-likelihood for minimization