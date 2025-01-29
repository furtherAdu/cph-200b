import torch


def cox_partial_likelihood(risk_scores, c, t, weights=None):
    event_idxs = (c.squeeze() == 1).nonzero()

    if weights is None:
        weights = torch.ones_like(risk_scores)

    # Compute the log-likelihood
    log_likelihood = torch.tensor([0.0], requires_grad=True)
    for i in event_idxs:
        at_risk = (t >= t[i]).squeeze()  # Breslow's method for ties

        log_likelihood = (
            log_likelihood 
            + torch.log(weights[i])  # following log product rule
            + risk_scores[i] 
            - torch.logsumexp(risk_scores[at_risk] + torch.log(weights[at_risk]), dim=0)
            )

    return -log_likelihood / c.sum()  # Return negative log-likelihood for minimization