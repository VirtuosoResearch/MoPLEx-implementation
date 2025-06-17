import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def get_log_prob(logits, labels, prompt_lengths):
    """
    Computes log-probability of the response part of the sequence, averaged per token.
    Args:
        logits: Tensor of shape (batch_size, seq_len, vocab_size)
        labels: Tensor of shape (batch_size, seq_len)
        prompt_lengths: Tensor of shape (batch_size,) indicating length of prompt (to exclude from logprob)

    Returns:
        log_probs: Tensor of shape (batch_size,) — average log prob of the response
    """
    prompt_lengths = prompt_lengths.to(labels.device)

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)

    batch_size, seq_len = labels.shape
    response_mask = torch.arange(seq_len, device=labels.device).unsqueeze(0) >= prompt_lengths.unsqueeze(1)
    response_mask = response_mask.float()

    response_log_probs = (token_log_probs * response_mask).sum(dim=-1)
    response_lengths = response_mask.sum(dim=-1).clamp(min=1)
    return response_log_probs / response_lengths


def get_delta_theta(current_model, ref_model):
    """
    Return a torch.nn.Parameter representing delta_theta = current - ref.
    This will be treated as a trainable parameter in approximate DPO.
    """
    current_params = torch.cat([p.flatten() for p in current_model.parameters()])
    ref_params = torch.cat([p.detach().flatten() for p in ref_model.parameters()])
    delta_init = current_params - ref_params
    return torch.nn.Parameter(delta_init)  # requires_grad=True by default



def load_ref_cache(path):
    """
    Load the reference cache from file.
    """
    return torch.load(path)


def save_ref_cache(ref_cache, path):
    """
    Save the reference cache to file.
    """
    torch.save(ref_cache, path)


def approximate_dpo_loss_cached(reward_diff_star, grad_diff, delta_theta, beta):
    """
    Compute approximate DPO loss using precomputed gradient differences and parameter deltas.
    """
    margin = beta * (grad_diff @ delta_theta) + reward_diff_star
    loss = -F.logsigmoid(margin).mean()
    reward_accuracies = (margin > 0).float()
    reward_margins = margin.detach()
    return loss, reward_accuracies, reward_margins


def calculate_DPO_loss(model_preferred_logprob, model_dispreferred_logprob,
                       ref_preferred_logprob, ref_dispreferred_logprob,
                       beta=0.5):
    """
    Computes the DPO loss and associated statistics.

    Returns:
        loss: scalar tensor
        preferred_relative_logprob: scalar mean
        dispreferred_relative_logprob: scalar mean
        reward_accuracies: scalar mean
        reward_margins: scalar mean
    """
    preferred_relative_logprob = model_preferred_logprob - ref_preferred_logprob
    dispreferred_relative_logprob = model_dispreferred_logprob - ref_dispreferred_logprob

    reward_accuracies = (preferred_relative_logprob > dispreferred_relative_logprob).float().mean()
    reward_margins = (preferred_relative_logprob - dispreferred_relative_logprob).mean()

    loss = -F.logsigmoid(beta * (preferred_relative_logprob - dispreferred_relative_logprob)).mean()

    return loss, preferred_relative_logprob.mean(), dispreferred_relative_logprob.mean(), reward_accuracies, reward_margins



def generate_ref_cache(ref_model, dataloader, device):
    ref_model.train()
    for param in ref_model.parameters():
        param.requires_grad = True
 
    ref_cache = {}

    for batch in tqdm(dataloader, desc="Generating ref cache"):
        sample_ids = batch['sample_ids'].tolist()

        # Preferred
        logits_pref = ref_model(
            input_ids=batch['prompt_preferred_ids'],
            attention_mask=batch['prompt_preferred_mask']
        ).logits
        logp_pref = get_log_prob(logits_pref, batch['prompt_preferred_ids'], batch['prompt_lengths'])
        grads_pref = torch.autograd.grad(logp_pref.sum(), ref_model.parameters(), retain_graph=True, allow_unused=True)
        grads_pref = torch.cat([g.flatten() for g in grads_pref if g is not None])

        # Dispreferred
        logits_dispref = ref_model(
            input_ids=batch['prompt_dispreferred_ids'],
            attention_mask=batch['prompt_dispreferred_mask']
        ).logits
        logp_dispref = get_log_prob(logits_dispref, batch['prompt_dispreferred_ids'], batch['prompt_lengths'])
        grads_dispref = torch.autograd.grad(logp_dispref.sum(), ref_model.parameters(), retain_graph=True, allow_unused=True)
        grads_dispref = torch.cat([g.flatten() for g in grads_dispref if g is not None])

        for i, sample_id in enumerate(sample_ids):
            ref_cache[f"{sample_id}_preferred"] = {
                'logp': logp_pref[i].detach().to(device),
                'grad': grads_pref.detach().to(device)
            }
            ref_cache[f"{sample_id}_dispreferred"] = {
                'logp': logp_dispref[i].detach().to(device),
                'grad': grads_dispref.detach().to(device)
            }

    return ref_cache






