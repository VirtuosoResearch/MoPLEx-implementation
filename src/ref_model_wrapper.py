import torch
from dpo_cache_utils import get_log_prob

class RefModel:
    def __init__(self, mode, model=None, cache=None):
        """
        A unified wrapper for reference model or reference cache.

        Parameters:
        - mode (str): 'exact' or 'approx'
        - model (transformers.PreTrainedModel): reference model (required if mode == 'exact')
        - cache (dict): dictionary of cached logprobs and grads (required if mode == 'approx')
        """
        assert mode in ['exact', 'approx'], "mode must be either 'exact' or 'approx'"
        self.mode = mode
        self.model = model
        self.cache = cache

    def get_log_probs(self, input_ids=None, attention_mask=None, prompt_lengths=None, ids=None):
        """
        Returns log probabilities or cached values depending on mode.

        - In 'exact' mode: requires input_ids, attention_mask, prompt_lengths.
        - In 'approx' mode: requires ids (list of string IDs).
        """
        if self.mode == 'exact':
            assert self.model is not None, "Reference model is not provided."
            with torch.no_grad():
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                return get_log_prob(logits, input_ids, prompt_lengths)

        elif self.mode == 'approx':
            assert self.cache is not None, "Reference cache is not provided."
            assert ids is not None, "IDs must be provided in approx mode."

            logps = []
            grads = []
            for id_ in ids:
                item = self.cache[id_]
                logps.append(item['logp'])
                grads.append(item['grad'])

            device = grads[0].device if torch.is_tensor(grads[0]) else torch.device('cpu')
            return torch.tensor(logps, device=device), torch.stack(grads).to(device)

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
