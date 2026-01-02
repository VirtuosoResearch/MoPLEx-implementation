"""
Group-conditional scoring model for RLCHF.
Implements s_θ(x, y, z) using K heads or group embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from trainers.modeling_base import PreTrainedModelWrapper
from typing import Optional


class GroupConditionalScoringHead(nn.Module):
    """
    Group-conditional scoring head that outputs K scores (one per latent group).
    
    Option 1: K separate heads (simple implementation)
    Option 2: Group embeddings concatenated to hidden states
    """
    
    def __init__(
        self,
        config,
        num_groups: int,
        use_group_embeddings: bool = False,
        group_embedding_dim: int = 64,
        **kwargs
    ):
        super().__init__()
        self.num_groups = num_groups
        self.use_group_embeddings = use_group_embeddings
        
        # Get hidden size
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        else:
            hidden_size = config.hidden_size
        
        if use_group_embeddings:
            # Option 2: Use group embeddings
            self.group_embeddings = nn.Embedding(num_groups, group_embedding_dim)
            self.dropout = nn.Dropout(kwargs.pop("summary_dropout_prob", 0.1) if kwargs.get("summary_dropout_prob") else 0.1)
            input_dim = hidden_size + group_embedding_dim
            self.summary = nn.Linear(input_dim, 1)
        else:
            # Option 1: K separate heads (simpler)
            self.dropout = nn.Dropout(kwargs.pop("summary_dropout_prob", 0.1) if kwargs.get("summary_dropout_prob") else 0.1)
            # Create K heads
            self.heads = nn.ModuleList([
                nn.Linear(hidden_size, 1) for _ in range(num_groups)
            ])
    
    def forward(self, hidden_states, group_ids: Optional[torch.LongTensor] = None):
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size) or (batch_size, hidden_size)
            group_ids: (batch_size,) or None. If None and use_group_embeddings=True, will return all K scores.
                      If None and use_group_embeddings=False, will return all K scores.
        Returns:
            scores: (batch_size, seq_len, num_groups) or (batch_size, num_groups) if group_ids is None
                    (batch_size, seq_len, 1) or (batch_size, 1) if group_ids is specified
        """
        output = self.dropout(hidden_states)
        
        if output.dtype != torch.float32:
            output = output.to(torch.float32)
        
        if self.use_group_embeddings:
            if group_ids is not None:
                # Get embeddings for specified groups
                group_embeds = self.group_embeddings(group_ids)  # (batch_size, group_embedding_dim)
                # Expand to match sequence length if needed
                if output.dim() == 3:
                    group_embeds = group_embeds.unsqueeze(1).expand(-1, output.size(1), -1)
                # Concatenate
                output = torch.cat([output, group_embeds], dim=-1)
                scores = self.summary(output)  # (batch_size, seq_len, 1) or (batch_size, 1)
                return scores
            else:
                # Return scores for all groups
                all_scores = []
                for z in range(self.num_groups):
                    group_ids_z = torch.full((output.size(0),), z, dtype=torch.long, device=output.device)
                    group_embeds = self.group_embeddings(group_ids_z)
                    if output.dim() == 3:
                        group_embeds = group_embeds.unsqueeze(1).expand(-1, output.size(1), -1)
                    output_with_embed = torch.cat([output, group_embeds], dim=-1)
                    score_z = self.summary(output_with_embed)  # (batch_size, seq_len, 1) or (batch_size, 1)
                    all_scores.append(score_z)
                # Stack to get (batch_size, seq_len, num_groups) or (batch_size, num_groups)
                scores = torch.cat(all_scores, dim=-1)
                return scores
        else:
            # Option 1: K separate heads
            all_scores = []
            for head in self.heads:
                score = head(output)  # (batch_size, seq_len, 1) or (batch_size, 1)
                all_scores.append(score)
            # Stack to get (batch_size, seq_len, num_groups) or (batch_size, num_groups)
            scores = torch.cat(all_scores, dim=-1)
            
            if group_ids is not None:
                # Select scores for specified groups
                if scores.dim() == 3:
                    # (batch_size, seq_len, num_groups) -> (batch_size, seq_len, 1)
                    scores = torch.gather(scores, dim=2, index=group_ids.unsqueeze(1).unsqueeze(2).expand(-1, scores.size(1), -1))
                else:
                    # (batch_size, num_groups) -> (batch_size, 1)
                    scores = torch.gather(scores, dim=1, index=group_ids.unsqueeze(1))
            
            return scores


class GatingNetwork(nn.Module):
    """
    Gating network π_z(x) that outputs mixing coefficients for K groups.
    
    Can be:
    - Global: Learn a global π (independent of prompt)
    - Prompt-dependent: π depends on prompt x
    """
    
    def __init__(
        self,
        config,
        num_groups: int,
        gating_type: str = "prompt_dependent",
        hidden_size: int = 128,
        **kwargs
    ):
        super().__init__()
        self.num_groups = num_groups
        self.gating_type = gating_type
        
        if gating_type == "global":
            # Simple global mixing weights
            self.global_weights = nn.Parameter(torch.ones(num_groups) / num_groups)
        else:
            # Prompt-dependent gating
            if hasattr(config, "word_embed_proj_dim"):
                input_hidden_size = config.word_embed_proj_dim
            else:
                input_hidden_size = config.hidden_size
            
            self.gating_network = nn.Sequential(
                nn.Linear(input_hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_groups),
            )
            self.dropout = nn.Dropout(kwargs.pop("gating_dropout", 0.1) if kwargs.get("gating_dropout") else 0.1)
    
    def forward(self, hidden_states: Optional[torch.Tensor] = None):
        """
        Args:
            hidden_states: Optional. For prompt-dependent gating, this should be the hidden states
                          from the prompt (e.g., mean-pooled). Shape: (batch_size, hidden_size)
                          For global gating, this is ignored.
        Returns:
            mixing_weights: (batch_size, num_groups) with softmax applied
        """
        if self.gating_type == "global":
            # Return same weights for all samples
            batch_size = hidden_states.size(0) if hidden_states is not None else 1
            weights = F.softmax(self.global_weights, dim=-1)
            return weights.unsqueeze(0).expand(batch_size, -1)
        else:
            # Prompt-dependent
            if hidden_states is None:
                raise ValueError("hidden_states must be provided for prompt-dependent gating")
            
            # Mean pool over sequence dimension if needed
            if hidden_states.dim() == 3:
                # (batch_size, seq_len, hidden_size) -> (batch_size, hidden_size)
                hidden_states = hidden_states.mean(dim=1)
            
            output = self.dropout(hidden_states)
            logits = self.gating_network(output)  # (batch_size, num_groups)
            weights = F.softmax(logits, dim=-1)  # (batch_size, num_groups)
            return weights


class AutoModelForCausalLMWithGroupConditionalScoring(PreTrainedModelWrapper):
    """
    Model wrapper that adds group-conditional scoring heads and gating network for RLCHF.
    """
    
    transformers_parent_class = AutoModelForCausalLM
    lm_head_namings = ["lm_head", "embed_out"]
    
    def __init__(
        self,
        pretrained_model,
        num_groups: int = 4,
        use_group_embeddings: bool = False,
        group_embedding_dim: int = 64,
        gating_type: str = "prompt_dependent",
        gating_hidden_size: int = 128,
        **kwargs
    ):
        super().__init__(pretrained_model)
        
        # Initialize group-conditional scoring head
        self.group_scoring_head = GroupConditionalScoringHead(
            self.pretrained_model.config,
            num_groups=num_groups,
            use_group_embeddings=use_group_embeddings,
            group_embedding_dim=group_embedding_dim,
            **kwargs
        )
        
        # Initialize gating network
        self.gating_network = GatingNetwork(
            self.pretrained_model.config,
            num_groups=num_groups,
            gating_type=gating_type,
            hidden_size=gating_hidden_size,
            **kwargs
        )
        
        self.num_groups = num_groups
        self.is_peft_model = hasattr(self.pretrained_model, "peft_config") or hasattr(self.pretrained_model, "base_model")
    
    def _has_lm_head(self):
        """Check if the model has a language model head."""
        if any(hasattr(self.pretrained_model, attribute) for attribute in self.lm_head_namings):
            return True
        if hasattr(self.pretrained_model, "base_model") and hasattr(self.pretrained_model.base_model, "model"):
            if any(hasattr(self.pretrained_model.base_model.model, attribute) for attribute in self.lm_head_namings):
                return True
        for name, _ in self.pretrained_model.named_modules():
            if any(attribute in name for attribute in self.lm_head_namings):
                return True
        return False
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        return_group_scores: bool = True,
        return_gating_weights: bool = False,
        **kwargs
    ):
        """
        Forward pass that returns group-conditional scores.
        
        Args:
            input_ids: Input token ids
            attention_mask: Attention mask
            return_group_scores: If True, return scores for all groups. If False, return only LM logits.
            return_gating_weights: If True, also return gating weights π_z(x)
        Returns:
            lm_logits: Language model logits
            loss: Language model loss (if computed)
            group_scores: Group-conditional scores (batch_size, seq_len, num_groups) or None
            gating_weights: Gating weights π_z(x) (batch_size, num_groups) or None
        """
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        
        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        last_hidden_state = base_model_output.hidden_states[-1]
        lm_logits = base_model_output.logits
        loss = getattr(base_model_output, 'loss', None)
        
        group_scores = None
        gating_weights = None
        
        if return_group_scores:
            # Get scores for all groups
            group_scores = self.group_scoring_head(last_hidden_state)  # (batch_size, seq_len, num_groups)
            
            if return_gating_weights:
                # Compute gating weights from prompt (first part of sequence)
                # For simplicity, use mean-pooled hidden states
                gating_weights = self.gating_network(last_hidden_state)  # (batch_size, num_groups)
        
        if return_group_scores or return_gating_weights:
            return (lm_logits, loss, group_scores, gating_weights)
        else:
            return (lm_logits, loss)
    
    def get_group_scores(self, hidden_states, group_ids: Optional[torch.LongTensor] = None):
        """Get group-conditional scores for given hidden states."""
        return self.group_scoring_head(hidden_states, group_ids)
    
    def get_gating_weights(self, hidden_states):
        """Get gating weights π_z(x) for given hidden states."""
        return self.gating_network(hidden_states)
    
    def state_dict(self, *args, **kwargs):
        r"""
        Returns the state dictionary of the model. We add the state dictionary of the group scoring head
        and gating network to the state dictionary of the wrapped model.
        """
        if not self.is_peft_model:
            pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)
        else:
            # if it is a peft model, only save the group_scoring_head and gating_network
            pretrained_model_state_dict = {}

        group_scoring_head_state_dict = self.group_scoring_head.state_dict(*args, **kwargs)
        for k, v in group_scoring_head_state_dict.items():
            pretrained_model_state_dict[f"group_scoring_head.{k}"] = v
        
        gating_network_state_dict = self.gating_network.state_dict(*args, **kwargs)
        for k, v in gating_network_state_dict.items():
            pretrained_model_state_dict[f"gating_network.{k}"] = v
        
        return pretrained_model_state_dict

