# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
from dataclasses import dataclass, field
from typing import Literal, Optional
import tyro
from collections.abc import Mapping

from trainers.dpo_config import DPOConfig, flatten_dict, is_wandb_available


@dataclass
class RLCHFConfig(DPOConfig):
    """
    Configuration class for RLCHF (Reinforcement Learning from Collective Human Feedback) Trainer.
    Extends DPOConfig with RLCHF-specific parameters.
    """

    # RLCHF-specific hyperparameters
    num_latent_groups: int = 4
    """Number of latent groups K (e.g., 4 to 16)"""
    
    aggregation_type: Literal["weighted_avg", "log_sum_exp"] = "weighted_avg"
    """Type of collective aggregation: 'weighted_avg' (cardinal) or 'log_sum_exp' (soft aggregation)"""
    
    aggregation_temperature: float = 1.0
    """Temperature τ for log-sum-exp aggregation. Smaller τ = more like max, larger τ = more like average"""
    
    gating_type: Literal["global", "prompt_dependent"] = "prompt_dependent"
    """Type of gating: 'global' (learn a global π) or 'prompt_dependent' (π depends on prompt x)"""
    
    gating_hidden_size: int = 128
    """Hidden size for prompt-dependent gating network"""
    
    entropy_reg_coef: float = 0.1
    """Entropy regularization coefficient to prevent collapse (prevent π from becoming too sharp)"""
    
    use_em_style: bool = False
    """Whether to use EM-style updates for responsibility q(z) instead of end-to-end backprop"""
    
    group_embedding_dim: int = 64
    """Dimension of group embedding ez (used in alternative implementation with group embeddings)"""
    
    use_group_embeddings: bool = False
    """If True, use group embeddings concatenated to hidden states. If False, use K separate heads"""
    
    stakeholder_weights: Optional[list] = None
    """Optional manual stakeholder weights wz for aggregation. If None, uses learned πz"""
    
    def __post_init__(self):
        # Call parent __post_init__ first
        super().__post_init__()
        
        if self.num_latent_groups < 2:
            raise ValueError("num_latent_groups must be at least 2")
        
        if self.aggregation_type == "log_sum_exp" and self.aggregation_temperature <= 0:
            raise ValueError("aggregation_temperature must be positive for log_sum_exp aggregation")
        
        if self.entropy_reg_coef < 0:
            raise ValueError("entropy_reg_coef must be non-negative")
        
        if self.stakeholder_weights is not None:
            if len(self.stakeholder_weights) != self.num_latent_groups:
                raise ValueError(f"stakeholder_weights length ({len(self.stakeholder_weights)}) must match num_latent_groups ({self.num_latent_groups})")
            if abs(sum(self.stakeholder_weights) - 1.0) > 1e-6:
                raise ValueError("stakeholder_weights must sum to 1.0")

