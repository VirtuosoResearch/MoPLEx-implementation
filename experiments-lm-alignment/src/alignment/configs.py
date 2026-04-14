# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass, field
from typing import Any, Optional

import trl


@dataclass
class DatasetConfig:
    """Configuration for a dataset in a mixture."""

    id: str
    config: Optional[str] = None
    split: str = "train"
    columns: Optional[list[str]] = None
    weight: Optional[float] = None


@dataclass
class DatasetMixtureConfig:
    """Configuration for a mixture of datasets."""

    datasets: list[DatasetConfig]
    seed: int = 0
    test_split_size: Optional[float] = None


@dataclass
class ScriptArguments(trl.ScriptArguments):
    """
    Extended version of ScriptArguments with support for dataset mixtures.

    Args:
        dataset_mixture (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Configuration for creating dataset mixtures with advanced options.
            Format:
              dataset_mixture:
                datasets:
                  - id: dataset_id1
                    config: config_name
                    columns:
                      - col1
                      - col2
                    weight: 0.5
                  - id: dataset_id2
                    config: config_name
                    columns:
                      - col1
                      - col2
                    weight: 0.5
                seed: 42
                test_split_size: 0.1
    """

    dataset_mixture: Optional[dict[str, Any]] = field(
        default=None,
        metadata={"help": "Configuration for creating dataset mixtures with advanced options like shuffling."},
    )
    dataset_format: str = field(
        default="pairwise",
        metadata={"help": "Dataset format to load: 'pairwise' (default) or 'listwise'."},
    )
    preference_dimension: Optional[str] = field(
        default=None,
        metadata={
            "help": "Preference dimension for listwise training (e.g. helpfulness, honesty, instruction_following, truthfulness)."
        },
    )
    listwise_num_responses: int = field(
        default=4,
        metadata={"help": "Number of ranked responses per prompt to keep for listwise training."},
    )
    listwise_prompt_column: str = field(
        default="instruction",
        metadata={"help": "Prompt column in the raw listwise dataset."},
    )
    listwise_responses_column: str = field(
        default="completions",
        metadata={"help": "Column containing candidate responses in listwise datasets."},
    )
    listwise_response_text_key: str = field(
        default="response",
        metadata={"help": "Key used to read response text from each completion item."},
    )
    listwise_scores_key: str = field( # not used in UltraFeedback
        default="scores",
        metadata={"help": "Key used to read per-dimension scores from each completion item."},
    )
    listwise_annotations_key: str = field(
        default="annotations",
        metadata={"help": "Fallback key used to read per-dimension scores from each completion item."},
    )
    listwise_min_responses: int = field(
        default=2,
        metadata={"help": "Drop prompts with fewer than this number of scored responses."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.dataset_mixture is None:
            raise ValueError("Either `dataset_name` or `dataset_mixture` must be provided")

        if self.dataset_format not in {"pairwise", "listwise"}:
            raise ValueError("`dataset_format` must be either 'pairwise' or 'listwise'")
        if self.dataset_format == "listwise" and self.preference_dimension is None:
            raise ValueError("`preference_dimension` is required when dataset_format='listwise'")
        if self.listwise_num_responses < 2:
            raise ValueError("`listwise_num_responses` must be >= 2")
        if self.listwise_min_responses < 2:
            raise ValueError("`listwise_min_responses` must be >= 2")

        if self.dataset_mixture is not None:
            if not isinstance(self.dataset_mixture, dict) or "datasets" not in self.dataset_mixture:
                raise ValueError(
                    "dataset_mixture must be a dictionary with a 'datasets' key. "
                    "Expected format: {'datasets': [...], 'seed': int}"
                )

            datasets_list = []
            datasets_data = self.dataset_mixture.get("datasets", [])

            if isinstance(datasets_data, list):
                for dataset_config in datasets_data:
                    datasets_list.append(
                        DatasetConfig(
                            id=dataset_config.get("id"),
                            config=dataset_config.get("config"),
                            split=dataset_config.get("split", "train"),
                            columns=dataset_config.get("columns"),
                            weight=dataset_config.get("weight", 1.0),
                        )
                    )
            else:
                raise ValueError("'datasets' must be a list of dataset configurations")

            self.dataset_mixture = DatasetMixtureConfig(
                datasets=datasets_list,
                seed=self.dataset_mixture.get("seed", 0),
                test_split_size=self.dataset_mixture.get("test_split_size", None),
            )

            # Check that column names are consistent across all dataset configs
            columns_sets = [set(dataset.columns) for dataset in datasets_list if dataset.columns is not None]
            if columns_sets:
                first_columns = columns_sets[0]
                if not all(columns == first_columns for columns in columns_sets):
                    raise ValueError(
                        "Column names must be consistent across all dataset configurations in a mixture. "
                        f"Found different column sets: {[list(cols) for cols in columns_sets]}"
                    )


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})


@dataclass
class DPOConfig(trl.DPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    listwise: bool = field(default=False, metadata={"help": "Enable listwise DPO training."})
    listwise_beta: Optional[float] = field(
        default=None,
        metadata={"help": "Optional beta override for listwise PL loss. Falls back to DPO beta when unset."},
    )


@dataclass
class ORPOConfig(trl.ORPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
