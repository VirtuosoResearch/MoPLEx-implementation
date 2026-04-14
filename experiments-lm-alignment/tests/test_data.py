# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import unittest

import pytest
from datasets import Dataset

from alignment import ScriptArguments, get_dataset
from alignment.data import _to_listwise_dataset


class GetDatasetTest(unittest.TestCase):
    """Test the new get_dataset() method with dataset_mixture API"""

    def test_loading_dataset_mixture(self):
        dataset_mixture = {
            "datasets": [
                {"id": "HuggingFaceH4/testing_alpaca_small", "columns": ["prompt", "completion"], "weight": 0.5},
                {
                    "id": "HuggingFaceH4/testing_self_instruct_small",
                    "columns": ["prompt", "completion"],
                    "weight": 0.3,
                },
                {"id": "HuggingFaceH4/testing_codealpaca_small", "columns": ["prompt", "completion"], "weight": 0.2},
            ],
            "seed": 42,
            "test_split_size": 0.1,
        }
        args = ScriptArguments(dataset_mixture=dataset_mixture)
        datasets = get_dataset(args)
        # With weights 0.5, 0.3, 0.2 on 100-sample datasets and test_split_size=0.1
        # Total samples = 50 + 30 + 20 = 100
        # Train: 90, Test: 10
        self.assertEqual(len(datasets["train"]), 90)
        self.assertEqual(len(datasets["test"]), 10)

    def test_loading_dataset_mixture_no_test_split(self):
        dataset_mixture = {
            "datasets": [
                {"id": "HuggingFaceH4/testing_alpaca_small", "columns": ["prompt", "completion"], "weight": 0.5},
                {
                    "id": "HuggingFaceH4/testing_self_instruct_small",
                    "columns": ["prompt", "completion"],
                    "weight": 0.3,
                },
                {"id": "HuggingFaceH4/testing_codealpaca_small", "columns": ["prompt", "completion"], "weight": 0.2},
            ],
            "seed": 42,
        }
        args = ScriptArguments(dataset_mixture=dataset_mixture)
        datasets = get_dataset(args)
        # Total samples = 50 + 30 + 20 = 100 (all in train split)
        self.assertEqual(len(datasets["train"]), 100)
        self.assertNotIn("test", datasets)

    def test_loading_with_unit_weights(self):
        dataset_mixture = {
            "datasets": [
                {"id": "HuggingFaceH4/testing_alpaca_small", "columns": ["prompt", "completion"], "weight": 1.0},
                {
                    "id": "HuggingFaceH4/testing_self_instruct_small",
                    "columns": ["prompt", "completion"],
                    "weight": 1.0,
                },
                {"id": "HuggingFaceH4/testing_codealpaca_small", "columns": ["prompt", "completion"], "weight": 1.0},
            ],
            "seed": 42,
            "test_split_size": 0.1,
        }
        args = ScriptArguments(dataset_mixture=dataset_mixture)
        datasets = get_dataset(args)
        # Total samples = 100 + 100 + 100 = 300
        # Train: 270, Test: 30
        self.assertEqual(len(datasets["train"]), 270)
        self.assertEqual(len(datasets["test"]), 30)

    def test_loading_with_fractional_weights(self):
        dataset_mixture = {
            "datasets": [
                {"id": "HuggingFaceH4/testing_alpaca_small", "columns": ["prompt", "completion"], "weight": 0.7},
                {
                    "id": "HuggingFaceH4/testing_self_instruct_small",
                    "columns": ["prompt", "completion"],
                    "weight": 0.4,
                },
            ],
            "seed": 42,
            "test_split_size": 0.1,
        }
        args = ScriptArguments(dataset_mixture=dataset_mixture)
        datasets = get_dataset(args)
        # Total samples = 70 + 40 = 110
        # Train: 99, Test: 11
        self.assertEqual(len(datasets["train"]), 99)
        self.assertEqual(len(datasets["test"]), 11)

    def test_loading_fails_with_invalid_dataset_mixture(self):
        # Test that invalid dataset_mixture configuration raises error
        with pytest.raises(ValueError, match=r"'datasets' must be a list"):
            _ = ScriptArguments(dataset_mixture={"datasets": "invalid"})

        with pytest.raises(ValueError, match=r"dataset_mixture must be a dictionary"):
            _ = ScriptArguments(dataset_mixture="invalid")

    def test_loading_single_dataset(self):
        # Test loading a single dataset using dataset_name instead of dataset_mixture
        args = ScriptArguments(dataset_name="HuggingFaceH4/testing_alpaca_small")
        datasets = get_dataset(args)
        # Single dataset should have both train and test splits
        self.assertIn("train", datasets)
        self.assertEqual(len(datasets["train"]), 100)
        self.assertIn("test", datasets)
        self.assertEqual(len(datasets["test"]), 100)

    def test_listwise_conversion_sorts_by_dimension(self):
        data = Dataset.from_list(
            [
                {
                    "prompt": "p1",
                    "completions": [
                        {"response": "A", "scores": {"helpfulness": 1.0}},
                        {"response": "B", "scores": {"helpfulness": 3.0}},
                        {"response": "C", "scores": {"helpfulness": 2.0}},
                        {"response": "D", "scores": {"helpfulness": 0.5}},
                    ],
                }
            ]
        )
        args = ScriptArguments(
            dataset_name="dummy",
            dataset_format="listwise",
            preference_dimension="helpfulness",
            listwise_num_responses=4,
        )
        converted = _to_listwise_dataset(data, args)
        self.assertEqual(len(converted), 1)
        first = converted[0]
        self.assertEqual(first["responses"], ["B", "C", "A", "D"])

    def test_listwise_requires_dimension(self):
        with pytest.raises(ValueError, match=r"preference_dimension"):
            _ = ScriptArguments(dataset_name="dummy", dataset_format="listwise")

    def test_listwise_random_subsample_is_seeded(self):
        data = Dataset.from_list(
            [
                {
                    "prompt": "p1",
                    "completions": [
                        {"response": "A", "scores": {"helpfulness": 6.0}},
                        {"response": "B", "scores": {"helpfulness": 5.0}},
                        {"response": "C", "scores": {"helpfulness": 4.0}},
                        {"response": "D", "scores": {"helpfulness": 3.0}},
                        {"response": "E", "scores": {"helpfulness": 2.0}},
                        {"response": "F", "scores": {"helpfulness": 1.0}},
                    ],
                }
            ]
        )

        args_seed_1a = ScriptArguments(
            dataset_name="dummy",
            dataset_format="listwise",
            preference_dimension="helpfulness",
            listwise_num_responses=3,
            listwise_seed=123,
        )
        args_seed_1b = ScriptArguments(
            dataset_name="dummy",
            dataset_format="listwise",
            preference_dimension="helpfulness",
            listwise_num_responses=3,
            listwise_seed=123,
        )
        args_seed_2 = ScriptArguments(
            dataset_name="dummy",
            dataset_format="listwise",
            preference_dimension="helpfulness",
            listwise_num_responses=3,
            listwise_seed=999,
        )

        converted_1a = _to_listwise_dataset(data, args_seed_1a)
        converted_1b = _to_listwise_dataset(data, args_seed_1b)
        converted_2 = _to_listwise_dataset(data, args_seed_2)

        responses_1a = converted_1a[0]["responses"]
        responses_1b = converted_1b[0]["responses"]
        responses_2 = converted_2[0]["responses"]

        self.assertEqual(responses_1a, responses_1b)
        self.assertNotEqual(responses_1a, responses_2)
