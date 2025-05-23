# Copyright 2024 Bytedance Ltd. and/or its affiliates
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


import torch
from transformers import PreTrainedTokenizer

from ... import DataProto
from ...utils.reward_score import math_compute_score, r1v_compute_score, tvg_compute_score, v1_compute_score


class CustomRewardManager:
    def __init__(self, tokenizer: PreTrainedTokenizer, num_examine: int, compute_score: str):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        if compute_score == "math":
            self.compute_score = math_compute_score
        elif compute_score == "r1v":
            self.compute_score = r1v_compute_score
        elif compute_score == 'tgrpo':
            self.compute_score = 'tgrpo'
        else:
            raise NotImplementedError()


    def get_score_and_details(self, data_item, key_suffix=""):
        # Build keys based on suffix
        suffix = f"_{key_suffix}" if key_suffix else ""
        prompts_key = f"prompts{suffix}"
        responses_key = f"responses{suffix}"
        mask_key = f"attention_mask{suffix}"

        prompt_ids = data_item.batch[prompts_key]
        prompt_length = prompt_ids.shape[-1]
        # For non-shuffle, reverse slice the prompt tokens; for shuffle, use the beginning tokens.
        valid_prompt_length = data_item.batch[mask_key][:prompt_length].sum()
        valid_prompt_ids = (
            prompt_ids[-valid_prompt_length:] if key_suffix == "" else prompt_ids[:valid_prompt_length]
        )

        response_ids = data_item.batch[responses_key]
        # Compute response length using the attention mask for tokens after the prompt
        valid_response_length = data_item.batch[mask_key][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # Decode tokens into strings
        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

        # Get ground truth and compute score based on problem type.
        ground_truth = data_item.non_tensor_batch["ground_truth"]
        problem_type = data_item.non_tensor_batch.get("problem_type", "")
        if problem_type == 'tvg':
            video_length = data_item.non_tensor_batch["video_length"]
            score = tvg_compute_score(response_str, ground_truth, video_length)
        elif problem_type == 'v1':
            score = v1_compute_score(response_str, ground_truth)
        else:
            score = self.compute_score(response_str, ground_truth)
        return score, valid_response_length, prompt_str, response_str


    def __call__(self, data: DataProto) -> torch.Tensor:
        num_items = len(data)

        if isinstance(self.compute_score, str) and self.compute_score == 'tgrpo':
            # Initialize two reward tensors for the normal and shuffled outputs
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            reward_tensor_shuffle = torch.zeros_like(data.batch["responses_shuffle"], dtype=torch.float32)
            already_print = 0
            for i in range(num_items):
                data_item = data[i]  # DataProtoItem

                # Compute reward on the normal branch.
                score, valid_resp_len, prompt_str, response_str = self.get_score_and_details(data_item)
                reward_tensor[i, valid_resp_len - 1] = score

                # Compute reward for the shuffled branch.
                score_shuffle, valid_resp_len_shuffle, prompt_str_shuffle, response_str_shuffle = self.get_score_and_details(
                    data_item, key_suffix="shuffle"
                )
                reward_tensor_shuffle[i, valid_resp_len_shuffle - 1] = score_shuffle

                # Optionally print the first few items for examination.
                if already_print < self.num_examine:
                    already_print += 1
                    print("[prompt]", prompt_str)
                    print("[response]", response_str)
                    print("[ground_truth]", data_item.non_tensor_batch["ground_truth"])
                    print("[score]", score)

            return reward_tensor, reward_tensor_shuffle

        else:
            # Only compute the non-shuffled reward tensor.
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            already_print = 0
            for i in range(num_items):
                data_item = data[i]  # DataProtoItem
                score, valid_resp_len, prompt_str, response_str = self.get_score_and_details(data_item)
                reward_tensor[i, valid_resp_len - 1] = score

                if already_print < self.num_examine:
                    already_print += 1
                    print("[prompt]", prompt_str)
                    print("[response]", response_str)
                    print("[ground_truth]", data_item.non_tensor_batch["ground_truth"])
                    print("[score]", score)

            return reward_tensor
