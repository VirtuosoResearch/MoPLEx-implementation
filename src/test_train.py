import unittest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dpo_cache_utils import get_log_prob, calculate_DPO_loss  # 替换成你实际模块名

class TestDPOFunctions(unittest.TestCase):

    def setUp(self):
        # 模拟输入
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelForCausalLM.from_pretrained("gpt2").to(self.device)

        # 准备dummy数据
        self.batch_size = 2
        self.seq_len = 5
        self.vocab_size = self.model.config.vocab_size

        # 模拟logits
        self.logits = torch.randn(self.batch_size, self.seq_len, self.vocab_size, device=self.device)
        self.labels = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len), device=self.device)
        self.prompt_lengths = torch.tensor([2, 3], device=self.device)

    def test_get_log_prob(self):
        log_probs = get_log_prob(self.logits, self.labels, self.prompt_lengths)
        self.assertEqual(log_probs.shape[0], self.batch_size)  # 应该返回 batch_size 长度
        self.assertTrue(torch.is_tensor(log_probs))
    
    def test_calculate_DPO_loss(self):
        model_preferred_logprob = torch.tensor([0.5, 1.0])
        model_dispreferred_logprob = torch.tensor([0.3, 0.4])
        ref_preferred_logprob = torch.tensor([0.1, 0.6])
        ref_dispreferred_logprob = torch.tensor([0.2, 0.5])

        loss, pref_rel, dispref_rel, acc, margin = calculate_DPO_loss(
            model_preferred_logprob, model_dispreferred_logprob,
            ref_preferred_logprob, ref_dispreferred_logprob,
            beta=0.5
        )

        self.assertTrue(loss.item() > 0)
        self.assertTrue(isinstance(acc.item(), float))
        self.assertTrue(isinstance(margin.item(), float))


if __name__ == "__main__":
    unittest.main()

