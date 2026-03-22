import argparse
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from collections import defaultdict
from functools import reduce

PROMPT_TOKEN = '<|prompter|>'
ASSISTANT_TOKEN = '<|assistant|>'
EOS_TOKEN = '<|endoftext|>'

def robust_load_dataset(dataset_path, split):
    pref_dataset = load_dataset(dataset_path, split=split)
    remove_columns = ['output', 'text', 'alpaca_text', 'y_ref', 'y_1', 'y_2', 'y_w', 'y_w_alpaca', 'y_l', 'y_l_alpaca', 'y_w_score', 'y_l_score', 'score_diff', 'prompt', 'alpaca_prompt']

    def process_dataset(batch):
        new_batch = {}
        new_batch['query'] = batch['prompt']
        new_batch['text_w'] =  batch['y_w'] 
        new_batch['text_l'] = batch['y_l']
        new_batch['response_w'] = [x.split(ASSISTANT_TOKEN)[-1] for x in batch['y_w']]
        new_batch['response_l'] = [x.split(ASSISTANT_TOKEN)[-1] for x in batch['y_l']]
        
        shapes = {}
        for k, v in new_batch.items():
            shapes[k] = len(v)
        if reduce(lambda x,y: x if x==y else -1, list(shapes.values())) == -1:
            assert False, f"Shapes of all columns must be equal, but got {shapes}, {list(shapes.values())}"
        return new_batch
    
    pref_dataset = pref_dataset.map(
        process_dataset,
        batched=1,
        num_proc=1,
        # remove_columns=remove_columns,
    )
    return pref_dataset

def get_response_logprob(model, tokenizer, query, response, device):
    query_ids = tokenizer(query, return_tensors="pt", add_special_tokens=False).input_ids[0]
    response_ids = tokenizer(response, return_tensors="pt", add_special_tokens=False).input_ids[0]
    input_ids = torch.cat([query_ids, response_ids], dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    response_start = query_ids.shape[0]
    log_probs = []
    for i in range(response_start, input_ids.shape[1]):
        token_id = input_ids[0, i]
        prev_logits = logits[0, i - 1]
        log_prob = torch.log_softmax(prev_logits, dim=-1)[token_id]
        log_probs.append(log_prob.item())
    return sum(log_probs) / (len(log_probs) + 1e-8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--field_query", type=str, default="test")
    parser.add_argument("--field_w", type=str, default="response_w")
    parser.add_argument("--field_l", type=str, default="response_l")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
    model.eval()

    dataset = robust_load_dataset(args.dataset_path, args.split)
    correct = 0
    total = 0

    for example in tqdm(dataset, desc=f"Evaluating split '{args.split}'"):
        print(example.keys())
        query = example[args.field_query]
        response_w = example[args.field_w]
        response_l = example[args.field_l]

        logprob_w = get_response_logprob(model, tokenizer, query, response_w, device)
        logprob_l = get_response_logprob(model, tokenizer, query, response_l, device)
        if logprob_w > logprob_l:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"Accuracy on split '{args.split}': {accuracy:.4f} ({correct}/{total})")

if __name__ == "__main__":
    main()


# import argparse
# import torch
# from datasets import load_dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from tqdm import tqdm

# def get_response_logprob(model, tokenizer, query, response, device):
#     query_ids = tokenizer(query, return_tensors="pt", add_special_tokens=False).input_ids[0]
#     response_ids = tokenizer(response, return_tensors="pt", add_special_tokens=False).input_ids[0]
#     input_ids = torch.cat([query_ids, response_ids], dim=0).unsqueeze(0).to(device)

#     with torch.no_grad():
#         outputs = model(input_ids)
#         logits = outputs.logits

#     response_start = query_ids.shape[0]
#     log_probs = []
#     for i in range(response_start, input_ids.shape[1]):
#         token_id = input_ids[0, i]
#         prev_logits = logits[0, i - 1]
#         log_prob = torch.log_softmax(prev_logits, dim=-1)[token_id]
#         log_probs.append(log_prob.item())
#     return sum(log_probs) / (len(log_probs) + 1e-8)

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_path", type=str, required=True)
#     parser.add_argument("--dataset_path", type=str, required=True)
#     parser.add_argument("--split", type=str, default="test")
#     parser.add_argument("--field_query", type=str, default="query")
#     parser.add_argument("--field_w", type=str, default="response_w")
#     parser.add_argument("--field_l", type=str, default="response_l")
#     args = parser.parse_args()

#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
#     tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
#     model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
#     model.eval()

#     dataset = load_dataset(args.dataset_path, split=args.split)
#     correct = 0
#     total = 0

#     for example in tqdm(dataset, desc="Evaluating"):
#         query = example[args.field_query]
#         response_w = example[args.field_w]
#         response_l = example[args.field_l]

#         logprob_w = get_response_logprob(model, tokenizer, query, response_w, device)
#         logprob_l = get_response_logprob(model, tokenizer, query, response_l, device)
#         if logprob_w > logprob_l:
#             correct += 1
#         total += 1

#     accuracy = correct / total if total > 0 else 0.0
#     print(f"Test accuracy: {accuracy:.4f} ({correct}/{total})")

# if __name__ == "__main__":
#     main()