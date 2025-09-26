from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

import llm_utils
import opt_utils

import argparse
parser = argparse.ArgumentParser()
# the following demo is sutiable for gpt2 (all sizes)
# other models (such as meta-llama/Llama-2-7b-chat-hf) may require different settings (such as tokenization and plots configurations)
parser.add_argument('--model_name', type=str, default='gpt2', help='model name (only models that are supported by llm_utils)')
parser.add_argument('--model_args', type=str, default='')
parser.add_argument('--top_k', type=int, default=5, help='top k tokens to extract using logit lens. more tokens will be slower')
parser.add_argument('--disable_pad_token', action='store_true')
parser.add_argument('--norm_ll', action='store_true', help='if to use normalize logit lens')
parser.add_argument('--device', type=str, default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

args, unknown = parser.parse_known_args()
print('unknown args:', unknown)
print('args:', args)

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
if not args.disable_pad_token:
    print(f'adding pad token: {tokenizer.eos_token}')
    tokenizer.pad_token = tokenizer.eos_token

try:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"  # not blocking, just to prevent warnings and faster tokenization
except:
    pass

device = torch.device(args.device)
print(f'Using device: {device} [cuda available? => {torch.cuda.is_available()}, cuda version: {torch.version.cuda}, args.device = "{args.device}"]')

model_extra_args = {}
for arg in args.model_args.split(','):
    if arg == '':
        continue
    k, v = arg.split('=')
    model_extra_args[k] = v
print(f'model_extra_args: {model_extra_args}')

model = AutoModelForCausalLM.from_pretrained(args.model_name).eval().requires_grad_(False).to(device)
model_aux = llm_utils.model_extra(model=model, device=device)
config = model_aux.config  # should be the same as auto_model_to_config(args.model_name)
del model # deleting the model to save memory

n_embd = model_aux.n_embd
n_head = model_aux.n_head
head_size = model_aux.head_size
n_layer = model_aux.n_layer

params_names_filter = opt_utils.only_mlp_filter  # mainly to save memory

subject = 'ACL'
prompt_tmp = "This year's {} is located in"  # not mandatory to have the prompt in a format that includes "{}" but it is the format used in CounterFact dataset
target_true = ' Vienna'  # not really used (see final_anwer_ids below)
target_new = ' Bankok'


prefix_prompt, postfix_prompt = prompt_tmp.split('{}')
prompt = prompt_tmp.format(subject)
print(f"prompt: {prompt}")
prompt_encoded = tokenizer(prompt, return_tensors="pt").to(device)

if target_new[0] != ' ':
    target_new = ' ' + target_new

prompt_len = prompt_encoded['input_ids'].shape[1]
prompt_list = [tokenizer.decode(x) for x in prompt_encoded['input_ids'][0]]

print(f"prompt_len : {prompt_len} \n prompt_list : {prompt_list}")

if prefix_prompt == '':
    start_index_of_subject = 0
    subject_len = tokenizer.encode(subject, return_tensors='pt').shape[1]
else:
    start_index_of_subject = tokenizer.encode(prefix_prompt.rstrip(' '), return_tensors='pt').shape[1]
    subject_len = tokenizer.encode(' ' + subject, return_tensors='pt').shape[1]

print(f'prompt "{prompt}", change: "{target_true}" --> "{target_new}", prompt_len {prompt_len}')
# print(f"tokenizer.encode(prefix_prompt.rstrip(' '), return_tensors='pt'): {tokenizer.encode('who are you', return_tensors='pt', padding='max_length', max_length=20)}")


res = opt_utils.get_nll_opt_model(prompt, target_new, model_name=args.model_name,
        tokenizer=tokenizer, opt='SGD', device=device, lr=1.0,
        params_names_filter=params_names_filter,
        wrapp_forward_pass_config='AUTO',
        wrapp_backward_pass_config='AUTO',
        loops=10)

model_sgd = res['model']  # updated model
hs_collector = res['hs_collector']  # hidden states
grad_collector = res['grad_collector']  # VJPs

# what did the model actually predict (before any optimization)
model_final_output = hs_collector[n_layer-1][config.layer_format]['output'][-1]
answer = model_aux.hs_to_token_top_k(model_final_output, k=1)['top_k'][0]
ans2, ans3 = model_aux.hs_to_token_top_k(model_final_output, k=3)['top_k'][1], model_aux.hs_to_token_top_k(model_final_output, k=3)['top_k'][2]
final_anwer_ids = tokenizer.encode(answer, return_tensors='pt')[0][-1:].item()
print(f'\nmodel answer: "{answer}" (id {final_anwer_ids})')
print(f"\nans2: {ans2}, ans3: {ans3}")

target_new_ids = tokenizer.encode(target_new, return_tensors='pt')[0][-1:].item()  # the "-1:" is used if the output is more than one token long. in particular, in Llama there is a special token for " " (space)
print(f'\ntarget_new: "{target_new}" (id {target_new_ids})')
target_new_tmp = tokenizer.decode(target_new_ids)