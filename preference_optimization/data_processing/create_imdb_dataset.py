# =====================================================================================
#                       Import Necessary Libraries
# =====================================================================================
import torch
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import random
import json

# =====================================================================================
#                               Configuration
# =====================================================================================
# You can adjust this batch size depending on your GPU memory.
BATCH_SIZE = 8 # Qwen2-0.5B is larger than GPT-2, so you might need a smaller batch size.
# Using a smaller sample for faster demonstration.
NUM_SAMPLES = 500
# Define the model to be used
MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"

# =====================================================================================
# 1. Initialize Models and Tokenizers
# =====================================================================================

print(f"Loading model and tokenizer for: {MODEL_ID}...")

# Load a sentiment analysis pipeline to score the sentiment of generated text.
sentiment_pipeline = pipeline(
    "sentiment-analysis", 
    model="distilbert-base-uncased-finetuned-sst-2-english", 
    device=0 if torch.cuda.is_available() else -1
)

# Load the Qwen2 model and tokenizer.
# trust_remote_code=True is required for some models.
generation_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
generation_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto", # Use bfloat16 for better performance if available
    device_map="auto",
    trust_remote_code=True
)

# Set pad token for batching. Qwen2 uses eos_token as pad_token.
if generation_tokenizer.pad_token is None:
    generation_tokenizer.pad_token = generation_tokenizer.eos_token
# Set padding side to 'left' for decoder-only models.
generation_tokenizer.padding_side = 'left'


# =====================================================================================
# 2. Load the IMDB Dataset
# =====================================================================================

print("Loading the IMDB dataset...")
imdb_dataset = load_dataset("imdb", split="test").shuffle(seed=42)

# =====================================================================================
# 3. Generate Response Pairs and Construct Preference Data (with Batching)
# =====================================================================================

def generate_responses_batch(prompts, num_responses_per_prompt=2):
    """Generates responses for a batch of prompts using the Qwen2 chat template."""
    
    # Apply the official chat template for Qwen2-Instruct.
    # This formats the input correctly for the model.
    messages_batch = [[{"role": "user", "content": p}] for p in prompts]
    templated_prompts = [generation_tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_batch]

    # The tokenizer can process a list of templated prompts at once.
    inputs = generation_tokenizer(
        templated_prompts, 
        return_tensors="pt", 
        max_length=128, 
        truncation=True, 
        padding=True
    )
    
    # Move input tensors to the GPU.
    inputs = {k: v.to(generation_model.device) for k, v in inputs.items()}

    # Generate responses for the entire batch.
    outputs = generation_model.generate(
        **inputs,
        max_new_tokens=50,
        num_return_sequences=num_responses_per_prompt,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=generation_tokenizer.eos_token_id
    )
    
    # Decode the generated token IDs back into text.
    all_responses_text = generation_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Reshape the output and clean it by removing the templated prompt.
    batched_responses = []
    for i in range(len(prompts)):
        # For each prompt, we get num_responses_per_prompt responses
        start_index = i * num_responses_per_prompt
        end_index = (i + 1) * num_responses_per_prompt
        
        # Clean the response by finding the start of the assistant's part and taking the text after it.
        # The original user content is used for cleaning.
        user_content = prompts[i]
        prompt_responses = [
            # The generated text contains the user prompt, so we remove it.
            # A simple way is to find the user content and remove everything before and including it.
            resp.split(user_content, 1)[-1].strip()
            for resp in all_responses_text[start_index:end_index]
        ]
        batched_responses.append(prompt_responses)
        
    return batched_responses

# Lists to store the preference pairs for each group.
sentiment_preferences = []
conciseness_preferences = []

print(f"Generating responses with a batch size of {BATCH_SIZE}...")
# Process the dataset in batches.
for i in tqdm(range(0, len(imdb_dataset), BATCH_SIZE)):
    batch_prompts = imdb_dataset[i : i + BATCH_SIZE]['text']
    
    try:
        batch_generated_responses = generate_responses_batch(batch_prompts, num_responses_per_prompt=2)

        for prompt, responses in zip(batch_prompts, batch_generated_responses):
            if len(responses) < 2 or not all(responses) or responses[0] == responses[1]:
                continue
            
            response1, response2 = responses[0], responses[1]

            # a. Construct "Positive Sentiment" Preferences (Majority Group)
            sentiments = sentiment_pipeline([response1, response2])
            score1 = sentiments[0]['score'] if sentiments[0]['label'] == 'POSITIVE' else 1 - sentiments[0]['score']
            score2 = sentiments[1]['score'] if sentiments[1]['label'] == 'POSITIVE' else 1 - sentiments[1]['score']

            if score1 > score2:
                sentiment_preferences.append({
                    "prompt": prompt, "chosen": response1, "rejected": response2, 
                    "preference_source": "sentiment"
                })
            else:
                sentiment_preferences.append({
                    "prompt": prompt, "chosen": response2, "rejected": response1,
                    "preference_source": "sentiment"
                })

            # b. Construct "Conciseness" Preferences (Minority Group)
            if len(response1) < len(response2):
                conciseness_preferences.append({
                    "prompt": prompt, "chosen": response1, "rejected": response2,
                    "preference_source": "conciseness"
                })
            else:
                conciseness_preferences.append({
                    "prompt": prompt, "chosen": response2, "rejected": response1,
                    "preference_source": "conciseness"
                })

    except Exception as e:
        print(f"An error occurred during batch processing: {e}")
        continue


# =====================================================================================
# 4. Mix Datasets According to the 80/20 Ratio and Save
# =====================================================================================

print("Mixing the datasets...")

# Determine the number of samples from each group.
total_samples = min(len(sentiment_preferences), len(conciseness_preferences))
majority_count = int(total_samples * 0.8)
minority_count = total_samples - majority_count

# Randomly sample from each preference list.
final_preferences = (
    random.sample(sentiment_preferences, majority_count) +
    random.sample(conciseness_preferences, minority_count)
)

# Shuffle the final dataset.
random.shuffle(final_preferences)

print(f"Dataset creation complete!")
print(f"Total samples: {len(final_preferences)}")
print(f"Majority group (sentiment preference) samples: {majority_count}")
print(f"Minority group (conciseness preference) samples: {minority_count}")

# Save the final dataset to a JSON file.
output_filename = "imdb_preference_dataset_qwen2.json"
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(final_preferences, f, ensure_ascii=False, indent=4)

print(f"Dataset saved to: {output_filename}")

# Print a sample from the final dataset for verification.
if final_preferences:
    print("\nSample from the final dataset:")
    print(json.dumps(final_preferences[0], indent=2, ensure_ascii=False))