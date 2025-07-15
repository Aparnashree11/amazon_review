import os
import requests

def download_file(presigned_url, local_path):
    if os.path.exists(local_path):
        print(f"{local_path} already exists. Skipping.")
        return

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    print(f"Downloading {local_path}...")
    response = requests.get(presigned_url)
    if response.status_code == 200:
        with open(local_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {local_path}")
    else:
        raise Exception(f"Failed to download {local_path}. Status: {response.status_code}")

# Fetch presigned URLs from env variables
presigned_urls = {
    'models/distilbert-final/model.safetensors': os.getenv('SAFE_TENSORS_URL'),
    'models/distilbert-final/config.json': os.getenv('CONFIG_JSON_URL'),
    'models/distilbert-final/tokenizer.json': os.getenv('TOKENIZER_JSON_URL'),
    'models/distilbert-final/tokenizer_config.json': os.getenv('TOKENIZER_CONFIG_URL'),
    'models/distilbert-final/training_args.bin': os.getenv('TRAIN_ARGS_URL'),
    'models/distilbert-final/special_tokens_map.json': os.getenv('SPL_TOKENS_URL'),
    'models/distilbert-final/vocab.txt': os.getenv('VOCAB_URL')
}

for local_path, url in presigned_urls.items():
    if url:
        download_file(url, local_path)
    else:
        raise Exception(f"Missing presigned URL for {local_path}")
