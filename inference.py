from mingpt.model import GPT
from mingpt.dataset import PileDataset
from transformers import GPT2Tokenizer
import os
import torch


def generate(model, prompt='', num_samples=10, steps=20, do_sample=True):
    device = 'cuda'
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if prompt == '': 
        # to create unconditional samples...
        # huggingface/transformers tokenizer special cases these strings
        prompt = '<|endoftext|>'
    tokenizer.pad_token = tokenizer.eos_token
    x = tokenizer.encode(prompt, return_tensors='pt').to(device)

    # we'll process all desired num_samples in a batch, so expand out the batch dim
    x = x.expand(num_samples, -1)

    # forward the model `steps` times to get samples, in a batch
    y = model.generate(x, max_new_tokens=steps, do_sample=do_sample, top_k=40)

    for i in range(num_samples):
        out = tokenizer.decode(y[i].cpu().squeeze())
        print('-'*80)
        print(out)

def main():

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    gpt_config = GPT.get_default_config()

    gpt_config.vocab_size = tokenizer.vocab_size
    gpt_config.block_size = 1024
    gpt_config.model_type = 'gpt2'

    # check for checkpoints
    checkpoints = os.listdir('./checkpoints')
    # checkpoints.sort()
    checkpoint = 'checkpoints/' + \
        checkpoints[-1] if checkpoints else None

    checkpoint = torch.load(checkpoint)
    print(checkpoint.keys())

    model = GPT(gpt_config)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)
    print("running on device", device)

    if checkpoint != None:
            model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    prompt = 'President Biden said this in a speech on Wednesday:'

    print(generate(model, prompt, steps=200))

if __name__ == "__main__":
    main()