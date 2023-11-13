import pickle
from mingpt.dataset import PileDataset
from mingpt.trainer import Trainer
from mingpt.model import GPT
from mingpt.utils import CfgNode as CN
from transformers import GPT2Tokenizer
import torch

def generate(model, prompt='', num_samples=10, steps=20, do_sample=True):
    device = 'cuda'
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if prompt == '': 
        # to create unconditional samples...
        # huggingface/transformers tokenizer special cases these strings
        prompt = '<|endoftext|>'
    encoded_input = tokenizer(prompt, return_tensors='pt').to(device)
    x = encoded_input['input_ids']

    # we'll process all desired num_samples in a batch, so expand out the batch dim
    x = x.expand(num_samples, -1)

    # forward the model `steps` times to get samples, in a batch
    y = model.generate(x, max_new_tokens=steps, do_sample=do_sample, top_k=40)

    for i in range(num_samples):
        out = tokenizer.decode(y[i].cpu().squeeze())
        print('-'*80)
        print(out)

def main():
    with open('dataset.pkl', "rb") as dataset_file:
        data = pickle.load(dataset_file)
    print(type(data))
    dataset = PileDataset(data)


    gpt_config = GPT.get_default_config()

    gpt_config.vocab_size = dataset.vocab_size
    gpt_config.block_size = 1024
    gpt_config.model_type = 'gpt-nano'

    model = GPT(gpt_config)
    model.train()

    trainer_config = Trainer.get_default_config()
    trainer_config.max_iters = 500

    my_trainer = Trainer(trainer_config, model, dataset)

    my_trainer.run()

    model.eval()
    with torch.no_grad():
        generate(model, "Davis Forster is a guy who ", num_samples=1, steps=30)



if __name__ == "__main__":
    main()