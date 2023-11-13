import pickle
from mingpt.dataset import PileDataset
from mingpt.trainer import Trainer
from mingpt.model import GPT
from mingpt.utils import CfgNode as CN
from transformers import GPT2Tokenizer
import torch
from datasets import load_dataset


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
    dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", 'plain_text', cache_dir='datasets')
    # dataset = load_dataset("json", data_files="minipile.jsonl")
    dataset = dataset['train']
    # print(dataset)

    dataset = PileDataset(dataset, max_length=1024)


    gpt_config = GPT.get_default_config()

    gpt_config.vocab_size = dataset.vocab_size
    gpt_config.block_size = 1024
    gpt_config.model_type = 'gpt2'

    model = GPT(gpt_config)
    model.train()

    trainer_config = Trainer.get_default_config()
    trainer_config.max_iters = 50000
    trainer_config.batch_size = 1
    trainer_config.num_workers = 0

    def batch_end_callback(trainer):
        if trainer.iter_num % 1000 == 0 and trainer.iter_num !=0:      
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
        if trainer.iter_num % 10000 == 0 and trainer.iter_num !=0:
            loss = trainer.loss.item()
            iter_num = trainer.iter_num
            filepath = "davis_mingpt"+str(iter_num)+".pt"
            torch.save({
                        'epoch': iter_num,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': trainer.optimizer.state_dict(),
                        'loss': loss,
                        'train_config':trainer_config,
                        'model_config':gpt_config

            }, filepath)
            print("Saving model")

    my_trainer = Trainer(trainer_config, model, dataset)

    my_trainer.set_callback('on_batch_end', batch_end_callback)

    my_trainer.run()

    loss = my_trainer.loss.item()
    iter_num = my_trainer.iter_num
    filepath = "davis_mingpt"+str(iter_num)+".pt"
    torch.save({
                'epoch': iter_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': my_trainer.optimizer.state_dict(),
                'loss': loss,
                'train_config':trainer_config,
                'model_config':gpt_config

    }, filepath)
    print("Saving model")

    model.eval()
    with torch.no_grad():
        generate(model, "Davis Forster is a guy who ", num_samples=10, steps=30)



if __name__ == "__main__":
    main()