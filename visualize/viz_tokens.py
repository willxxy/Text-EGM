from data_loader import EGMDataset2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.cm as cm
import argparse 

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--data', type=str, default = None, help = 'Please choose the dataset you want to visualize')
    parser.add_argument('--mask', type=float, default = None, help = 'Please choose the percentage of masking')
    parser.add_argument('--ratio', type=int, default = None, help = 'Please choose the ratio')
    parser.add_argument('--vocab_size', type=int, default = None, help = 'Please choose the vocab size')
    parser.add_argument('--viz_tokens', action='store_true', help = 'Please choose whether to visualize tokens or not')
    return parser.parse_args()

def visualize_tokens(dataset, num_samples=1, args = None, vocab_size = None):
    count = 0 
    for i in tqdm(dataset):
        masked_sample, tokenized_sample, original, mask_indices = i
        mask_indice_list = list(mask_indices.squeeze().numpy())
        mask_indice_list = [int(m) for m in mask_indice_list]
        mask_indice_list.sort()
        masked_sample = masked_sample.squeeze().numpy()
        tokenized_sample = tokenized_sample.squeeze().numpy()
        original_sequence = original.squeeze().numpy()
        
        global_min = min(np.min(masked_sample[masked_sample != 0]), np.min(tokenized_sample[tokenized_sample != 0]))
        global_max = max(np.max(masked_sample), np.max(tokenized_sample))
        
        unique_tokens = np.unique(tokenized_sample)
        colors = cm.rainbow(np.linspace(0, 1, len(unique_tokens)))
        
        plt.figure(figsize=(20, 5))

        plt.subplot(1, 3, 1)
        plt.title(f"{int(args.mask * 100)}% Masked Tokens")
        for token, color in zip(unique_tokens, colors):
            if token == 0:
                pass
            else:
                indices = np.where(masked_sample == token)
                plt.scatter(indices, masked_sample[indices], color=color, s=10)
        plt.xlabel('Time (ms)')
        plt.ylabel('Token Value')
        plt.ylim(global_min, global_max)
        
        plt.subplot(1, 3, 2)
        plt.title(f"Tokenized Sample")
        for token, color in zip(unique_tokens, colors):
            indices = np.where(tokenized_sample == token)
            plt.scatter(indices, tokenized_sample[indices], color=color, s=10)
        plt.xlabel('Time (ms)')
        plt.ylabel('Token Value')
        plt.ylim(global_min, global_max)
        
        
        plt.subplot(1, 3, 3)
        plt.title(f"Original Sequence")
        plt.plot(np.linspace(0, len(original_sequence), len(original_sequence)), original_sequence)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        plt.tight_layout()
        plt.savefig(f'../pngs/tokens_viz_{args.ratio}_{vocab_size}.png')
        plt.show()

        count += 1
        if count >= num_samples:
            break


if __name__ == '__main__':
    args = get_args()
    val = np.load(args.data, allow_pickle=True).item()
    dataset = EGMDataset2(val, vocab_size = args.vocab_size, args = args)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)   
    visualize_tokens(dataloader, args=args, vocab_size = args.vocab_size)