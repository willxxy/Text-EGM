import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
import torch
torch.set_num_threads(2)
from transformers import BigBirdTokenizer, BigBirdForMaskedLM, AutoModelForMaskedLM, AutoTokenizer, \
                            LongformerForMaskedLM, LongformerTokenizer
from captum.attr import LayerIntegratedGradients
import argparse

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--checkpoint', type=str, default = None, help = 'Please choose the checkpoint you want to visualize')
    parser.add_argument('--device', type=str, default = None, help = 'Please choose the device')
    parser.add_argument('--model', type=str, default = None, help = 'Please choose the model')
    parser.add_argument('--pre', action='store_true', default = None, help = 'Please choose the device')
    parser.add_argument('--afibmask', action='store_true', default = None, help = 'Please choose the device')
    parser.add_argument('--TA', action='store_true', default = None, help = 'Please choose if Token Addition')
    parser.add_argument('--TS', action='store_true', default = None, help = 'Please choose if Token Substitution')
    parser.add_argument('--LF', action='store_true', default = None, help = 'Please choose if Label Flipping')
    parser.add_argument('--CF', action='store_true', default = None, help = 'Implement Counterfactuals')
    return parser.parse_args()

def plot_attributions(signal, attributions, key, mask, args):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(signal, label='Signal')
    masked_indices = np.where(mask == 0)[0]
    ax.scatter(masked_indices, [signal[i] for i in masked_indices], color='black', zorder=2, label='Masked Tokens', s = 15, alpha=1)
    
    ax2 = ax.twinx()
    # Assuming attributions is a list of attribution scores
    ax2.fill_between(range(len(attributions)), 0, attributions, color='red', alpha=0.3, label='Attribution Score')
    ax2.set_ylim(0, 0.2) 

    
    ax.set_xlabel('Sequence Length', fontsize='large', fontweight='bold')
    ax.set_ylabel('Signal Amplitude', fontsize='large', fontweight='bold')
    ax2.set_ylabel('Attribution Score', fontsize='large', fontweight='bold')

    plt.savefig(f'../runs/checkpoint/{args.checkpoint}/attribution_score_{key}_{args.pre}_{args.afibmask}_{args.TS}_{args.TA}_{args.LF}_{args.CF}.png')


def label_flip(afib_label):
    if afib_label == 0:
        afib_label = 1
    elif afib_label == 1:
        afib_label = 0
    return afib_label

def moving_average(signal, window_size=50):
    return np.convolve(signal, np.ones(window_size), 'same') / window_size
                
def sample_consecutive(signal, sample_size):
    max_start_index = len(signal) - sample_size
    start_index = np.random.randint(0, max_start_index)
    return signal[start_index:start_index + sample_size]  


def forward_func(input_ids, attention_mask, mask_token_index):
    outputs = model(input_ids, attention_mask=attention_mask)
    return outputs.logits[:, mask_token_index, :]

if __name__ == '__main__':
    args = get_args()
    torch.manual_seed(123)
    np.random.seed(123)
    
    norm_test = np.load(f'../data/test_data_by_placement_na_True_True_True_False_False.npy', allow_pickle=True).item()
    afib_test = np.load(f'../data/test_data_by_placement_na_True_True_True_False_True.npy', allow_pickle=True).item()
    
    norm_test.update(afib_test)
    
    test = norm_test
    
    ###
    # test = np.load(f'../data/manip_arrs.npy', allow_pickle=True).item()
    ###
    
    norm_keys_list = list(norm_test.keys())
    afib_keys_list = list(afib_test.keys())
    
    norm_keys_list = norm_keys_list[:50]
    afib_keys_list = afib_keys_list[:50]
    keys_list = norm_keys_list + afib_keys_list
    
    ###
    # keys_list = list(test.keys())
    ###

    device_name = args.device
    device = torch.device(device_name)
    
    custom_tokens = [
        f"signal_{i}" for i in range(250+1)
    ] + [
        f"afib_{i}" for i in range(2)
    ]
    
    if args.TA:
        custom_tokens += [
        f"augsig_{i}" for i in range(250+1)
    ]

    if args.model == 'big':
        tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")
        model = BigBirdForMaskedLM.from_pretrained("google/bigbird-roberta-base")
        model.config.attention_type = 'original_full'
        num_added_tokens = tokenizer.add_tokens(custom_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model_hidden_size = model.config.hidden_size
    
    if args.model =='clin_bird':
        model = AutoModelForMaskedLM.from_pretrained("yikuan8/Clinical-BigBird").to(device)
        model.config.attention_type = 'original_full'
        tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-BigBird")
        tokenizer.add_tokens(custom_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model_hidden_size = model.config.hidden_size
        
    if args.model =='clin_long':
        model = AutoModelForMaskedLM.from_pretrained("yikuan8/Clinical-Longformer").to(device)
        tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
        tokenizer.add_tokens(custom_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model_hidden_size = model.config.hidden_size
    if args.model == 'long':
        model = LongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096").to(device)
        tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
        tokenizer.add_tokens(custom_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model_hidden_size = model.config.hidden_size
        
    if args.pre:
        checkpoint = torch.load(f'../runs/checkpoint/{args.checkpoint}/best_checkpoint.chkpt', map_location = device_name)    
        model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    model.zero_grad()
    lig = LayerIntegratedGradients(forward_func, model.bert.embeddings)
    
    mask_token =  tokenizer.cls_token
    cls_token =  tokenizer.mask_token
    sep_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token
    
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)
    cls_id = tokenizer.convert_tokens_to_ids(cls_token)
    sep_id = tokenizer.convert_tokens_to_ids(sep_token)
    pad_id = tokenizer.convert_tokens_to_ids(pad_token)
    
    
    np_dic = {}
    count2 =0
    
    for i in range(len(keys_list)):
        
        seq = test[keys_list[i]]
        signal = seq[:1000]
        if args.TS and args.CF: 
            signal = moving_average(signal)
        key = keys_list[i]
        afib_label = key[-1]
        
        if args.LF and args.CF:
            afib_label = label_flip(afib_label)
        
        afib_token = f"afib_{int(afib_label)}" 
        
        min_val, max_val = np.min(signal), np.max(signal)
        normalized_signal = (signal - min_val) / (max_val - min_val)
        quantized_signal = np.floor(normalized_signal * 250).astype(int)
        quantized_signal_tokens = [f"signal_{i}" for i in quantized_signal]

        quantized_signal_ids = tokenizer.convert_tokens_to_ids(quantized_signal_tokens)
        
        afib_id = tokenizer.convert_tokens_to_ids([afib_token])
        
        if args.TA:
            quantized_augsignal_tokens = [f"augsig_{i}" for i in quantized_signal]
            sampled_quantized_augsignal_tokens = sample_consecutive(quantized_augsignal_tokens, int(0.25 * len(quantized_signal_ids)))
            sampled_quantized_augsignal_ids = tokenizer.convert_tokens_to_ids(sampled_quantized_augsignal_tokens)
            len_sampled_quantized_augsignal_ids = len(sampled_quantized_augsignal_ids)
            all_tokens  =[cls_id] + quantized_signal_ids + [pad_id] * len_sampled_quantized_augsignal_ids + [sep_id] + afib_id + [sep_id]
            if args.CF:
                all_tokens2  =[cls_id] + quantized_signal_ids + sampled_quantized_augsignal_ids + [sep_id] + afib_id + [sep_id]    
            else:
                all_tokens2  =[cls_id] + quantized_signal_ids + [pad_id] * len_sampled_quantized_augsignal_ids + [sep_id] + afib_id + [sep_id]
            ref_tokens = [cls_id] + [pad_id] * (len(quantized_signal_ids) + len_sampled_quantized_augsignal_ids) + [sep_id] + [pad_id] * len(afib_id) + [sep_id]
        else:
            len_sampled_quantized_augsignal_ids = 0
            all_tokens = [cls_id] + quantized_signal_ids + [sep_id] + afib_id + [sep_id] 
            ref_tokens = [cls_id] + [pad_id] * len(quantized_signal_ids) + [sep_id] + [pad_id] * len(afib_id) + [sep_id]
        
        mask = np.ones_like(all_tokens)
        
        if args.afibmask == None:
            mask_indices_signal = np.random.choice(len(quantized_signal_ids), int(0.75 * len(quantized_signal_ids)), replace=False)
            mask[1:len(quantized_signal_ids)+1][mask_indices_signal] = 0
        
        mask[-2] = 0
        attention_mask = np.ones_like(all_tokens)
        
        if args.TA:
            masked_sample = np.copy(all_tokens2)
        else:
            masked_sample = np.copy(all_tokens)
            
        masked_sample[mask == 0] = mask_id
        
        input_ids = torch.LongTensor(masked_sample).to(device)
        mask_ids = torch.LongTensor(mask).to(device)
        attention_in = torch.tensor(attention_mask, dtype=torch.int).to(device)
        ref = torch.LongTensor(ref_tokens).to(device)
        
        input_ids = input_ids.unsqueeze(0)
        mask_ids= mask_ids.unsqueeze(0)
        attention_in = attention_in.unsqueeze(0)
        ref = ref.unsqueeze(0)
        mask_token_indices = torch.where(input_ids == mask_id)[1].tolist()

        all_attributions = []
        count = 0
        for index in tqdm(mask_token_indices, desc = 'Calculating Integrated Gradients..'):
            attributions, delta = lig.attribute(inputs=(input_ids, attention_in),
                                                target=index,
                                                additional_forward_args=(index,),
                                                baselines=(ref, torch.zeros_like(attention_in)),
                                                return_convergence_delta=True,
                                                internal_batch_size = 4)
            # Sum across embedding dimensions and take absolute value
            summed_attributions = attributions.sum(dim=-1).squeeze(0).abs()
            all_attributions.append(summed_attributions.detach().cpu())
            
        aggregated_attributions = torch.zeros_like(input_ids[0], dtype=torch.float).detach().cpu()
        for attributions in all_attributions:
            attributions = attributions[:1004 + len_sampled_quantized_augsignal_ids]
            aggregated_attributions += attributions
            
        normalized_attributions = aggregated_attributions / aggregated_attributions.max()
        tokens = [int(i) for i in input_ids[0]]
        mask[1000] = 0
        plot_attributions(signal, normalized_attributions.tolist(), key, mask[1:1001 + len_sampled_quantized_augsignal_ids], args)
        
        np_dic[key] = {
            'signal' :signal,
            'attr' : normalized_attributions.tolist(),
            'mask' : mask[1:1001 + len_sampled_quantized_augsignal_ids]
        }
    np.save(f'../runs/checkpoint/{args.checkpoint}/att_dic_{args.pre}_{args.afibmask}_{args.TS}_{args.TA}_{args.LF}_{args.CF}.npy', np_dic)