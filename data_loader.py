from transformers import AutoImageProcessor
import torch
from torch.utils.data import Dataset
import numpy as np
from pyts.image import GramianAngularField, RecurrencePlot, MarkovTransitionField


class EGMDataset(Dataset):
    def __init__(self, data_dict, tokenizer = None, args = None):
        self.data = list(data_dict.values())
        self.keys = list(data_dict.keys())
        self.args = args
        self.signal_size = self.args.signal_size
        self.tokenizer = tokenizer
        self.vocab_size = len(self.tokenizer.get_vocab())
        self.cls = self.tokenizer.cls_token
        self.mask = self.tokenizer.mask_token
        self.SEP = self.tokenizer.sep_token
        self.curr_signal_len = 1000
                
        if self.args.TA:
            self.pad=  self.tokenizer.pad_token
                            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        signal = sample[:1000]
        key = self.keys[index]
        afib_label = key[3]
        
        #### Augmentation
        ### Token Substitution (TS), Token Addition (TA), Label Flipping (LF)
        ## 1 - None
        ## 2 - TS or TA or LF
        augmentation_scheme = np.random.randint(1, 5)  # Choose between 1 or 2
        if self.args.TS and augmentation_scheme == 2:
            signal = self.moving_average(signal)
        if self.args.LF and augmentation_scheme == 2:
            afib_label = self.label_flip(afib_label)
        
        afib_token = f"afib_{int(afib_label)}" 
        
        min_val, max_val = np.min(signal), np.max(signal)
        normalized_signal = (signal - min_val) / (max_val - min_val)
        quantized_signal = np.floor(normalized_signal * self.signal_size).astype(int)
        quantized_signal_tokens = [f"signal_{i}" for i in quantized_signal]
        
        quantized_signal_ids = self.tokenizer.convert_tokens_to_ids(quantized_signal_tokens)
        
        concatenated_sample = np.concatenate([signal, np.array([afib_label])])
        
        afib_id = self.tokenizer.convert_tokens_to_ids([afib_token])
        mask_id = self.tokenizer.convert_tokens_to_ids(self.mask)
        cls_id = self.tokenizer.convert_tokens_to_ids(self.cls)
        sep_id = self.tokenizer.convert_tokens_to_ids(self.SEP)
        
        if self.args.TA:
            quantized_augsignal_tokens = [f"augsig_{i}" for i in quantized_signal]
            sampled_quantized_augsignal_tokens = self.sample_consecutive(quantized_augsignal_tokens, int(0.25 * len(quantized_signal_ids)))
            sampled_quantized_augsignal_ids = self.tokenizer.convert_tokens_to_ids(sampled_quantized_augsignal_tokens)
            pad_id = self.tokenizer.convert_tokens_to_ids(self.pad)
            all_tokens  =[cls_id] + quantized_signal_ids + [pad_id] * int(0.25 * len(quantized_signal_ids)) + [sep_id] + afib_id + [sep_id]
            if augmentation_scheme == 2:
                all_tokens2  =[cls_id] + quantized_signal_ids + sampled_quantized_augsignal_ids + [sep_id] + afib_id + [sep_id]
            else:
                all_tokens2  =[cls_id] + quantized_signal_ids + [pad_id] * int(0.25 * len(quantized_signal_ids)) + [sep_id] + afib_id + [sep_id]
        else:
            all_tokens = [cls_id] + quantized_signal_ids + [sep_id] + afib_id + [sep_id]

        mask = np.ones_like(all_tokens)
        
            
        mask_indices_signal = np.random.choice(self.curr_signal_len, int(self.args.mask * self.curr_signal_len), replace=False)
        mask[1:self.curr_signal_len+1][mask_indices_signal] = 0

        mask[-2] = 0
        
        if self.args.TA:
            masked_sample = np.copy(all_tokens2)
        else:
            masked_sample = np.copy(all_tokens)
            
        attention_mask = np.ones_like(all_tokens)

        masked_sample[mask == 0] = mask_id
        

        return torch.LongTensor(masked_sample), torch.LongTensor(all_tokens), torch.tensor(concatenated_sample, dtype=torch.float32), torch.LongTensor(mask), \
            torch.tensor(attention_mask, dtype=torch.int), key, torch.tensor(min_val, dtype=torch.float32), torch.tensor(max_val, dtype=torch.float32)
        
    def label_flip(self, afib_label):
        if afib_label == 0:
            afib_label = 1
        elif afib_label == 1:
            afib_label = 0
        return afib_label
    
    def moving_average(self, signal, window_size=50):
        return np.convolve(signal, np.ones(window_size), 'same') / window_size
                 
    def sample_consecutive(self, signal, sample_size):
        max_start_index = len(signal) - sample_size
        start_index = np.random.randint(0, max_start_index)
        return signal[start_index:start_index + sample_size]  
                
                
class EGMIMGDataset(Dataset):
    def __init__(self, data_dict, tokenizer = None, args = None):
        self.data = list(data_dict.values())
        self.keys = list(data_dict.keys())
        self.args = args
        self.tokenizer = tokenizer
        self.gaf = GramianAngularField(method='summation')
        self.rp = RecurrencePlot()
        self.mtf = MarkovTransitionField(n_bins=4)
        
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        signal = sample[:1000]
        key = self.keys[index]
        afib_label = key[3]
        
        augmentation_scheme = np.random.randint(1, 5)  # Choose between 1 or 2
        
        if self.args.TS and augmentation_scheme == 2:
            signal = self.moving_average(signal)
        if self.args.LF and augmentation_scheme == 2:
            afib_label = self.label_flip(afib_label)
            
        if self.args.TA:
            sampled_quantized_augsignal_tokens = self.sample_consecutive(signal, int(0.25 * len(signal)))
            pad_id = np.zeros(len(sampled_quantized_augsignal_tokens))
            if augmentation_scheme == 2:
                signal = np.concatenate([signal, sampled_quantized_augsignal_tokens])
            else:
                signal = np.concatenate([signal, pad_id])
            
        img = self.prepare_img(signal)
        mask = torch.rand(size=(1, self.args.num_patches)) < 0.75
        mask = mask.squeeze(0)
        pixel_values = self.tokenizer(images=img, return_tensors="pt").pixel_values.squeeze(0)
        
        return pixel_values, mask, afib_label
                
    def prepare_img(self, x):
        x = x.reshape(1, -1)
        gaf_img = self.gaf.fit_transform(x)
        rp_img = self.rp.fit_transform(x)
        mtf_img = self.mtf.fit_transform(x)
        
        gaf_img = np.interp(gaf_img, [-1., 1.], [0., 255.])
        rp_img = np.interp(rp_img, [0, 1], [0., 255.])
        mtf_img = np.interp(mtf_img, [0, 1], [0., 255.])
        
        
        img = np.vstack([gaf_img, rp_img, mtf_img])
        img = img.astype(np.uint8)
        img = np.moveaxis(img, 0, -1)
        
        return img
    
    def label_flip(self, afib_label):
        if afib_label == 0:
            afib_label = 1
        elif afib_label == 1:
            afib_label = 0
        return afib_label
    
    def moving_average(self, signal, window_size=50):
        return np.convolve(signal, np.ones(window_size), 'same') / window_size
                 
    def sample_consecutive(self, signal, sample_size):
        max_start_index = len(signal) - sample_size
        start_index = np.random.randint(0, max_start_index)
        return signal[start_index:start_index + sample_size]  



class EGMTSDataset(Dataset):
    def __init__(self, data_dict, args = None):
        self.data = list(data_dict.values())
        self.keys = list(data_dict.keys())
        self.args = args
        self.signal_size = 250
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        signal = sample[:1000]
                                
        key = self.keys[index]
        afib_label = key[3]
        
        augmentation_scheme = np.random.randint(1, 5)  # Choose between 1 or 2
        
        if self.args.TS and augmentation_scheme == 2:
            signal = self.moving_average(signal)
        if self.args.LF and augmentation_scheme == 2:
            afib_label = self.label_flip(afib_label)
            
        if self.args.TA:
            sampled_quantized_augsignal_tokens = self.sample_consecutive(signal, int(0.25 * len(signal)))
            pad_id = np.zeros(len(sampled_quantized_augsignal_tokens))
            if augmentation_scheme == 2:
                signal = np.concatenate([signal, sampled_quantized_augsignal_tokens])
            else:
                signal = np.concatenate([signal, pad_id])
        
        min_val, max_val = np.min(signal), np.max(signal)
        normalized_signal = (signal - min_val) / (max_val - min_val)
        
        mask = np.ones_like(normalized_signal)
        mask_indices_signal_curr = np.random.choice(1000, int(self.args.mask * (1000)), replace=False)    
        mask[mask_indices_signal_curr] = 0
        masked_signal = np.copy(normalized_signal)
        masked_signal[mask_indices_signal_curr] = 0
        attention_mask = np.ones_like(normalized_signal)
        
        
        return torch.tensor(masked_signal, dtype= torch.float32), torch.LongTensor(normalized_signal), afib_label, torch.LongTensor(mask), \
                torch.tensor(attention_mask, dtype=torch.int)
                
    def label_flip(self, afib_label):
        if afib_label == 0:
            afib_label = 1
        elif afib_label == 1:
            afib_label = 0
        return afib_label
    
    def moving_average(self, signal, window_size=50):
        return np.convolve(signal, np.ones(window_size), 'same') / window_size
                 
    def sample_consecutive(self, signal, sample_size):
        max_start_index = len(signal) - sample_size
        start_index = np.random.randint(0, max_start_index)
        return signal[start_index:start_index + sample_size]  
    