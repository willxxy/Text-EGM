import torch
torch.set_num_threads(2)
import numpy as np
from transformers import BigBirdForMaskedLM, LongformerForMaskedLM, BigBirdTokenizer, BigBirdForQuestionAnswering, \
                        AutoModelForMaskedLM, LongformerTokenizer, AutoTokenizer, BigBirdConfig, AutoImageProcessor, \
                            ViTForMaskedImageModeling, LongformerConfig
import argparse
from data_loader import EGMDataset, EGMIMGDataset, EGMTSDataset
from models import VITModel, TimeSeriesModel
from torch.utils.data import DataLoader
import gc
from runners import inference
import os

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--lr', type = float, default = 1e-4, help='Please choose the learning rate')
    parser.add_argument('--patience', type = int, default = 5, help = 'Please choose the patience of the early stopper')
    parser.add_argument('--signal_size', type = int, default = 250, help = 'Please choose the signal size')
    parser.add_argument('--device', type = str, default = 'cuda:1', help = 'Please choose the type of device' )
    parser.add_argument('--warmup', type = int, default = 2000, help = 'Please choose the number of warmup steps for the optimizer' )
    parser.add_argument('--epochs', type = int, default = 50, help = 'Please choose the number of epochs' )
    parser.add_argument('--batch', type = int, default = 2, help = 'Please choose the batch size')
    parser.add_argument('--weight_decay', type = float, default = 1e-2, help = 'Please choose the weight decay')
    parser.add_argument('--checkpoint', type = str, default = None, help = 'Please choose the path to the checkpoint to infer on')
    parser.add_argument('--model', type = str, default = 'big', help = 'Please choose which model to use')
    parser.add_argument('--mask', type=float, default=0.15, help = 'Pleasee choose percentage to mask for signal')
    parser.add_argument('--TS', action='store_true', help = 'Please choose whether to do Token Substitution')
    parser.add_argument('--TA', action='store_true', help = 'Please choose whether to do Token Addition')
    parser.add_argument('--LF', action='store_true', help = 'Please choose whether to do label flipping')    
    parser.add_argument('--toy', action = 'store_true', help = 'Please choose whether to use a toy dataset or not')
    parser.add_argument('--inference', action='store_true', help = 'Please choose whether it is inference or not')
    return parser.parse_args()

def create_toy(dataset, spec_ind):
    toy_dataset = {}
    for i in dataset.keys():
        _, placement, _, _ = i
        if placement in spec_ind:
            toy_dataset[i] = dataset[i]    
    return toy_dataset

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")

if __name__ == '__main__':

    gc.collect()
    torch.cuda.empty_cache()
    args = get_args()
    torch.manual_seed(2)
    device = torch.device(args.device)
    print(device)
    print('Loading Data...')

    test = np.load('./data/test_intra.npy', allow_pickle = True).item()
    
    if args.toy:
        test = create_toy(test, [18])

    print('Creating Custom Tokens')
    custom_tokens = [
        f"signal_{i}" for i in range(args.signal_size+1)
    ] + [
        f"afib_{i}" for i in range(2)
    ]
    if args.TA:
        custom_tokens += [
        f"augsig_{i}" for i in range(args.signal_size+1)
    ]

    print('Initalizing Model...')
    if args.model == 'big':
        model = BigBirdForMaskedLM.from_pretrained("google/bigbird-roberta-base").to(device)
        model.config.attention_type = 'original_full'
        tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")
        num_added_tokens = tokenizer.add_tokens(custom_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model_hidden_size = model.config.hidden_size
    
    if args.model == 'qa_big':
        model = BigBirdForQuestionAnswering.from_pretrained("google/bigbird-roberta-base").to(device)
        tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")
        num_added_tokens = tokenizer.add_tokens(custom_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model_hidden_size = model.config.hidden_size
    
    if args.model == 'long':
        model = LongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096").to(device)
        tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
        num_added_tokens = tokenizer.add_tokens(custom_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model_hidden_size = model.config.hidden_size
    
    if args.model == 'raw_big':
        configuration = BigBirdConfig(attention_type = 'original_full')
        model = BigBirdForMaskedLM(config = configuration).to(device)
        tokenizer = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")
        num_added_tokens = tokenizer.add_tokens(custom_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model_hidden_size = model.config.hidden_size
       
    if args.model =='clin_bird':
        model = AutoModelForMaskedLM.from_pretrained("yikuan8/Clinical-BigBird").to(device)
        model.config.attention_type = 'original_full'
        tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-BigBird")
        num_added_tokens = tokenizer.add_tokens(custom_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model_hidden_size = model.config.hidden_size
        
    if args.model =='clin_long':
        model = AutoModelForMaskedLM.from_pretrained("yikuan8/Clinical-Longformer").to(device)
        tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
        num_added_tokens = tokenizer.add_tokens(custom_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model_hidden_size = model.config.hidden_size
        
    if args.model == 'vit':
        tokenizer = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        pt_model = ViTForMaskedImageModeling.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
        args.num_patches = (pt_model.config.image_size // pt_model.config.patch_size) ** 2
        model_hidden_size = pt_model.config.hidden_size
        model = VITModel(pt_model, model_hidden_size, 2).to(device)
        
    if args.model == 'big_ts':
        pt_model = BigBirdForMaskedLM.from_pretrained("google/bigbird-roberta-base").to(device)
        pt_model.config.attention_type = 'original_full'
        model_hidden_size = pt_model.config.hidden_size
        model = TimeSeriesModel(pt_model, model_hidden_size, 2).to(device)
        tokenizer = None
        
    if args.model == 'long_ts':
        pt_model = LongformerForMaskedLM.from_pretrained("allenai/longformer-base-4096").to(device)
        model_hidden_size = pt_model.config.hidden_size
        model = TimeSeriesModel(pt_model, model_hidden_size, 2).to(device)
        tokenizer = None
        
    if args.model == 'raw_long':
        configuration = LongformerConfig()
        model = LongformerForMaskedLM(config = configuration).to(device)
        tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
        tokenizer.add_tokens(custom_tokens)
        model.resize_token_embeddings(len(tokenizer))
        model_hidden_size = model.config.hidden_size
        
    print('Creating Dataset and DataLoader...')
    
    if args.model == 'vit':
        test_dataset = EGMIMGDataset(test, tokenizer, args= args)
    elif args.model == 'big_ts' or args.model == 'long_ts':
        test_dataset = EGMTSDataset(test, args = args)        
    else:
        test_dataset = EGMDataset(test, tokenizer, args = args)
        
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False)   
    
    checkpoint = torch.load(f'./runs/checkpoint/{args.checkpoint}/best_checkpoint.chkpt', map_location = args.device)
    model.load_state_dict(checkpoint['model'])
    print(f'Inferencing checkpoint {args.checkpoint}... ')
    inference(model, tokenizer, test_loader, device, args)
