import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, mean_absolute_error

def trainer(model, train_dataloader, optimizer, device, args, ce = None):
    model.train()  
    losses = 0
    len_of_batch = 0
    count =0
    for batch in tqdm(train_dataloader, desc = 'Training... '):
        optimizer.zero_grad()
        
        if args.model == 'vit':
            batch_data, mask, afib_label = batch
            batch_data, mask, afib_label = batch_data.to(device), mask.to(device), afib_label.to(device)
            logits, loss = model(batch_data, afib_label, mask=mask)
        elif args.model == 'big_ts' or args.model == 'long_ts':
            batch_data, tokenized_sample, afib_label, mask, batch_attention_mask = batch
            batch_data, tokenized_sample, afib_label, mask, batch_attention_mask = batch_data.to(device), tokenized_sample.to(device), afib_label.to(device), mask.to(device), batch_attention_mask.to(device)
            logits, loss = model(batch_data, tokenized_sample, batch_attention_mask, afib_label)
            loss = torch.mean(loss)
        else:
            batch_data, tokenized_sample, concat, mask, batch_attention_mask, _, _, _, = batch
            batch_data, tokenized_sample, concat, mask, batch_attention_mask = batch_data.to(device), tokenized_sample.to(device), concat.to(device), mask.to(device), batch_attention_mask.to(device)

            if args.model == 'big' or args.model == 'clin_bird' or args.model == 'raw_big':
                outputs = model(input_ids = batch_data, attention_mask = batch_attention_mask, labels = tokenized_sample, output_hidden_states = True)
                logits = outputs.logits
                
            if args.model == 'long' or args.model == 'clin_long' or args.model == 'raw_long':
                
                outputs = model(input_ids = batch_data, attention_mask = batch_attention_mask, labels = tokenized_sample, output_hidden_states = True, global_attention_mask = mask)
                logits = outputs.logits
        
            loss = outputs.loss
            loss = torch.mean(loss)
    
            if ce != None:
                logits_reshaped = logits.view(batch_data.size(0), batch_data.size(1), -1)
                afib_logits = logits_reshaped[:, -2, :]
                afib_label = tokenized_sample[:, -2]
                afib_ce_loss = ce(afib_logits, afib_label)    
                afib_ce_loss = torch.mean(afib_ce_loss)
                loss = (args.ce_weight * afib_ce_loss) + loss              
                
        loss.backward()
        optimizer.step_and_update_lr()
        losses+=loss.item()
        len_of_batch +=1
        
    average_loss = losses/len_of_batch
    
    return average_loss

    
    
def validate(model, val_dataloader, device, args, ce = None):
    model.eval()
    total_loss = 0
    len_of_batch = 0
    count = 0
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc = 'Validating... '):
            
            if args.model == 'vit':
                batch_data, mask, tokenized_sample = batch
                batch_data, mask, tokenized_sample = batch_data.to(device), mask.to(device), tokenized_sample.to(device)
                logits, loss = model(batch_data, tokenized_sample, mask=mask)
            elif args.model == 'big_ts' or args.model == 'long_ts':
                batch_data, tokenized_sample, afib_label, mask, batch_attention_mask = batch
                batch_data, tokenized_sample, afib_label, mask, batch_attention_mask = batch_data.to(device), tokenized_sample.to(device), afib_label.to(device), mask.to(device), batch_attention_mask.to(device)
                logits, loss = model(batch_data, tokenized_sample, batch_attention_mask, afib_label)
                loss = torch.mean(loss)
            else:
                batch_data, tokenized_sample, concat, mask, batch_attention_mask, _,  _, _, = batch
                batch_data, tokenized_sample, concat, mask, batch_attention_mask = batch_data.to(device), tokenized_sample.to(device), concat.to(device), mask.to(device), batch_attention_mask.to(device)
                if args.model == 'big' or args.model == 'clin_bird' or args.model == 'raw_big':
                    outputs = model(input_ids = batch_data, attention_mask = batch_attention_mask, labels = tokenized_sample, output_hidden_states = True)
                    logits = outputs.logits
                if args.model == 'long' or args.model == 'clin_long' or args.model == 'raw_long':
                    outputs = model(input_ids = batch_data, attention_mask = batch_attention_mask, labels = tokenized_sample, output_hidden_states = True, global_attention_mask = mask)
                    logits = outputs.logits
                                    
                loss = outputs.loss
                loss = torch.mean(loss)
        
                if ce != None:
                    logits_reshaped = logits.view(batch_data.size(0), batch_data.size(1), -1)
                    afib_logits = logits_reshaped[:, -2, :]
                    afib_label = tokenized_sample[:, -2]
                    afib_ce_loss = ce(afib_logits, afib_label)    
                    afib_ce_loss = torch.mean(afib_ce_loss)
                    loss = (args.ce_weight * afib_ce_loss) + loss                 
                    
            total_loss += loss.item()
            len_of_batch +=1

    avg_loss = total_loss / len_of_batch
    
    return avg_loss

    
def stitch_sequences(input_seq, mask, pred_masked):
    full_seq = np.copy(input_seq)
    masked_positions = np.where(mask == 0)[0]
    full_seq[masked_positions] = pred_masked
    return full_seq, masked_positions

    
def extract_value(token):
    if '_' in token:
        if '>' in token:
            return -2
        else:
            num = int(token.split('_')[1])
            return num
    else:
        return -1 
    
def decode_from_tokens(tokenizer, tokens, signal_size, min_vals, max_vals, args):
    decoded_signals = []
    decoded_afibs = []
    for i in range(tokens.shape[0]):
        output_tokens = tokenizer.convert_ids_to_tokens(tokens[i])
        
        quantized_signal = torch.tensor([extract_value(token) for token in output_tokens[1:1001]]).to(args.device)
            
    
        quantized_afib = torch.tensor([extract_value(token) for token in [output_tokens[-2]]]).to(args.device)
                
        min_val = min_vals[i]
        max_val = max_vals[i]
        
        # Decode signal
        normalized_signal_values = (quantized_signal - 1) / (signal_size - 1)
        decoded_signal = normalized_signal_values * (max_val - min_val) + min_val
        
        decoded_signals.append(decoded_signal)
        decoded_afibs.append(quantized_afib)
    
    return torch.stack(decoded_signals), torch.stack(decoded_afibs)
    
def inference(model, tokenizer, test_dataloader, device, args):
    model.eval()
    stitched_sequences = []
    count =0
    ground_truth_sequences = []
    masked_positions_list = []
    MSEs_signals = []
    MAEs_signals = []
    ground_truth_afib = []
    pred_afib = []
    mean_accuracies_afib = []
    all_attentions = []
    count_afib = 0
    count_norm = 0
    all_global_attentions = []
    all_tokens = []
    count_index = 0
    count_index_list = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='Inference...'):
            
            if args.model == 'vit':
                batch_data, mask, tokenized_sample = batch
                batch_data, mask, tokenized_sample = batch_data.to(device), mask.to(device), tokenized_sample.to(device)
                logits, _ = model(batch_data, tokenized_sample, mask=mask)
                preds = torch.argmax(logits, dim=-1)
            elif args.model == 'big_ts' or args.model == 'long_ts':
                batch_data, tokenized_sample, afib_label, mask, batch_attention_mask = batch
                batch_data, tokenized_sample, afib_label, mask, batch_attention_mask = batch_data.to(device), tokenized_sample.to(device), afib_label.to(device), mask.to(device), batch_attention_mask.to(device)
                logits, _ = model(batch_data, tokenized_sample, batch_attention_mask, afib_label)
                tokenized_sample = afib_label
                preds = torch.argmax(logits, dim=-1)
            else:
                batch_data, tokenized_sample, batch_raw, batch_mask, batch_attention_mask, key, min_val, max_val = batch
                batch_data, tokenized_sample, batch_raw, batch_mask, batch_attention_mask, min_val, max_val = batch_data.to(device), tokenized_sample.to(device), batch_raw.to(device), batch_mask.to(device), batch_attention_mask.to(device), min_val.to(device), max_val.to(device)
                if args.model == 'big' or args.model == 'clin_bird' or args.model == 'raw_big':
                    outputs = model(input_ids = batch_data, attention_mask = batch_attention_mask, output_attentions = True)
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=-1)        
                    attentions = outputs.attentions
                    
                    if int(key[-1][0]) == 0 and not count_norm > 3:
                        tokens_cpu = [tokens.detach().cpu().numpy() for tokens in tokenized_sample]
                        attentions_cpu = [attn.detach().cpu().numpy() for attn in attentions]
                        all_tokens.append(tokens_cpu)
                        all_attentions.append(attentions_cpu)
                        count_norm +=1
                        count_index_list.append(count_index)
                    if int(key[-1][0]) == 1 and not count_afib > 3: 
                        tokens_cpu = [tokens.detach().cpu().numpy() for tokens in tokenized_sample]
                        attentions_cpu = [attn.detach().cpu().numpy() for attn in attentions]
                        all_tokens.append(tokens_cpu)
                        all_attentions.append(attentions_cpu)
                        count_afib +=1
                        count_index_list.append(count_index)
                        
                if args.model == 'long' or args.model == 'clin_long' or args.model == 'raw_long':
                    outputs = model(input_ids = batch_data, attention_mask = batch_attention_mask, output_hidden_states = True, output_attentions = True, global_attention_mask = batch_mask)
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=-1)
                    attentions = outputs.attentions
                    global_attentions = outputs.global_attentions
                    if int(key[-1][0]) == 0 and not count_norm > 3:
                        tokens_cpu = [tokens.detach().cpu().numpy() for tokens in tokenized_sample]
                        all_tokens.append(tokens_cpu)
                        attentions_cpu = [attn.detach().cpu().numpy() for attn in attentions]
                        all_attentions.append(attentions_cpu)
                        global_attentions_cpu = [attn.detach().cpu().numpy() for attn in global_attentions]
                        all_global_attentions.append(global_attentions_cpu)
                        count_norm +=1
                        count_index_list.append(count_index)
                    if int(key[-1][0]) == 1 and not count_afib > 3:
                        tokens_cpu = [tokens.detach().cpu().numpy() for tokens in tokenized_sample]
                        all_tokens.append(tokens_cpu)
                        attentions_cpu = [attn.detach().cpu().numpy() for attn in attentions]
                        all_attentions.append(attentions_cpu)
                        global_attentions_cpu = [attn.detach().cpu().numpy() for attn in global_attentions]
                        all_global_attentions.append(global_attentions_cpu)
                        count_afib +=1
                        count_index_list.append(count_index)
                
                new_args = {
                    'signal_size': args.signal_size, 
                    'min_val': min_val,
                    'max_val': max_val
                }
            
            if args.model == 'big' or args.model == 'clin_bird' or args.model == 'clin_long' or args.model == 'long' or args.model == 'raw_big' or args.model == 'raw_long':
                decoded_signal, decoded_afib = decode_from_tokens(tokenizer, preds, new_args['signal_size'], new_args['min_val'],new_args['max_val'], args)
                decoded = torch.cat([decoded_signal, decoded_afib], dim=1)

                for i in range(batch_data.shape[0]):
                    masked_positions_i = (batch_mask[i] == 0)
                    new_mask = batch_mask[i]
                    masked_positions_i  = torch.cat([masked_positions_i[1:1001],masked_positions_i[-2].unsqueeze(0)], dim=0)
                    preds_masked_i = decoded.cpu().numpy()[i][masked_positions_i.cpu().numpy()]
                    new_mask  = torch.cat([batch_mask[i][1:1001], batch_mask[i][-2].unsqueeze(0)], dim=0)
                        
                    stitched_seq, masked_position = stitch_sequences(batch_raw[i].cpu().numpy(), new_mask.cpu().numpy(), preds_masked_i)
                    masked_positions_list.append(masked_position)
                    ground_truth_seq = batch_raw[i].cpu().numpy()
                    
                    stitched_sequences.append(stitched_seq[:1000])
                    ground_truth_sequences.append(ground_truth_seq[:1000])
                    afib_stitched = stitched_seq[-1]
                    afib_gt = ground_truth_seq[-1]
                    
                    ground_truth_afib.append(afib_gt)
                    pred_afib.append(afib_stitched)
                    
                    # MSE for signal
                    mse_signal = mean_squared_error(stitched_seq, ground_truth_seq)
                    MSEs_signals.append(mse_signal)
                    
                    # MAE for signal
                    mae_signal = mean_absolute_error(stitched_seq, ground_truth_seq)
                    MAEs_signals.append(mae_signal)

                    # Acc for Elec 
                    afib_stitched = int(afib_stitched)
                    afib_gt = int(afib_gt)
                    
                    mean_acc_afib = accuracy_score([afib_stitched], [afib_gt])
                    mean_accuracies_afib.append(mean_acc_afib)

            if args.model == 'vit' or args.model == 'long_ts' or args.model == 'big_ts':
                # Afib Accuracy
                for i in range(preds.shape[0]):
                    pred = preds[i]
                    pred_afib.append(pred.detach().cpu().numpy())
                    ground_truth_afib.append(tokenized_sample[i].detach().cpu().numpy())
                    mean_acc_afib = accuracy_score([pred.detach().cpu().numpy()], [tokenized_sample[i].detach().cpu().numpy()] )
                    mean_accuracies_afib.append(mean_acc_afib)

            count_index +=1
            
    if args.model == 'vit' or args.model == 'big_ts' or args.model == 'long_ts':
        print("Average Accuracy for Afib:", np.mean(mean_accuracies_afib))
    else:
        print('MSE for Signal Interpolation', np.mean(MSEs_signals))
        print('MAE for Signal Interpolation', np.mean(MAEs_signals))
        print("Average Accuracy for AFib:", np.mean(mean_accuracies_afib))
    
    cm = confusion_matrix(ground_truth_afib, pred_afib)
    print(f'Confusion Matrix: {cm}')

    try:
        TP = cm[0, 0]
        FN = cm[0, 1]
        FP = cm[1, 0]
        TN = cm[1, 1]
        sensitivity = TP / float(TP + FN) if (TP + FN) != 0 else 0
        specificity = TN / float(TN + FP) if (TN + FP) != 0 else 0
        npv = TN / float(TN + FN) if (TN + FN) != 0 else 0
        ppv = TP / float(TP + FP) if (TP + FP) != 0 else 0
        print("Sensitivity:", sensitivity)
        print("Specificity:", specificity)
        print("NPV:", npv)
        print("PPV:", ppv)
    except:
        print('ravel error')
        
    if args.model == 'long' or args.model == 'clin_long'or args.model == 'raw_long':
        np_save = {
            'masked_signals' : masked_positions_list,
            'gt_signals' : ground_truth_sequences,
            'pred_signals' : stitched_sequences,
            'gt_afib' : ground_truth_afib,
            'pred_afib' : pred_afib,
            'attentions' : all_attentions,
            'global_attentions': all_global_attentions,
            'tokens' : all_tokens,
            'index': count_index_list
        }
    elif args.model == 'vit' or args.model == 'big_ts' or args.model == 'long_ts':
        np_save = {
                'gt_afib' : ground_truth_afib,
                'pred_afib' : pred_afib,
            }
    else:
        np_save = {
            'masked_signals' : masked_positions_list,
            'gt_signals' : ground_truth_sequences,
            'pred_signals' : stitched_sequences,
            'gt_afib' : ground_truth_afib,
            'pred_afib' : pred_afib,
            'attentions' : all_attentions,
            'tokens' : all_tokens,
            'index': count_index_list
        }
    np.save(f'./runs/checkpoint/{args.checkpoint}/best_np.npy', np_save)
