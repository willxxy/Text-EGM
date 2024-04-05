import numpy as np
import matplotlib.pyplot as plt
import argparse 

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--checkpoint', type=str, default = None, help = 'Please choose the checkpoint to open. Do not include the .chkpt.')
    parser.add_argument('--instance', type= int, default = 0, help = 'Please choose an instance to visualize')
    parser.add_argument('--time', type= int, help = 'Please specify the time size')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    print(args.checkpoint)
    data = np.load(f'../runs/checkpoint/{args.checkpoint}/best_np.npy', allow_pickle=True).item()

    mask = data['masked_signals']
    gt = data['gt_signals']
    pred = data['pred_signals']

    if args.time == 1:
        time = np.linspace(0, 1, 1000) # for regular
    elif args.time == 2:
        time = np.linspace(0, 2, 2000) # for forecasting
    elif args.time == 3:
        time = np.linspace(0, 3, 3000) # for forecasting
    elif args.time == 4:
        time = np.linspace(0, 4, 4000) # for forecasting
    inst = args.instance

    plt.figure(figsize=(10, 6))
    plt.plot(time, gt[inst])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Electrogram Signal')
    plt.grid(True)
    plt.savefig(f'../runs/checkpoint/{args.checkpoint}/gt_signal_{args.instance}.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(time, pred[inst])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Electrogram Signal')
    plt.grid(True)
    plt.savefig(f'../runs/checkpoint/{args.checkpoint}/pred_signal_{args.instance}.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(time, pred[inst], label='Stitched Signal')
    
    valid_mask_indices = mask[inst][mask[inst] < len(time)]
    plt.scatter(time[valid_mask_indices], pred[inst][valid_mask_indices], color='r', label='Predicted Sections', s = 10)
    plt.title('Stitched Signals with Predicted Sections')
    plt.xlabel('Time')
    plt.ylabel('Signal Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'../runs/checkpoint/{args.checkpoint}/stitched_{args.instance}.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(time, gt[inst], color='orange', label='Ground Truth')
    plt.plot(time, pred[inst], label='Predicted')
    plt.title('Comparison of Ground Truth and Predicted Signals')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'../runs/checkpoint/{args.checkpoint}/comparison_{args.instance}.png')
    plt.close()
