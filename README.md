# Interpretation of Intracardiac Electrograms Through Textual Representations

William Jongwon Han, Diana Gomez, Avi Alok, Chaojing Duan, Michael A. Rosenberg, Douglas Weber, Emerson Liu, Ding Zhao.

Official code for "[Interpretation of Intracardiac Electrograms Through Textual Representations](https://arxiv.org/abs/2402.01115)" accepted by 2024 Conference on Health, Inference, and Learning (CHIL).

If you experience any bugs or have any questions, please submit an issue or contact at wjhan{at}andrew{dot}cmu{dot}edu.

We thank the Mario Lemieux Center for Heart Rhythm Care at Allegheny General Hospital for supporting this work.

## Set Up Environment

Note: We have only tested on Ubuntu 20.04.5 LTS. 

1. `conda create -n envname python=3.8`

2. `conda activate envname`

3. `git clone https://github.com/willxxy/ekg-af.git`

4. `cd ekg-af`

5. `pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113`

Note: Please ensure that the pip you are using is from the conda environment

6. Test if pytorch version is compatible with current, available gpus by executing `python gpu.py`. Currently, we have only tested on A5000 (24 GB) and A6000 (48 GB) NVIDIA GPUs.

7. `pip install -r requirements.txt`

## Set Up Data

1. Although the data we curated is not publicly available, we do have experimental results on an external dataset (main results are in Table 2 in the paper), namely the "Intracardiac Atrial Fibrillation Database" available on PhysioNet.

2. To set up this data, `cd` into the `preprocess` folder.

3. Please execute the following command to download the data.

```
wget https://physionet.org/static/published-projects/iafdb/intracardiac-atrial-fibrillation-database-1.0.0.zip
```

4. Unzip the file by executing

```
unzip intracardiac-atrial-fibrillation-database-1.0.0
```

5. Now execute the folllowing command to preprocess the data.

```
sh preprocess.sh
```

6. This should create a data folder with several `.npy` for training, validation, and test.


## Start Training

1. From the preprocess folder `cd ../` back to the main directory.

2. You can now directly use `train.sh` files to start training.

## Inference

1. Please execute `sh inference.sh` after training. Make sure to specify the checkpoint path.

## Visualizations

All visualizations will be saved under their respective checkpoint folder.
Please `cd visualize` before visualizing. 
Under the `visualize` folder, please view the following scripts:


1. `stitch.sh` - Visualizes the reconstructed and forecasted signals. 

2. `viz_tokens.sh` - Visualizes the tokenized representation of the signal. 

3. `viz_attentions.sh` - Visualizes the attention map of the model. 

4. `viz_int_grad.sh` - Visualizes the attribution scores of the model.

## Citation

If you found this repository or work helpful to your own, please cite the following bibtex.

```
@misc{han2024interpretation,
      title={Interpretation of Intracardiac Electrograms Through Textual Representations}, 
      author={William Jongwon Han and Diana Gomez and Avi Alok and Chaojing Duan and Michael A. Rosenberg and Douglas Weber and Emerson Liu and Ding Zhao},
      year={2024},
      eprint={2402.01115},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
