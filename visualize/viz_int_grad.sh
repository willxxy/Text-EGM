# python int_grad.py --checkpoint=saved_best_0.0001_8_5_0.01_big_True_0.75_1.0_1.0_None_False_False_False_True_False --device=cuda:1 --model=big --TA
# python int_grad.py --checkpoint=saved_best_0.0001_8_5_0.01_big_True_0.75_1.0_1.0_None_False_False_False_True_False --device=cuda:1 --pre --model=big --TA
# python int_grad.py --checkpoint=saved_best_0.0001_8_5_0.01_big_True_0.75_1.0_1.0_None_False_False_False_True_False --device=cuda:1 --pre --afibmask --model=big --TA
# python int_grad.py --checkpoint=saved_best_0.0001_8_5_0.01_big_True_0.75_1.0_1.0_None_False_False_False_True_False --device=cuda:1 --afibmask --model=big --TA

# python int_grad.py --checkpoint=saved_best_0.0001_8_5_0.01_big_True_0.75_1.0_1.0_None_False_False_False_True_False --device=cuda:1 --model=big --TA --CF
# python int_grad.py --checkpoint=saved_best_0.0001_8_5_0.01_big_True_0.75_1.0_1.0_None_False_False_False_True_False --device=cuda:1 --pre --model=big --TA --CF
# python int_grad.py --checkpoint=saved_best_0.0001_8_5_0.01_big_True_0.75_1.0_1.0_None_False_False_False_True_False --device=cuda:1 --pre --afibmask --model=big --TA --CF
# python int_grad.py --checkpoint=saved_best_0.0001_8_5_0.01_big_True_0.75_1.0_1.0_None_False_False_False_True_False --device=cuda:1 --afibmask --model=big --TA --CF


python int_grad.py --checkpoint=saved_best_0.0001_8_5_0.01_big_True_0.75_1.0_1.0_None_False_False_False --device=cuda:2 --model=big --afibmask
python int_grad.py --checkpoint=saved_best_0.0001_8_5_0.01_big_True_0.75_1.0_1.0_None_False_False_False --device=cuda:2 --model=big --pre --afibmask
python int_grad.py --checkpoint=saved_best_0.0001_8_5_0.01_big_True_0.75_1.0_1.0_None_False_False_False --device=cuda:2 --model=big
python int_grad.py --checkpoint=saved_best_0.0001_8_5_0.01_big_True_0.75_1.0_1.0_None_False_False_False --device=cuda:2 --model=big --pre