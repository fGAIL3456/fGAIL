# fGAIL
This repository contains codes for the project of f-GAIL, which aims at automatically learn an f-divergence for GAIL. Please find the paper at (https://arxiv.org/abs/2010.01207). The f-GAIL implementation can be found in 
```
a2c_ppo_acktr/algo/fgail.py
```
## Running the code
Train expert with predefined rewards:
```
python main.py --env-name HalfCheetah-v2 --algo ppo --use-gae --lr 3e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 1000 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01
```
Trained expert policy will be saved under ./trained_models/ folder. To save expert demonstrations using the epert policy, run
```
python save_expert_traj.py --env-name HalfCheetah-v2 --traj-num 200
```
Generated expert trajectories are saved under ./expert_traj/ directory. To use generated trajectories train an f-GAIL model, run
```
python main.py --env-name HalfCheetah-v2 --algo ppo --use-gae --log-interval 1 --num-steps 1000 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 10000000 --use-linear-lr-decay --use-proper-time-limits --fgail --fgail-batch-size 50 --fnum 3 --save-dir ./trained_models/fGAIL/
```
Then trained model will be saved under the ./trained_models/fGAIL/ folder.



