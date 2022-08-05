# Bootstrapping-RL-4-Sequential-Picking

## Initial instruction

### Use anaconda to create a virtual environment

**Step 1.** install miniconda

```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

**Step 2.** clone repo and create conda environment

```shell
git clone https://github.com/VittorioGiammarino/Bootstrapping-RL-4-Sequential-Picking.git
```

```shell
cd Bootstrapping-RL-4-Sequential-Picking
conda env create -f environment.yml
conda activate BRL4SP
```

**Step 3.** Download pretrained policy and assets

Assets: https://drive.google.com/file/d/1UqU3PPLOr9Y4cY9mQNXKK4QkxlnVVVlB/view?usp=sharing
Pretrained policy: https://drive.google.com/file/d/16hTQUBs1Y4ua9Q69JMWEww2UlSV9t3aq/view?usp=sharing

or 

```shell
pip install gdown
gdown 1UqU3PPLOr9Y4cY9mQNXKK4QkxlnVVVlB
tar -xf assets.tar.xz
rm assets.tar.xz
gdown 16hTQUBs1Y4ua9Q69JMWEww2UlSV9t3aq
tar -xf train_from_demonstr.tar.xz
rm train_from_demonstr.tar.xz 
```

**Step 4.** Run experiments

```shell
python train_RL_PPO.py entropy=false num_episodes_per_rollout=3 total_number_of_training_steps=1000 GUI=false train_from_imitation=true num_epochs=1
```

```shell
python train_RL_PPO.py entropy=true num_episodes_per_rollout=3 total_number_of_training_steps=1000 GUI=false train_from_imitation=true num_epochs=1
```
```shell
python train_RL_PPO.py entropy=true num_episodes_per_rollout=3 total_number_of_training_steps=1000 GUI=false train_from_imitation=false num_epochs=10
```

```shell
python train_RL_PPO.py entropy=false num_episodes_per_rollout=3 total_number_of_training_steps=1000 GUI=false train_from_imitation=false num_epochs=10
```


```shell
python train_RL_DrQ.py augment=false num_seed_steps=1000 num_train_steps=50000 GUI=false train_from_imitation=true replay_buffer_size=50000
```

```shell
python train_RL_DrQ.py augment=false num_seed_steps=1000 num_train_steps=50000 GUI=false train_from_imitation=true replay_buffer_size=50000
```
```shell
python train_RL_DrQ.py augment=true num_seed_steps=100 num_train_steps=50000 GUI=false train_from_imitation=false replay_buffer_size=50000
```

```shell
python train_RL_DrQ.py augment=false num_seed_steps=100 num_train_steps=50000 GUI=false train_from_imitation=false replay_buffer_size=50000
```


```shell
python train_RL_PPO_expert_obs_adv.py entropy=true num_episodes_per_rollout=3 total_number_of_training_steps=1000 GUI=false num_epochs=10
```

```shell
python train_RL_PPO_expert_obs_adv.py entropy=false num_episodes_per_rollout=3 total_number_of_training_steps=1000 GUI=false num_epochs=10
```

```shell
python train_RL_DrQ_expert_obs_adv.py augment=false num_seed_steps=100 num_train_steps=50000 GUI=false replay_buffer_size=50000
```

```shell
python train_RL_DrQ_expert_obs_adv.py augment=false num_seed_steps=100 num_train_steps=50000 GUI=false replay_buffer_size=50000
```
