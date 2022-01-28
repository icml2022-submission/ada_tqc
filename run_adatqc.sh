python main.py --exp_name=halfcheetah_bullet --env=HalfCheetahBulletEnv-v0 --batch_size=256 --log_dir=data --max_timesteps=5000000 --n_nets=2 --top_quantiles_to_drop_per_net=1 --Q_G_eval_interval=10 --Q_G_n_episodes=200 --Q_G_n_per_episode=20 --Q_G_rollout_length=500 --sampling_scheme=uniform --delta_gamma=0.999 --d_update_interval=50000
python main.py --exp_name=ant_bullet --env=AntBulletEnv-v0 --batch_size=256 --log_dir=data --max_timesteps=5000000 --n_nets=2 --top_quantiles_to_drop_per_net=1 --Q_G_eval_interval=10 --Q_G_n_episodes=200 --Q_G_n_per_episode=20 --Q_G_rollout_length=500 --sampling_scheme=uniform --delta_gamma=0.999 --d_update_interval=50000
python main.py --exp_name=walker2d_bullet --env=Walker2dBulletEnv-v0 --batch_size=256 --log_dir=data --max_timesteps=5000000 --n_nets=2 --top_quantiles_to_drop_per_net=1 --Q_G_eval_interval=10 --Q_G_n_episodes=200 --Q_G_n_per_episode=20 --Q_G_rollout_length=500 --sampling_scheme=uniform --delta_gamma=0.999 --d_update_interval=50000
python main.py --exp_name=hopper_bullet --env=HopperBulletEnv-v0 --batch_size=256 --log_dir=data --max_timesteps=3000000 --n_nets=2 --top_quantiles_to_drop_per_net=1 --Q_G_eval_interval=10 --Q_G_n_episodes=200 --Q_G_n_per_episode=20 --Q_G_rollout_length=500 --sampling_scheme=uniform --delta_gamma=0.999 --d_update_interval=50000
python main.py --exp_name=humanoid_bullet --env=HumanoidBulletEnv-v0 --batch_size=256 --log_dir=data --max_timesteps=10000000 --n_nets=2 --top_quantiles_to_drop_per_net=1 --Q_G_eval_interval=10 --Q_G_n_episodes=200 --Q_G_n_per_episode=20 --Q_G_rollout_length=500 --sampling_scheme=uniform --delta_gamma=0.999 --d_update_interval=50000

python main.py --exp_name=halfcheetah --env=HalfCheetah-v3 --batch_size=256 --log_dir=data --max_timesteps=5000000 --n_nets=2 --top_quantiles_to_drop_per_net=1 --Q_G_eval_interval=10 --Q_G_n_episodes=200 --Q_G_n_per_episode=20 --Q_G_rollout_length=500 --sampling_scheme=uniform --delta_gamma=0.999 --d_update_interval=50000
python main.py --exp_name=ant --env=Ant-v3 --batch_size=256 --log_dir=data --max_timesteps=5000000 --n_nets=2 --top_quantiles_to_drop_per_net=1 --Q_G_eval_interval=10 --Q_G_n_episodes=200 --Q_G_n_per_episode=20 --Q_G_rollout_length=500 --sampling_scheme=uniform --delta_gamma=0.999 --d_update_interval=50000
python main.py --exp_name=walker2d --env=Walker2d-v3 --batch_size=256 --log_dir=data --max_timesteps=5000000 --n_nets=2 --top_quantiles_to_drop_per_net=1 --Q_G_eval_interval=10 --Q_G_n_episodes=200 --Q_G_n_per_episode=20 --Q_G_rollout_length=500 --sampling_scheme=uniform --delta_gamma=0.999 --d_update_interval=50000
python main.py --exp_name=hopper --env=Hopper-v3 --batch_size=256 --log_dir=data --max_timesteps=3000000 --n_nets=2 --top_quantiles_to_drop_per_net=1 --Q_G_eval_interval=10 --Q_G_n_episodes=200 --Q_G_n_per_episode=20 --Q_G_rollout_length=500 --sampling_scheme=uniform --delta_gamma=0.999 --d_update_interval=50000
python main.py --exp_name=humanoid --env=Humanoid-v3 --batch_size=256 --log_dir=data --max_timesteps=10000000 --n_nets=2 --top_quantiles_to_drop_per_net=1 --Q_G_eval_interval=10 --Q_G_n_episodes=200 --Q_G_n_per_episode=20 --Q_G_rollout_length=500 --sampling_scheme=uniform --delta_gamma=0.999 --d_update_interval=50000

