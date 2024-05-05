# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# isaacgym-rlgpu
from isaacgym import rlgpu
from rlgpu.utils.config import set_np_formatting, set_seed
# leibniz-gym: dump all environments for loading
from leibnizgym.envs.trifinger import TrifingerEnv as Trifinger
# leibnizgym
from leibnizgym.wrappers.vec_task import VecTaskPython
from leibnizgym.utils.config_utils import load_cfg, get_args
from leibnizgym.utils.errors import InvalidTaskNameError
from leibnizgym.utils.message import *
# rl-games
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from rl_games.common import wrappers
from rl_games.common.algo_observer import AlgoObserver
from rl_games.algos_torch import torch_ext
import torch
import numpy as np
# python
import os
import argparse
import yaml
from datetime import datetime
from copy import deepcopy
from rl_games.algos_torch import model_builder
#from document.leibnizgym.scripts import ppo_continuous_action_isaacgym
import sys
sys.path.append('/data/user/wanqiang/document/leibnizgym/scripts')
import ppo_tt
import cql
import sacn

def parse_vec_task(args: argparse.Namespace, cfg: dict) -> VecTaskPython:
    """Parses the configuration parameters for the environment task.

    TODO (@mayank): Remove requirement for args and make this a normal function
                    inside utils.
    Args:
        args: command line arguments.
        cfg: environment configuration dictionary (task)

    Returns:
        TThe vectorized RL-env wrapped around the task.
    """
    # create native task and pass custom config
    if args.task_type == "Python":
        # check device on which to run agent and environment
        if args.device == "CPU":
            print_info("Running using python CPU...")
            # check if agent is on different device
            sim_device = 'cpu'
            ppo_device = 'cuda:0' if args.ppo_device == "GPU" else 'cpu'
        else:
            print_info("Running using python GPU...")
            sim_device = 'cuda:0'
            ppo_device = 'cuda:0'
        # create the IsaacEnvBase defined using leibnizgym
        try:
            task = eval(args.task)(config=cfg, device=sim_device,
                                   visualize=not args.headless,
                                   verbose=args.verbose)
        except NameError:
            raise InvalidTaskNameError(args.task)
        # wrap environment around vec-python wrapper
        env = VecTaskPython(task, rl_device=ppo_device, clip_obs=5, clip_actions=1)
    else:
        raise ValueError(f"No task of type `{args.task_type}` in leibnizgym.")

    return env

def create_rlgpu_env(**kwargs):
    """
    Creates the task from configurations and wraps it using RL-games wrappers if required.
    """
    #print(kwargs)
    # TODO (@arthur): leibnizgym parse task
    env = parse_vec_task(cli_args, task_cfg)
    # print the environment information
    print_info(env)
    # save environment config into file
    
    env.dump_config(os.path.join(logdir, 'env_config.yaml'))
    # wrap around the environment
    frames = kwargs.pop('frames', 1)
    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)
    return env

def create_rlgpu_env2(**kwargs):
    """
    Creates the task from configurations and wraps it using RL-games wrappers if required.
    """
    #print(kwargs)
    # TODO (@arthur): leibnizgym parse task
    env = parse_vec_task(kwargs['cli_args'], kwargs['gym_cfg'])
    # print the environment information
    print_info(env)
    # save environment config into file
    logdir=kwargs['cli_args']['logdir']
    env.dump_config(os.path.join(logdir, 'env_config.yaml'))
    # wrap around the environment
    frames = kwargs.pop('frames', 1)
    if frames > 1:
        env = wrappers.FrameStack(env, frames, False)
    return env


class RlGamesGpuEnvAdapter(vecenv.IVecEnv):
    """
    Adapter from VecPythonTask to Rl-Games VecEnv.
    """

    def __init__(self, config_name: str, num_actors: int, **kwargs):
        # this basically calls the `create_rlgpu_env()` function for RLGPU environment.
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)
        # check if environment is for asymmetric PPO or not
        self.use_global_obs = (self.env.num_states > 0)
        # get initial observations
        self.full_state = {
            "obs": self.env.reset()
        }
        # get state if assymmetric environment
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()

    """
    Properties
    """

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {
            'num_envs': self.env.num_envs,
            'action_space': self.env.action_space,
            'observation_space': self.env.observation_space
        }
        # print the spaces (for debugging)
        print(">> Action space: ", info['action_space'])
        print(">> Observation space: ", info['observation_space'])
        # check if environment is for asymmetric PPO or not
        if self.use_global_obs:
            info['state_space'] = self.env.state_space
            print(">> State space: ", info['state_space'])
        # return the information about spaces
        return info

    """
    Operations
    """

    def reset(self):
        # reset the environment
        self.full_state["obs"] = self.env.reset()
        # check if environment is for asymmetric PPO or not
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state
        else:
            return self.full_state["obs"]

    def step(self, action):
        # step through the environment
        next_obs, reward, is_done, info = self.env.step(action)
        # check if environment is for asymmetric PPO or not
        # TODO (@arthur): Improve the return only dictinary
        self.full_state["obs"] = next_obs
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state, reward, is_done, [[], info]
        else:
            return self.full_state["obs"], reward, is_done, [[], info]


# register the rl-games adapter to use inside the runner
vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RlGamesGpuEnvAdapter(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {
    'vecenv_type': 'RLGPU',
    'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs),
})


class LeibnizAlgoObserver(AlgoObserver):
    """Allows us to log stats from the env along with the algorithm running stats. """

    def __init__(self):
        pass

    def after_init(self, algo):
        self.algo = algo
        self.game_scores = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.direct_info = {}
        self.writer = self.algo.writer

    def process_infos(self, infos, done_indices):
        if not infos:
            return
        if len(infos) > 0 and isinstance(infos[0], dict):
            for ind in done_indices:
                if len(infos) <= ind // self.algo.num_agents:
                    continue
                info = infos[ind // self.algo.num_agents]
                game_res = None
                if 'battle_won' in info:
                    game_res = info['battle_won']
                if 'scores' in info:
                    game_res = info['scores']

                if game_res is not None:
                    self.game_scores.update(torch.from_numpy(np.asarray([game_res])).to(self.algo.ppo_device))
        if len(infos) > 1 and isinstance(infos[1], dict):  # allow direct logging from env
            self.direct_info = infos[1]

    def after_clear_stats(self):
        self.game_scores.clear()

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.game_scores.current_size > 0:
            mean_scores = self.game_scores.get_mean()
            self.writer.add_scalar('scores/mean', mean_scores, frame)
            self.writer.add_scalar('scores/iter', mean_scores, epoch_num)
            self.writer.add_scalar('scores/time', mean_scores, total_time)
        for k, v in self.direct_info.items():
            self.writer.add_scalar(k, v, frame)

def run_rlg_hydra(hydra_cfg):
    global task_cfg, agent_cfg_train, cli_args, logdir, vargs
    from omegaconf import OmegaConf
    task_cfg = OmegaConf.to_container(hydra_cfg.gym)
    agent_cfg_train = OmegaConf.to_container(hydra_cfg.rlg)
    cli_args= hydra_cfg.args
    logdir = cli_args['logdir']
    vargs = OmegaConf.to_container(cli_args)
    #run_rlg()
    #ppo_continuous_action_isaacgym.train(hydra_cfg)
    if cli_args.offline==False:
        print("start online")
        if cli_args.play==True:
            ppo_tt.play(hydra_cfg)
        else:
            ppo_tt.train(hydra_cfg)
    else:
        print("start offline")
        cql.train(hydra_cfg=hydra_cfg)
        #sacn.train(hydra_cfg=hydra_cfg)


def run_rlg():
    global logdir
    # Create default directories for weights and statistics
    os.makedirs("nn", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    # set numpy formatting for printing only
    set_np_formatting()

    # append the timestamp to logdir
    now = datetime.now()
    now_dir_name = now.strftime("%m-%d-%Y-%H-%M-%S")
    logdir = os.path.join(logdir, now_dir_name)
    os.makedirs(logdir, exist_ok=True)
    # print the common info
    print_notify(f'Saving logs at: {logdir}')
    print_notify(f'Verbosity     : {cli_args.verbose}')
    print_notify(f'Seed          : {agent_cfg_train["seed"]}')
    # set logdir and seed
    cli_args.logdir = logdir
    set_seed(agent_cfg_train["seed"])
    # print training configuration for debugging
    if cli_args.verbose:
        print_info(f'Agent training configuration: ')
        print_dict(agent_cfg_train)
        print(40 * '-')
    # save agent config into file
    with open(os.path.join(logdir, 'agent_config.yaml'), 'w') as file:
        yaml.dump(agent_cfg_train, file)
    
    rl_game=0
    if rl_game==1:
        # convert CLI arguments into dictionory
        # create runner and set the settings
            
        runner = Runner(LeibnizAlgoObserver())
        runner.load(agent_cfg_train)
        runner.reset()
        runner.run(vargs)
    else:
        if cli_args.play==True:
            print('Started to play')
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            agent_cfg = deepcopy(agent_cfg_train)
            params = deepcopy(agent_cfg['params'])
            config=params['config']
            env_config = config.get('env_config', {})
            num_actors = config['num_actors']
            env_name = config['env_name']
            vec_env = None
            env_info = config.get('env_info')
            if env_info is None:
                vec_env = vecenv.create_vec_env(env_name, num_actors, **env_config)
                env_info = vec_env.get_env_info()
            else:
                vec_env = config.get('vec_env', None)
            builder = model_builder.ModelBuilder()
            config['network'] = builder.load(params)
            network = config['network']
            action_space = env_info['action_space']
            
            actions_num = action_space.shape[0]
            actions_low = torch.from_numpy(action_space.low.copy()).float().to(device)
            actions_high = torch.from_numpy(action_space.high.copy()).float().to(device)
            observation_space=env_info['observation_space']
            print(observation_space)
            print(action_space)
            normalize_input = config['normalize_input']
            normalize_value = config.get('normalize_value', False)
            num_agents = env_info.get('agents', 1)
            model_config = {
            'actions_num' : actions_num, #9
            'input_shape' : observation_space.shape,
            'num_seqs' : num_agents, #1
            'value_size': 1,
            'normalize_value': normalize_value,#False,
            'normalize_input': normalize_input,#False,
            } 
            model = network.build(model_config)
            model.to(device)
            model.eval()
            is_rnn = model.is_rnn()
            need_init_rnn=is_rnn
            player_config = config.get('player', {})
            if 'deterministic' in player_config:
                is_deterministic = player_config['deterministic']
            else:
                is_deterministic = player_config.get(
                    'deterministic', True)
            if 'checkpoint' in vargs and vargs['checkpoint'] is not None and vargs['checkpoint'] !='':
                checkpoint = torch_ext.load_checkpoint(vargs['checkpoint'])
                model.load_state_dict(checkpoint['model'])
                env_state = checkpoint.get('env_state', None)
                if vec_env is not None and env_state is not None:
                     vec_env.set_env_state(env_state)


            n_games = 2000
            max_steps = 108000 // 4
            evaluation = player_config.get("evaluation", False)
            update_checkpoint_freq = player_config.get("update_checkpoint_freq", 100)
            # if we run player as evaluation worker this will take care of loading new checkpoints
            dir_to_monitor = player_config.get("dir_to_monitor")
            render_sleep = player_config.get('render_sleep', 0.002)
            # path to the newest checkpoint
            #checkpoint_to_load: Optional[str] = None
            render = False
            
            n_game_life = player_config.get('n_game_life', 1)
            sum_rewards = 0
            sum_steps = 0
            sum_game_res = 0
            n_games = n_games * n_game_life
            games_played = 0
            has_masks = False
            for _ in range(n_games):
                if games_played >= n_games:
                    break

                obses = vec_env.reset()
                obses=obses['obs']
                batch_size = 1
                #batch_size = self.get_batch_size(obses, batch_size)

                if need_init_rnn:
                    #self.init_rnn()
                    need_init_rnn = False

                cr = torch.zeros(batch_size, dtype=torch.float32)
                steps = torch.zeros(batch_size, dtype=torch.float32)
                print(obses)
   

                for n in range(max_steps):

                   
                    input_dict = {
                    'is_train': False,
                    'prev_actions': None, 
                    'obs' : obses,
                    'rnn_states' : None
                    }
                    with torch.no_grad():
                        res_dict = model(input_dict)
                    mu = res_dict['mus']
                    action = res_dict['actions']
                    states = res_dict['rnn_states']
                    if is_deterministic:
                        current_action = mu
                    else:
                        current_action = action

                    action= rescale_actions(actions_low, actions_high, torch.clamp(current_action, -1.0, 1.0))
                    obses, rewards, dones, infos = vec_env.step(action)
                    obses=obses['obs']
                    cr += rewards.cpu()
                    steps += 1

                    if render:
                        vec_env.render(mode='human')
                        time.sleep(render_sleep)
        else:
            agent_cfg = deepcopy(agent_cfg_train)
            params = deepcopy(agent_cfg['params'])
            config=params['config']
            env_config = config.get('env_config', {})
            print('-'*20)
            print("env_config:",env_config)
            num_actors = config['num_actors']
            env_name = config['env_name']
            vec_env = None
            env_info = config.get('env_info')
            if env_info is None:
                vec_env = vecenv.create_vec_env(env_name, num_actors, **env_config)
                env_info = vec_env.get_env_info()
            else:
                vec_env = config.get('vec_env', None)
            #vec_env = vecenv.create_vec_env(env_name, num_actors, **env_config)

            solved_reward = 100000  # stop training if avg_reward > solved_reward
            log_interval = 20  # print avg reward in the interval
            max_episodes = 10000  # max training episodes
            max_timesteps = 750  # max timesteps in one episode

            update_timestep = 4000  # update policy every n timesteps. it is a BATCH
            action_std = 0.5  # constant std for action distribution (Multivariate Normal)
            K_epochs = 80  # update policy for K epochs 80
            eps_clip = 0.2  # clip parameter for PPO
            gamma = 0.99  # discount factor

            lr = 0.0003  # parameters for Adam optimizer
            betas = (0.9, 0.999)
            
            state_dim = 113 #self.env.observation_space.shape[0]
            action_dim = 9  #self.env.action_space.shape[0]
            memory = Memory()
            ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)
            print(lr, betas)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # logging variables
            running_reward = 0
            avg_length = 0
            time_step = 0

            # training loop
            for i_episode in range(1, max_episodes + 1):
                state = vec_env.reset()
                state=state['states'].squeeze().to(device)
                #print(isinstance(state, torch.Tensor),state)
                # 一个episode
                for t in range(max_timesteps):
                    time_step += 1
                    # Running policy_old:
                    action = ppo.select_action(state, memory)  # a_(t)
                    #print(action)
                    action=torch.Tensor(action).unsqueeze(0).to(device)
                    state, reward, done, _ = vec_env.step(action)  # S_(t+1), R_(t+1)
                    state=state['states']
                    # Saving reward and is_terminals:
                    memory.rewards.append(reward)
                    memory.is_terminals.append(done)  # done 是一个bool变量，判断是否到达terminal

                    # update if it's time
                    if time_step % update_timestep == 0:
                        ppo.update(memory)  # LEARNING
                        memory.clear_memory()
                        time_step = 0
                    running_reward += reward
                    #if render:
                    #    env.render()
                    if done:
                        break

                avg_length += t
                #print('Episode {}  \t Avg reward: {}'.format(i_episode,  running_reward))
                # stop training if avg_reward > solved_reward
                if running_reward > (log_interval * solved_reward):
                    print("########## Solved! ##########")
                    # save the model
                    torch.save(ppo.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format(env_name))
                    break

                # save every 500 episodes
                if i_episode % 200 == 0:
                    torch.save(ppo.policy.state_dict(), '/data/user/wanqiang/document/leibnizgym/model/PPO_continuous_{}_{}.pth'.format(env_name,i_episode))

                # logging
                if i_episode % log_interval == 0:
                    avg_length = int(avg_length / log_interval)
                    running_reward = int((running_reward / log_interval))

                    print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
                    running_reward = 0
                    avg_length = 0


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action =  action * d + m
    return scaled_action
# EOF
