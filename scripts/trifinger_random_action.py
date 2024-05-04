"""
@brief      Demo script for checking tri-finger environment.
"""

# leibnizgym
from leibnizgym.utils import *
from leibnizgym.envs import TrifingerEnv
from leibnizgym.utils.torch_utils import unscale_transform
# python
import torch

if __name__ == '__main__':
    # configure the environment
    env_config = {
        'num_instances': 1,
        'aggregrate_mode': True,
        'control_decimation': 10,
        'command_mode': 'position',
        'sim': {
            "use_gpu_pipeline": True,
            "physx": {
                "use_gpu": True
            }
        }
    }
    # create environment
    env = TrifingerEnv(config=env_config, device='cuda', verbose=True, visualize=False)
    _ = env.reset()
    print_info("Trifinger environment creation successful.")
    action=torch.zeros(env.get_action_shape(),dtype=torch.float32,device=env.device)
    # sample run
    b=0
    while True:
        _, _, _, _ = env.step(action)
        # zero action agent
        action_transformed=unscale_transform(
            action,
            lower=env._action_scale.low,
            upper=env._action_scale.high
        )
        # step through physics
        if torch.all(torch.abs(action_transformed-env._dof_position)<0.1):
            print(b)
            b=0
            action = 2 * torch.rand(env.get_action_shape(), dtype=torch.float, device=env.device) - 1
        elif b>10:
            b=0
            print(f"bad_pos{action_transformed}\npos:{env._dof_position}")
            action = 2 * torch.rand(env.get_action_shape(), dtype=torch.float, device=env.device) - 1
        else:
            b+=1

        
        # render environment
        #env.render()

# EOF
