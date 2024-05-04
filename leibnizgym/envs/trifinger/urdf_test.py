# isaacgym
from isaacgym import gymtorch
from isaacgym import gymapi
# leibnitzgym
from leibnizgym.utils import *
from leibnizgym.utils.message import *
from leibnizgym.utils.helpers import update_dict
import random
# get default set of parameters
sim_params = gymapi.SimParams()
gym = gymapi.acquire_gym()
# set common parameters
sim_params.dt = 1 / 60
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

# set PhysX-specific parameters
sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0

# set Flex-specific parameters
sim_params.flex.solver_type = 5
sim_params.flex.num_outer_iterations = 4
sim_params.flex.num_inner_iterations = 20
sim_params.flex.relaxation = 0.8
sim_params.flex.warm_start = 0.5

compute_device_id=0
graphics_device_id=0
physics_engine= gymapi.SIM_PHYSX 
# create sim with these parameters
sim = gym.create_sim(compute_device_id, graphics_device_id, physics_engine, sim_params)

# configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0


# create the ground plane
gym.add_ground(sim, plane_params)

    # directory where assets for the simulator are present
_trifinger_assets_dir = os.path.join(get_resources_dir(), "assets", "trifinger")
    # robot urdf (path relative to `_trifinger_assets_dir`)
 #########   _robot_urdf_file = "robot_properties_fingers/urdf/pro/trifingerpro.urdf"
_robot_urdf_file = "robot_properties_fingers/urdf/edu/trifingeredu.urdf"

robot_asset_options = gymapi.AssetOptions()
robot_asset_options.flip_visual_attachments = False
robot_asset_options.fix_base_link = True
robot_asset_options.collapse_fixed_joints = False
robot_asset_options.disable_gravity = False
robot_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
robot_asset_options.thickness = 0.001
robot_asset_options.angular_damping = 0.01
robot_asset_options.use_physx_armature = True
# load tri-finger asset
trifinger_asset = gym.load_asset(sim, _trifinger_assets_dir,
                                        _robot_urdf_file, robot_asset_options)


spacing = 2.0
lower = gymapi.Vec3(-spacing, 0.0, -spacing)
upper = gymapi.Vec3(spacing, spacing, spacing)
env = gym.create_env(sim, lower, upper, 8)

pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 1.0, 0.0)
pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

actor_handle = gym.create_actor(env, trifinger_asset, pose, "MyActor", 0, 1)
# "MyActor":自定义的名称
#参数列表末尾的两个整数是collis_group和collis_filter

# set up the env grid
num_envs = 1
envs_per_row = 8
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# cache some common handles for later use
envs = []
actor_handles = []

# create and populate the environments
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
    envs.append(env)

    height = random.uniform(1.0, 2.5)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, height, 0.0)

    actor_handle = gym.create_actor(env, trifinger_asset, pose, "MyActor", i, 1)
    actor_handles.append(actor_handle)


cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)
# 将弹出一个带有默认尺寸的窗口，可以设置isaacgym.gymapi.CameraProperties调节尺寸

while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)


