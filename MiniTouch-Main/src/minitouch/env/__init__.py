import gym


gym.envs.register(
    id='Pushing-v0',
    entry_point='minitouch.env.panda.move_cube_easy:MoveCubeEasy',
    max_episode_steps=200,
    kwargs={
        "debug": False,
        "cube_spawn_distance":0.2,
        "sparse_reward_scale": 25,
        "random_side":True,
        "haptics_upper_bound": 50,
        "lf_force": 25,
        "rf_force": 20,
        "random_seed": 0,
        "min_mass":45,
        "max_mass":55
    }
)


gym.envs.register(
    id='PushingDebug-v0',
    entry_point='minitouch.env.panda.move_cube_easy:MoveCubeEasy',
    max_episode_steps=200,
    kwargs={
        "debug": True,
        "cube_spawn_distance":0.2,
        "sparse_reward_scale": 25,
        "random_side":True,
        "haptics_upper_bound": 50,
        "lf_force": 25,
        "rf_force": 20,
        "random_seed": 0,
        "min_mass": 45,
        "max_mass": 55
    }
)


gym.envs.register(
    id='Opening-v0',
    entry_point='minitouch.env.panda.door:DoorEnv',
    max_episode_steps=200,
    kwargs={
        "debug": False,
        "discrete_grasp": True,
        "grasp_threshold": 0.4,
        "threshold_found": 0.58,
        "haptics_upper_bound": 50,
        "lf_force": 25,
        "rf_force": 20,
        "random_seed": 0,
    }
)


gym.envs.register(
    id='OpeningDebug-v0',
    entry_point='minitouch.env.panda.door:DoorEnv',
    max_episode_steps=200,
    kwargs={
        "debug": True,
        "discrete_grasp":True,
        "grasp_threshold": 0.4,
        "haptics_upper_bound": 50,
        "lf_force": 25,
        "rf_force": 20,
        "random_seed": 0,
    }
)


gym.envs.register(
    id='Picking-v0',
    entry_point='minitouch.env.panda.grasp:Grasp',
    max_episode_steps=200,
    kwargs={
        "debug": False,
        "test": True,
        "min_num_cube": 1,
        "max_num_cube": 1,
        "min_scale": 0.65,
        "max_scale": 0.8,
        "min_mass": 7.5,
        "max_mass": 10,
        "randomize_color": False,
        "randomize_cube_pos": True,
        "max_z": 0.15,
        "haptics_upper_bound": 200,
        "lf_force": 350,
        "rf_force": 400,
        "discrete_grasp": False,
        "lift_threshold": 0.05,
        "random_seed": 0,
    }
)


gym.envs.register(
    id='PickingDebug-v0',
    entry_point='minitouch.env.panda.grasp:Grasp',
    max_episode_steps=200,
    kwargs={
        "debug": True,
        "test": True,
        "min_num_cube": 1,
        "max_num_cube": 1,
        "min_scale": 0.65,
        "max_scale": 0.8,
        "min_mass": 7.5,
        "max_mass": 10,
        "randomize_color": False,
        "randomize_cube_pos": True,
        "max_z": 0.15,
        "haptics_upper_bound": 200,
        "lf_force": 350,
        "rf_force": 400,
        "discrete_grasp": False,
        "lift_threshold": 0.05,
        "random_seed": 0,
    }
)

gym.envs.register(
    id='Inserting-v0',
    entry_point='minitouch.env.panda.peg_insertion:Insertion',
    max_episode_steps=200,
    kwargs={
        "debug": False,
        "cube_spawn_distance":0.2,
        "sparse_reward_scale": 25,
        "random_side":True,
        "haptics_upper_bound": 50,
        "lf_force": 400,
        "rf_force": 400,
        "random_seed": 0,
    }
)

gym.envs.register(
    id='InsertingDebug-v0',
    entry_point='minitouch.env.panda.peg_insertion:Insertion',
    max_episode_steps=200,
    kwargs={
        "debug": True,
        "cube_spawn_distance":0.2,
        "sparse_reward_scale": 25,
        "random_side":True,
        "haptics_upper_bound": 50,
        "lf_force": 400,
        "rf_force": 400,
        "random_seed": 0,
    }
)

