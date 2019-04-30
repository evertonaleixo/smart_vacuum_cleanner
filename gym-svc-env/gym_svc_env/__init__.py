from gym.envs.registration import register

register(
    id='svc-v0',
    entry_point='gym_svc_env.envs:SvcEnv',
)
