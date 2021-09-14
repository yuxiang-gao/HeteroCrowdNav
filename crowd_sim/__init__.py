from gym.envs.registration import register

register(
    id="CrowdSim-v0",
    entry_point="crowd_sim.envs:CrowdSim",
)


register(
    id="CrowdEnv-v0",
    entry_point="crowd_sim.envs:CrowdEnv",
)
