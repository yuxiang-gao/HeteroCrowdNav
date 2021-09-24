from gym.envs.registration import register

register(
    id="CrowdSim-v0",
    entry_point="crowd_sim.envs:CrowdSim",
)


register(
    id="CrowdEnv-v0",
    entry_point="crowd_sim.envs:CrowdEnv",
)

register(
    id="CrowdEnv-NoRoles-v0",
    entry_point="crowd_sim.envs:CrowdEnv",
    kwargs={"use_roles": False},
)

register(
    id="CrowdEnv-Continuous-v0",
    entry_point="crowd_sim.envs:CrowdEnv",
    kwargs={"discrete_action": False},
)

register(
    id="CrowdEnv-Continuous-NoRoles-v0",
    entry_point="crowd_sim.envs:CrowdEnv",
    kwargs={"discrete_action": False, "use_roles": False},
)
