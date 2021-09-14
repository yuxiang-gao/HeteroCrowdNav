import enum
import logging
import threading

import gym
from gym import spaces
from gym.spaces import space
import numpy as np
from pandas import DataFrame
import itertools

from pose2d import apply_tf_to_vel, inverse_pose2d, apply_tf_to_pose

from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.state import ObservableState
from crowd_nav.policy.policy_factory import policy_factory

logger = logging.getLogger(__name__)
gym.logger.set_level(40)


class CrowdEnv(gym.Env):
    metadata = {"render.modes": ["human", "traj", "video"]}

    def __init__(self, config, phase="test") -> None:
        super(CrowdEnv, self).__init__()
        self.phase = phase
        self.config = config
        self.sim = None
        self.robot = None
        self.total_steps = 0
        self.steps_since_reset = None
        self.episode_reward = None
        self.obstacles_as_agent = None

        self.sim, self.robot = self._make_env()
        self.sim.reset()
        self.obstacles_as_agent = self._obstacle_observation()

        self.kinematics = self.config("policy", "action_space", "kinematics")
        self.speed_samples = self.config(
            "policy", "action_space", "speed_samples"
        )
        self.rotation_samples = self.config(
            "policy", "action_space", "rotation_samples"
        )
        self.rotation_constraint = np.pi / 3
        self.num_human = self.config("env", "agents", "human_num")
        x_lim, y_lim = self.config("env", "scenarios", "map_size")

        # self.action_space = spaces.Box(
        #     low=-1, high=1, shape=(2,), dtype=np.float32
        # )  # vel, rot
        self.action_space = spaces.Discrete(81)
        self._build_action_space(1.2)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=((sum(self.num_human) + len(self.obstacles_as_agent)), 15),
            dtype=np.float32,
        )  # gx, gy, vx, vy, v_pref, theta, radius, px1, py1, vx1,vy1, radius1, role_onehot (obstacles, waiter, guest)

        self.viewer = None

    def reset(self):
        self.steps_since_reset = 0
        self.episode_reward = 0
        self.sim.reset(self.phase)
        obs = self._convert_obs()
        return obs

    def step(self, action):
        self.steps_since_reset += 1
        self.total_steps += 1
        time_step = self.sim.time_step
        # print([(h.id, h.px, h.py, h.gx, h.gy) for h in self.sim.humans])

        if (
            isinstance(action, tuple)
            or isinstance(action, list)
            or isinstance(action, np.ndarray)
        ):
            if self.robot.kinematics == "holonomic":
                input_action = ActionXY(action[0], action[1])
            else:
                input_action = ActionRot(action[0], action[1] * np.pi)
        else:
            input_action = self.action_spaces[action]

        _, reward, done, info = self.sim.step(input_action)
        obs = self._convert_obs()
        self.episode_reward = reward

        return obs, reward, done, info

    def onestep_lookahead(self, action):
        return self.sim.step(action, update=False)

    def get_time_step(self):
        return self.sim.time_step

    def _make_env(self):
        env = gym.make("CrowdSim-v0")
        env.configure(self.config("env"))

        robot = env.robot

        policy = policy_factory["dummy"]()
        policy.configure(self.config)
        policy.set_env(policy)
        robot.set_policy(policy)
        robot.print_info()
        return env, robot

    def _convert_obs(self, flatten=False):
        agent_onehot_encoder = np.eye(3).tolist()
        robot = self.sim.robot
        humans = self.sim.humans
        robot_state = (
            robot.get_full_state()
        )  # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta'
        gx, gy, vx, vy, vr = self._convert_agent_obs(robot_state)
        robot_ob = [
            gx,
            gy,
            vx,
            vy,
            robot_state.theta,
            robot_state.v_pref,
            robot_state.radius,
        ]

        obs = []
        for agent in humans:
            #'px1', 'py1', 'vx1', 'vy1', 'radius1'
            human_ob = [
                agent.px - robot_state.px,
                agent.py - robot_state.py,
                agent.vx - robot_state.vx,
                agent.vy - robot_state.vy,
                agent.radius,
            ] + agent_onehot_encoder[agent.type_idx]

            obs.append(robot_ob + human_ob)
        for obstacle in self.obstacles_as_agent:
            obstacle_ob = [
                obstacle.px - robot_state.px,
                obstacle.py - robot_state.py,
                obstacle.vx - robot_state.vx,
                obstacle.vy - robot_state.vy,
                obstacle.radius,
            ] + agent_onehot_encoder[0]
            obs.append(robot_ob + obstacle_ob)
        if flatten:
            obs = np.array(obs).flatten()
        else:
            obs = np.array(obs)
        return obs

    def _obstacle_observation(self):
        obstacles_as_pedestrians = []
        for i, ob in enumerate(self.sim.obstacle_vertices):
            if abs(ob[0][1] - ob[2][1]) == abs(ob[0][0] - ob[2][0]):
                px = (ob[0][0] + ob[2][0]) / 2.0
                py = (ob[0][1] + ob[2][1]) / 2.0
                radius = (ob[0][0] - px) * np.sqrt(2)
                obstacles_as_pedestrians.append(
                    ObservableState(px, py, 0, 0, radius)
                )
            else:
                py = (ob[0][1] + ob[2][1]) / 2.0
                radius = (ob[0][1] - py) * np.sqrt(2)
                px = ob[1][0] + radius
                while px <= ob[0][0]:
                    obstacles_as_pedestrians.append(
                        ObservableState(px, py, 0, 0, radius)
                    )
                    px = px + 2 * radius
        return obstacles_as_pedestrians

    @staticmethod
    def _convert_agent_obs(agent):
        baselink_in_world = np.array([agent.px, agent.py, agent.theta])
        world_in_baselink = inverse_pose2d(baselink_in_world)
        agent_vel_in_world = np.array([agent.vx, agent.vy, 0])
        agent_vel_in_baselink = apply_tf_to_vel(
            agent_vel_in_world, world_in_baselink
        )
        goal_in_world = np.array([agent.gx, agent.gy, 0])
        goal_in_baselink = apply_tf_to_pose(goal_in_world, world_in_baselink)
        agent_state_obs = np.hstack(
            [goal_in_baselink[:2], agent_vel_in_baselink]
        )
        return agent_state_obs

    def _build_action_space(self, v_pref):
        """
        Action space consists of 25 uniformly sampled actions in permitted range and 25 randomly sampled actions.
        """
        holonomic = True if self.kinematics == "holonomic" else False
        speeds = [
            (np.exp((i + 1) / self.speed_samples) - 1) / (np.e - 1) * v_pref
            for i in range(self.speed_samples)
        ]
        if holonomic:
            rotations = np.linspace(
                0, 2 * np.pi, self.rotation_samples, endpoint=False
            )
        else:
            rotations = np.linspace(
                -self.rotation_constraint,
                self.rotation_constraint,
                self.rotation_samples,
            )

        action_space = [ActionXY(0, 0) if holonomic else ActionRot(0, 0)]
        for rotation, speed in itertools.product(rotations, speeds):
            if holonomic:
                action_space.append(
                    ActionXY(speed * np.cos(rotation), speed * np.sin(rotation))
                )
            else:
                action_space.append(ActionRot(speed, rotation))
        self.action_spaces = action_space

    def render(
        self,
        mode="human",
        close=False,
        lidar_scan_override=None,
        goal_override=None,
        save_to_file=False,
        show_score=False,
        show_label=False,
        robocentric=False,
    ):
        if close:
            if self.viewer is not None:
                self.viewer.close()
            return
        if mode in ["traj", "video"]:
            self.sim.render(mode)
        elif mode in ["human"]:
            # Window and viewport size
            WINDOW_W, WINDOW_H = (256, 256)
            SCALE = 2
            VP_W = WINDOW_W * SCALE
            VP_H = WINDOW_H * SCALE
            from gym.envs.classic_control import rendering
            import pyglet
            from pyglet import gl

            # Create viewer
            if self.viewer is None:
                self.viewer = rendering.Viewer(
                    WINDOW_W * SCALE, WINDOW_H * SCALE
                )
                self.transform = rendering.Transform()
                self.transform.set_scale(10 * SCALE, 10 * SCALE)
                self.transform.set_translation(
                    WINDOW_W * SCALE / 2, WINDOW_H * SCALE / 2
                )
                self.score_label = pyglet.text.Label(
                    "0000",
                    font_size=12,
                    x=20,
                    y=WINDOW_H * 2.5 / 40.00,
                    anchor_x="left",
                    anchor_y="center",
                    color=(255, 255, 255, 255),
                )
                self.human_label = []
                for n, human in enumerate(self.sim.humans):
                    self.human_label.append(
                        pyglet.text.Label(
                            str(human.id) + "-" + human.role,
                            x=0,
                            y=0,
                            font_size=12,
                            anchor_x="center",
                            anchor_y="center",
                            color=(255, 255, 255, 255),
                        )
                    )
                #                 self.transform = rendering.Transform()
                self.currently_rendering_iteration = 0
                self.image_lock = threading.Lock()

            def make_circle(c, r, res=10):
                thetas = np.linspace(0, 2 * np.pi, res + 1)[:-1]
                verts = np.zeros((res, 2))
                verts[:, 0] = c[0] + r * np.cos(thetas)
                verts[:, 1] = c[1] + r * np.sin(thetas)
                return verts

            # Render in pyglet
            with self.image_lock:
                self.currently_rendering_iteration += 1
                self.viewer.draw_circle(r=10, color=(0.3, 0.3, 0.3))
                win = self.viewer.window
                win.switch_to()
                win.dispatch_events()
                win.clear()
                gl.glViewport(0, 0, VP_W, VP_H)
                # colors
                bgcolor = np.array([0.4, 0.8, 0.4])
                obstcolor = np.array([0.3, 0.3, 0.3])
                goalcolor = np.array([1.0, 1.0, 0.3])
                goallinecolor = 0.9 * bgcolor
                nosecolor = np.array([0.3, 0.3, 0.3])
                lidarcolor = np.array([1.0, 0.0, 0.0])
                agentcolor = np.array([0.0, 1.0, 1.0])
                # Green background
                gl.glBegin(gl.GL_QUADS)
                gl.glColor4f(bgcolor[0], bgcolor[1], bgcolor[2], 1.0)
                gl.glVertex3f(0, VP_H, 0)
                gl.glVertex3f(VP_W, VP_H, 0)
                gl.glVertex3f(VP_W, 0, 0)
                gl.glVertex3f(0, 0, 0)
                gl.glEnd()
                # Transform
                rx = self.sim.robot.px
                ry = self.sim.robot.py
                rth = self.sim.robot.theta
                if robocentric:
                    # sets viewport = robocentric a.k.a T_sim_in_viewport = T_sim_in_robocentric
                    from pose2d import inverse_pose2d

                    T_sim_in_robot = inverse_pose2d(np.array([rx, ry, rth]))
                    # T_robot_in_robocentric is trans(128, 128), scale(10), rot(90deg)
                    # T_sim_in_robocentric = T_sim_in_robot * T_robot_in_robocentric
                    rot = np.pi / 2.0
                    scale = 20
                    trans = (WINDOW_W / 2.0, WINDOW_H / 2.0)
                    T_sim_in_robocentric = [
                        trans[0]
                        + scale
                        * (
                            T_sim_in_robot[0] * np.cos(rot)
                            - T_sim_in_robot[1] * np.sin(rot)
                        ),
                        trans[1]
                        + scale
                        * (
                            T_sim_in_robot[0] * np.sin(rot)
                            + T_sim_in_robot[1] * np.cos(rot)
                        ),
                        T_sim_in_robot[2] + rot,
                    ]
                    self.transform.set_translation(
                        T_sim_in_robocentric[0], T_sim_in_robocentric[1]
                    )
                    self.transform.set_rotation(T_sim_in_robocentric[2])
                    self.transform.set_scale(scale, scale)
                #                     self.transform.set_scale(20, 20)
                self.transform.enable()  # applies T_sim_in_viewport to below coords (all in sim frame)
                # # Map closed obstacles ---
                for poly in self.sim.obstacle_vertices:
                    gl.glBegin(gl.GL_LINE_LOOP)
                    gl.glColor4f(obstcolor[0], obstcolor[1], obstcolor[2], 1)
                    for vert in poly:
                        gl.glVertex3f(vert[0], vert[1], 0)
                    gl.glEnd()

                # Agent body
                for n, agent in enumerate([self.sim.robot] + self.sim.humans):
                    px = agent.px
                    py = agent.py
                    angle = agent.theta
                    r = agent.radius
                    # Agent as Circle
                    poly = make_circle((px, py), r)
                    gl.glBegin(gl.GL_POLYGON)
                    if n == 0:
                        color = np.array([1.0, 1.0, 1.0])
                    else:
                        color = agentcolor
                    gl.glColor4f(color[0], color[1], color[2], 1)
                    for vert in poly:
                        gl.glVertex3f(vert[0], vert[1], 0)
                    gl.glEnd()
                    # Direction triangle
                    xnose = px + r * np.cos(angle)
                    ynose = py + r * np.sin(angle)
                    xright = px + 0.3 * r * -np.sin(angle)
                    yright = py + 0.3 * r * np.cos(angle)
                    xleft = px - 0.3 * r * -np.sin(angle)
                    yleft = py - 0.3 * r * np.cos(angle)
                    gl.glBegin(gl.GL_TRIANGLES)
                    gl.glColor4f(nosecolor[0], nosecolor[1], nosecolor[2], 1)
                    gl.glVertex3f(xnose, ynose, 0)
                    gl.glVertex3f(xright, yright, 0)
                    gl.glVertex3f(xleft, yleft, 0)
                    gl.glEnd()
                # Goal
                xgoal = self.sim.robot.gx
                ygoal = self.sim.robot.gy
                r = self.sim.robot.radius
                if goal_override is not None:
                    xgoal, ygoal = goal_override
                # Goal markers
                gl.glBegin(gl.GL_TRIANGLES)
                gl.glColor4f(goalcolor[0], goalcolor[1], goalcolor[2], 1)
                triangle = make_circle((xgoal, ygoal), r, res=3)
                for vert in triangle:
                    gl.glVertex3f(vert[0], vert[1], 0)
                gl.glEnd()
                # Goal line
                gl.glBegin(gl.GL_LINE_LOOP)
                gl.glColor4f(
                    goallinecolor[0], goallinecolor[1], goallinecolor[2], 1
                )
                gl.glVertex3f(rx, ry, 0)
                gl.glVertex3f(xgoal, ygoal, 0)
                gl.glEnd()
                # --
                self.transform.disable()
                # Text
                self.score_label.text = ""
                if show_score:
                    self.score_label.text = "R {}".format(self.episode_reward)
                self.score_label.draw()
                if show_label:
                    for n, human in enumerate(self.sim.humans):
                        gl.glPushMatrix()
                        gl.glTranslatef(
                            WINDOW_W * SCALE / 2, WINDOW_H * SCALE / 2, 0.0
                        )
                        gl.glRotatef(0.0, 0.0, 0.0, 1.0)
                        self.human_label[n].x = human.px * 10 * SCALE
                        self.human_label[n].y = human.py * 10 * SCALE
                        self.human_label[n].draw()
                        gl.glPopMatrix()
                # # human labels
                # for n, human in enumerate(self.sim.humans):
                #     self.human_label[n].draw()
                win.flip()
                if save_to_file:
                    pyglet.image.get_buffer_manager().get_color_buffer().save(
                        "/tmp/navreptrainenv{:05}.png".format(self.total_steps)
                    )
                return self.viewer.isopen
        else:
            raise NotImplementedError

    def close(self):
        self.render(close=True)

    def _get_viewer(self):
        return self.viewer


if __name__ == "__main__":
    from crowd_sim.envs.utils.env_player import EnvPlayer
    from crowd_sim.envs.utils.config import Config
    from crowd_sim.envs.utils.logging import logging_init

    config = Config("crowd_nav/configs/configs_hetero.toml")

    env = CrowdEnv(config)
    player = EnvPlayer(env, render_mode="human")
