import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class JacoEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.col_pen = 0#-0.5
        print("h")
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'jaco/jaco.xml', 2)
        self.width = 200
        self.height = 200


    def reward_control(self, a):
        squ_ac = np.square(a)
        clip_ac = np.clip(squ_ac - 0.1, 0, None)
        return -0.1 * clip_ac.sum()

    def _step(self, a):
        vec = self.get_body_com("jaco_fingertips")-self.get_body_com("target")
        reward_threshold = 0.2
        if np.linalg.norm(vec) < reward_threshold:
            reward_dist = 1
        else:
            reward_dist = 0
        reward_ctrl = self.reward_control(a)
        reward = reward_dist + reward_ctrl + self.col_pen * self.detect_col()
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False

        #qpos = np.random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        #qpos = np.array([0,0.5,-3,0,0,0,0,0])
        #self.set_state(qpos, self.model.data.qvel.flat.copy())

        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def detect_col(self):
        if self.data.ncon > 0:
            return 1
        else:
            return 0

    def viewer_setup(self):
        # keep camera fixed in scene
        #self.viewer.cam.trackbodyid = 1
        #self.viewer.cam.distance = 2
        self.viewer.cam.lookat[0] = self.model.data.xpos[2,0]
        self.viewer.cam.lookat[1] = self.model.data.xpos[2,1]
        self.viewer.cam.lookat[2] = self.model.data.xpos[2,2]
        self.viewer.cam.distance = 2
        #print("distance", self.viewer.cam.lookat[0],self.viewer.cam.lookat[1],self.viewer.cam.lookat[2])

    def initialize_random_qpos(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        qpos[0] = np.random.uniform(-np.pi,np.pi)
        qpos[1] = np.random.uniform(-1,0.5)
        qpos[2] = np.random.uniform(-np.pi,0)

        return qpos


    def reset_model(self):
        qpos = self.initialize_random_qpos()
        while True:
            self.goal = self.np_random.uniform(low=-.6, high=.6, size=2)
            if np.linalg.norm(self.goal) > 0.4 and np.linalg.norm(self.goal) < 0.6 and self.goal[1] < 0:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:6]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[-2:],
            self.model.data.qvel.flat[:6],
            self.get_body_com("jaco_fingertips")-self.get_body_com("target")
        ])

    def _get_obs_pixel(self):
        theta = self.model.data.qpos.flat[:6]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qvel.flat[:6]
        ])
