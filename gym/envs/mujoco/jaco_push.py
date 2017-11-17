import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class JacoPushEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.col_pen = -1
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'jaco_push/jaco_push.xml', 2)
        self.width = 200
        self.height = 200


    def _step(self, a):
        vec = self.get_body_com("jaco_fingertips")-self.get_body_com("target")
        vec_2 = self.get_body_com("target") - self.get_body_com("goal")
        reward_target_dist = - 0.2 * np.linalg.norm(vec)
        reward_ctrl = - 0.1 * np.square(a).sum()
        reward_goal_dist = - np.linalg.norm(vec_2)
        reward = reward_target_dist + reward_ctrl + reward_goal_dist + self.col_pen * self.detect_col_with_target()

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        if np.linalg.norm(vec_2) < 0.02 and (abs(self.model.data.qvel[-4:-2]) < 0.5).all():
            reward +=2
            done = True

        #qpos = np.random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        #qpos = np.array([0,0.5,-3,0,0,0,0,0])
        #self.set_state(qpos, self.model.data.qvel.flat.copy())

        return ob, reward, done, dict(reward_target_dist=reward_target_dist, reward_ctrl=reward_ctrl, reward_goal_dist=reward_goal_dist)

    def detect_col_with_target(self):
        if self.data.ncon > 0:
            for coni in range(self.data.ncon):
                con = self.data.obj.contact[coni]
                # contact with hand
                #if con.geom1 == 13 and con.geom2 in [10,11,12]:
                # contact with arm
                if con.geom1 == 13 and con.geom2 in [7,8,9]:
                    return 1

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
        """qpos[0] = np.random.uniform(-np.pi,np.pi)
        qpos[1] = np.random.uniform(-1,0.5)
        qpos[2] = np.random.uniform(-np.pi,0)
        """
        qpos[0] = np.random.uniform(-np.pi/2-0.1,-np.pi/2+0.1)
        qpos[1] = np.random.uniform(-0.1,0.1)
        qpos[2] = np.random.uniform(-np.pi/2+0.1,-np.pi/2+0.1)
        qpos[3] = -np.pi/2
        qpos[5] = np.random.uniform(-np.pi,np.pi)
        return qpos


    def reset_model(self):
        qpos = self.initialize_random_qpos()

        self.goal = np.array([-0.5,0])
        while True:
            self.target = self.np_random.uniform(low=-.6, high=0.6, size=2)
            if np.linalg.norm(self.goal-self.target) > 0.1 and np.linalg.norm(self.goal-self.target) < 0.2 and np.linalg.norm(self.target) > 0.45 and np.linalg.norm(self.target) < 0.55:
                break
        qpos[-4:-2] = self.target
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qpos [-7:-4] = 0.2
        qvel[-4:-2] = 0

        qpos[-2:]=self.goal
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs_state_space(self):
        theta = self.model.data.qpos.flat[:6]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qvel.flat[:6],
            self.model.data.qpos.flat[-4:-2],
            self.model.data.qpos.flat[-2:],
            self.get_body_com("jaco_fingertips")-self.get_body_com("target"),
            self.get_body_com("goal")-self.get_body_com("target")
        ])
    """def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:6],
            self.model.data.qvel.flat[:6],
            self.get_body_com("jaco_fingertips"),
            self.get_body_com("target")
        ])"""

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:6]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qvel.flat[:6]
        ])
