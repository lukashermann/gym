import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
import six
import cv2

try:
    import mujoco_py
    from mujoco_py.mjlib import mjlib
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments.
    """

    def __init__(self, model_path, frame_skip):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco_py.MjModel(fullpath)
        self.data = self.model.data
        self.viewer1 = None
        self.viewer2 = None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qpos = self.model.data.qpos.ravel().copy()
        self.init_qvel = self.model.data.qvel.ravel().copy()
        observation, _reward, done, _info = self._step(np.zeros(self.model.nu))
        assert not done
        self.obs_dim = observation.size

        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low, high)

        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)
        self.width = 500
        self.height = 500
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer1_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass
    def viewer2_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def _reset(self):
        mjlib.mj_resetData(self.model.ptr, self.data.ptr)
        ob = self.reset_model()
        if self.viewer1 is not None:
            self.viewer1.autoscale()
            self.viewer1_setup()
        if self.viewer2 is not None:
            self.viewer2.autoscale()
            self.viewer2_setup()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.model.data.qpos = qpos
        self.model.data.qvel = qvel
        self.model._compute_subtree()  # pylint: disable=W0212
        self.model.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.model.data.ctrl = ctrl
        for _ in range(n_frames):
            self.model.step()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer1 is not None:
                self._get_viewer1().finish()
                self.viewer1 = None
            if self.viewer2 is not None:
                self._get_viewer2().finish()
                self.viewer2 = None
            return
        if mode == 'rgb_array':
            #if self.viewer2 is not None:
            #    self.update_cam()
            self._get_viewer1().render()
            data, width, height = self._get_viewer1().get_image()
            img1 = np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1, :, :]
            self._get_viewer2().render()
            print(self.viewer2.cam.pose.head_pos[1])
            data, width, height = self._get_viewer2().get_image()
            img2 = np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1, :, :]
            return img1, img2
        elif mode == 'human':
            self._get_viewer1().loop_once()

    def _get_viewer1(self,visible=False):
        if self.viewer1 is None:
            self.viewer1 = mujoco_py.MjViewer(visible=visible, init_width=self.width, init_height=self.height)
            self.viewer1.start()
            self.viewer1.set_model(self.model)
            self.viewer1_setup()
        return self.viewer1



    def _get_viewer2(self,visible=False):
        if self.viewer2 is None:
            self.viewer2 = mujoco_py.MjViewer(visible=visible, init_width=self.width, init_height=self.height)
            self.viewer2.start()
            self.viewer2.set_model(self.model)
            self.viewer2_setup()
        return self.viewer2

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(six.b(body_name))
        return self.model.data.com_subtree[idx]

    def get_body_comvel(self, body_name):
        idx = self.model.body_names.index(six.b(body_name))
        return self.model.body_comvels[idx]

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(six.b(body_name))
        return self.model.data.xmat[idx].reshape((3, 3))

    def state_vector(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat
        ])

class MujocoPixelWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(MujocoPixelWrapper, self).__init__(env)
        data, width, height = self.get_viewer1().get_image()
        self.observation_space = spaces.Box(0, 255, [height, width, 3])

    def get_viewer1(self):
        return self.env.unwrapped._get_viewer1(visible=False)

    def _observation(self, observation):
        self.get_viewer1().render()
        data, width, height = self.get_viewer1().get_image()
        #print("later",np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1,:,:].mean())
        rgb_img = np.fromstring(data, dtype='uint8').reshape(height, width, 3)[::-1,:,:]

        depth_data, width_depth, height_depth = self.get_viewer1().get_depth_image()
        # depth image has dimesions 500x250, so it has to be resized
        depth_img = np.fromstring(depth_data, dtype='float').reshape(height_depth,width_depth/2)[::-1,:]
        depth_img = cv2.resize(depth_img,(height, width), interpolation = cv2.INTER_NEAREST)
        # return [rgb, depth, low_level observation]
        return [rgb_img, depth_img, observation]

def MujocoPixelEnv(base_env_id):
    return MujocoPixelWrapper(gym.make(base_env_id))
