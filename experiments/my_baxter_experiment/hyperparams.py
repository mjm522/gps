# To get started, copy over hyperparams from another experiment.
# Visit rll.berkeley.edu/gps/hyperparams.html for documentation.
""" Hyperparameters for MJC peg insertion trajectory optimization. """
from __future__ import division

from datetime import datetime
import os.path
import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION
from gps.gui.config import generate_experiment_info


SENSOR_DIMS = {
    JOINT_ANGLES: 19,  #Change  #total 19 joints for baxter
    JOINT_VELOCITIES: 19, #Change 
    END_EFFECTOR_POINTS: 6, #Change  #end effector pos
    ACTION: 19, #Change  #19 actuators 
}
#Change 2 
BXTR_GAINS = np.array([1e-10, #head
	                   1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, #right arm
	                   1e-10, 1e-10, #right gripper
	                   1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, #left arm
	                   1e-10,  1e-10]) #left gripper

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/my_baxter_experiment/' #Change 


common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 1,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentMuJoCo,
    'render' : False,
    'filename': '/Users/michaelmathew/GoogleDrive/mjpro131/otherModels/baxter/baxter.xml', #Change 
    'x0': np.concatenate([np.zeros(19), 
                          np.zeros(19), 
                          np.array([0.82822119,-1.02534485,0.320976, np.pi/4.0, np.pi/2.0,0.0])]), #Change 
    'dt': 0.05,
    'substeps': 5,
    'conditions': common['conditions'],
    'pos_body_idx': np.array([]),
    'pos_body_offset': np.array([]),
    'T': 1,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, 
                      JOINT_VELOCITIES, 
                      END_EFFECTOR_POINTS], #Change 
    'obs_include': [],
    'camera_pos': np.array([2.0, 4.0, 3., -0.725, -0.9, 0.0]), #Chnage 
}

algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
    'iterations': 1,
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains':  1.0 / BXTR_GAINS,
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 1.0,
    'stiffness': 1.0,
    'stiffness_vel': 0.5,
    'dt': agent['dt'],
    'T': agent['T'],
}

torque_cost = {
    'type': CostAction,
    'wu': 5e-5 / BXTR_GAINS,
}

fk_cost = {
    'type': CostFK,
    'target_end_effector': np.array([0.41337288, -0.49105372,  0.80980883, np.pi/4, np.pi/2, 0.0]), #Change 8
    'wp': np.array([1, 1, 1, 1, 1, 1]),
    'l1': 0.1,
    'l2': 10.0,
    'alpha': 1e-5,
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [torque_cost, fk_cost], #Change
    'weights': [1.0, 1.0], #Change
}

algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {}

config = {
    'iterations': algorithm['iterations'],
    'num_samples': 5,
    'verbose_trials': 5,
    'common': common,
    'agent': agent,
    'gui_on': False,
    'algorithm': algorithm,
}

common['info'] = generate_experiment_info(config)
