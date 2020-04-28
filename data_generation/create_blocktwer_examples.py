import matplotlib
import numpy as np
import pybullet_data
import time
import os
import ipdb
from skimage import io
from random import choice, shuffle
import random
import argparse
from itertools import product
import pickle
import gc
import warnings
import pybullet as pb
from PIL import Image
import cv2

# from old_modeling.meter import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Create and render scenarios of block tower.')
parser.add_argument('--height', default=3, type=int, help='number of block to stack')
parser.add_argument('--nb-total-examples', default=100, type=int, help='number of total seq to generate (all masses).')
parser.add_argument('--masses', default='1-10', type=str, help='possible masses for each block')
parser.add_argument('--dir-out', default='/tmp/blocktowerCF', type=str, help='where to store created files')
parser.add_argument('--seed', default=624, type=int, help='seed')
parser.add_argument('--frictions', default='0.5-1', type=str, help='possible frictions coefficients per blocks')
parser.add_argument('--gravity', default='-0.5_0_0.5', type=str,
                    help='possible gravity for X and Y -- only one axis selected')
parser.add_argument('--do-op-remove', default='true', type=str, help='possibility to remove the top block')
parser.add_argument('--nb-confounder-configurations', default=5, type=int,
                    help='number of different confounders configurations -- sampling over all the possible configurations')

# silence warning from https://stackoverflow.com/questions/52165705/how-to-ignore-root-warnings
import imageio.core.util

FPS = 5
NB_SUBSTEPS = 5
NB_SECONDS = 6
NB_SECONDS_FAST = 2

MAX, DELTA = 2., 0.05
W = H = 448  # speed rendering compared to 448
RANGE_UP = [float(f"{x:.2f}") for x in list(np.arange(0.1, MAX + 0.01, DELTA))]
RANGE_DOWN = [float(f"{x:.2f}") for x in list(np.arange(-0.1, -MAX - 0.001, -DELTA))]
RANGE_UP_DOWN = [float(f"{x:.2f}") for x in list(np.arange(-MAX, MAX + 0.01, DELTA))]
RANGE_UP_DOWN.remove(0)  # only for the first time (block 0 and x) otherwise it is redundant
OFF_MIN_X, OFF_MAX_X = -1., 1.  # to make sure it does not fall on the left or right of the screen
OFF_MIN_Y, OFF_MAX_Y = -1.5, 0.  # -3 is in front of the camera
COLORS = ['red', 'green', 'blue', 'yellow']


def silence_imageio_warning(*args, **kwargs):
    pass


imageio.core.util._precision_warn = silence_imageio_warning


def get_masses_permutations(masses, height):
    # 1 mass only
    if len(masses) == 1:
        return [masses * height]

    # Print the permutations
    list_masses_perm = []
    for combo in product(masses, repeat=height):
        list_masses_perm.append(list(combo))

    return list_masses_perm


def list_to_str(int_list, sep='-', type=int):
    int_list = [str(type(x)) for x in int_list]
    string = f"{sep}".join(int_list)

    return string


def get_list_confounders_to_str(list_confounder_config):
    list_mass = list_confounder_config[0]
    list_friction = list_confounder_config[1]
    gravity_xy = list_confounder_config[2]
    return list_to_str(list_mass) + '_' + \
           list_to_str(list_friction, type=float) + '_' + \
           str(float(gravity_xy[0])) + '_' + \
           str(float(gravity_xy[1]))

def str_to_list(str_list, sep='-', type=int):
    """ transform a list of str(int) into a list of int"""

    if len(str_list) == 0 or 'none' in str_list.lower():
        return []
    else:
        l = str_list.split(sep)
        return [type(s) for s in l]


def generate_random_tower(height):
    # From top to base
    z = height - 0.5  # 2.5 or 3.5 for height=3 or 4
    off_x, off_y = np.random.uniform(OFF_MIN_X, OFF_MAX_X), np.random.uniform(OFF_MIN_X, OFF_MAX_X)
    list_pose, list_angle = [], []

    # If the tower should be unstable choose an unstable block
    RANGE_MOOVE = list(np.arange(-0.6, 0.6, 1 / 100)) + [0]

    for idx in range(height):
        # Random rotation
        angle = choice(range(360))
        list_angle.append(angle)

        # Position
        if z + 0.5 == height:  # top block
            x, y = 0., 0.
        else:
            x = x + np.random.choice(RANGE_MOOVE)
            y = y + np.random.choice(RANGE_MOOVE)

        # Point with offset (offset only here because com is invariant to the offset
        list_pose.append((x + off_x, y + off_y, z))
        z -= 1

    # Reverse pose and angle
    list_pose.reverse()
    list_angle.reverse()

    return tuple(list_pose), tuple(list_angle)



def move_one_block(list_pose, idx, coord, delta):
    # tuple to list - immutable to mutable
    list_pose_up = []
    for i, pose in enumerate(list_pose):
        pose_up = list(list_pose[i])
        if idx == i:
            if coord == 'x':
                pose_up[0] += delta
            elif coord == 'y':
                pose_up[1] += delta

        # up
        list_pose_up.append(tuple(pose_up))

    return tuple(list_pose_up)


def set_up_env(fixedTimeStep=1, numSubSteps=10, gravity=(0., 0., -10.), plane_id=0):
    """ Create basic of the environment """

    # Create environment
    try:
        pb.resetSimulation()
    except:
        pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    pb.setGravity(gravity[0], gravity[1], gravity[2])  # X-Y-Z
    pb.setPhysicsEngineParameter(
        fixedTimeStep=fixedTimeStep,
        numSolverIterations=10000,  # need to be high !
        solverResidualThreshold=1e-10,
        numSubSteps=numSubSteps  # need to be high otherwise cubes go away !
    )

    pb.loadURDF(f"./data_generation/urdf/plane_{plane_id}/plane.urdf", useMaximalCoordinates=True)


def set_up_tower(list_pose_xy=None, list_angle=None, list_colors=None,
                 list_mass=None, list_friction=None, is_quaternion_angle=False):
    """ Init the tower to (x,y)=(0,0) - select the order only (i.e. z position) """

    assert len(list_pose_xy) == len(list_angle) == len(list_colors)

    # Loop over the block - from base to top
    list_cube = []
    for i, color in enumerate(list_colors):
        # Param of the block
        angle = list_angle[i]
        z = i + 0.5
        (x, y) = list_pose_xy[i][:2]

        # Cube from urdf
        cube = pb.loadURDF(f"./data_generation/urdf/{color}/cube.urdf",
                           [x, y, z],
                           pb.getQuaternionFromEuler([0, 0, angle]),
                           useMaximalCoordinates=True)

        # Mass and Friction
        if list_friction is None:
            pb.changeDynamics(cube, -1, mass=list_mass[i])
        else:
            pb.changeDynamics(cube, -1, mass=list_mass[i], lateralFriction=list_friction[i])

        # Append
        list_cube.append(cube)

    return list_cube


def get_current_states(list_cube):
    """ return the 3D pose and angle of each block """
    states = np.zeros((len(list_cube), 3 + 4 + 3 + 3))
    for i, cube in enumerate(list_cube):
        # Pose and angle
        pos, angle = pb.getBasePositionAndOrientation(cube)

        # Velocity
        vel_pose, vel_angle = pb.getBaseVelocity(cube)  # linear velocity [x,y,z] and angular velocity [wx,wy,wz]

        # States
        states[i] = list(pos) + list(angle) + list(vel_pose) + list(vel_angle)

    return states


def get_rendering(cameraEyePosition=[0, -7, 4.5], cameraTargetPosition=[0, 0, 1.5], cameraUpVector=[0, 0, 1],
                  fov=60, nearPlane=4, farPlane=20, shadow=1, lightDirection=[1, 1, 1], W=W, H=H):
    """ Rendering of the environment """
    viewMatrix = pb.computeViewMatrix(cameraEyePosition, cameraTargetPosition, cameraUpVector)
    projectionMatrix = pb.computeProjectionMatrixFOV(fov,
                                                     W / H,
                                                     nearPlane,
                                                     farPlane)
    img_arr = pb.getCameraImage(W, H, viewMatrix, projectionMatrix,
                                shadow=shadow,
                                lightDirection=lightDirection,
                                renderer=pb.ER_BULLET_HARDWARE_OPENGL)
    return img_arr


def isStable(seq_states, eps=0.1):
    # Difference in term of z position between t=0 and t=end of the top block
    diff = sum(abs(seq_states[0, :, 2] - seq_states[-1, :, 2]))
    stab = diff < eps
    return stab


def run_simulation(poses, angles, confounders, colors,
                   nb_secs=5, fps=5, nb_substep=5,
                   save=False, out_dir=None):
    # NB steps
    fixed_timestep = float(1 / fps)
    nb_steps = int(fps * nb_secs)

    # Confounders
    masses, frictions, gravity_xy = confounders

    # Create env
    set_up_env(fixed_timestep, nb_substep, gravity=gravity_xy + [-10])
    list_cube = set_up_tower(poses, angles, colors,
                             masses, frictions)

    if save:
        os.makedirs(out_dir, exist_ok=True)

    # Simulate NB_STEPS
    list_rgb = []
    seq_states = np.zeros((nb_steps, len(poses), 3 + 4 + 3 + 3))
    for t in range(nb_steps):
        # Invariant features
        seq_states[t] = get_current_states(list_cube)

        # Visual features
        if save:
            img_arr = get_rendering()
            rgb = img_arr[2][:, :, :3]  # color data RGB - 0->255
            list_rgb.append(rgb)

        # Do one step
        pb.stepSimulation()

    # Save the images and states
    if save:
        writer = cv2.VideoWriter(os.path.join(out_dir, 'rgb.mp4'),
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps,
                                 (rgb.shape[0], rgb.shape[1])
                                 )
        for rgb in list_rgb:
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        writer.release()
        cv2.destroyAllWindows()

    return seq_states


def run_simulation_mutiple_confouders(poses, angles, list_confounders, colors,
                                      nb_secs=5, fps=5, nb_substep=5):
    # Repeat to check the reproducibility
    confounderConfig2stab = {}
    for i, confounders in enumerate(list_confounders):
        stab = isStable(run_simulation(poses, angles, confounders, colors, nb_secs, fps, nb_substep, False, None))
        name = get_list_confounders_to_str(confounders)
        confounderConfig2stab[name] = stab

    return confounderConfig2stab


def main(args):
    # All possible confounder configurations
    list_allMassConfig = get_masses_permutations(str_to_list(args.masses, '-', float), args.height)  # list of list
    list_allFrictionConfig = get_masses_permutations(str_to_list(args.frictions, '-', float),
                                                     args.height)  # list of list
    list_allGravity = get_masses_permutations(str_to_list(args.gravity, '_', float), 2)
    list_colors = list(tuple(COLORS))

    # Another do-operation with block removing - 10%
    do_op_remove = True if args.do_op_remove.lower() == 'true' else False

    # Fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Nb of total examples
    nb_ex = 0
    while nb_ex < args.nb_total_examples:
        print("{}/{}".format(nb_ex, args.nb_total_examples))

        # Random stab indicators
        stab_ab, stab_cd = random.choice([True, False]), random.choice([True, False])

        # Look for a ill-posed AB sequence
        i = 0
        found_ab = False
        while not found_ab:
            # Generate a random A
            list_pose, list_angle = generate_random_tower(args.height)

            # Sample confouder configs to make it faster
            list_sample_confouder_config = []
            for _ in range(args.nb_confounder_configurations):
                masses = random.choice(list_allMassConfig)
                frictions = random.choice(list_allFrictionConfig)
                gravity_xy = random.choice(list_allGravity)
                list_sample_confouder_config.append((masses, frictions, gravity_xy))

            # Shuffle colors
            list_colors = random.sample(COLORS, args.height)

            # Make sure that the AB sequence is ill-posed regarding to the initial geometrical position
            confouderConfig2stab_ab = run_simulation_mutiple_confouders(list_pose,
                                                                        list_angle,
                                                                        list_sample_confouder_config,
                                                                        list_colors,
                                                                        nb_secs=NB_SECONDS_FAST,
                                                                        fps=FPS,
                                                                        nb_substep=NB_SUBSTEPS)
            list_stability_ab = list(confouderConfig2stab_ab.values())
            found_ab = len(set(list_stability_ab)) == 2
            i += 1

        # Find an approriate CD
        found_cd = False
        # Simulate possible counterfactuals by mobing a block by a delta on x or y
        j = choice(list(range(args.height)))
        COORD_RANGE = [['x', RANGE_UP], ['x', RANGE_DOWN], ['y', RANGE_UP], ['y', RANGE_DOWN]]
        random.shuffle(COORD_RANGE)
        if do_op_remove and random.uniform(0, 1) < 0.3:  # 30% of the times
            COORD_RANGE.insert(0, ['rm', [0]])

        for coord, _range in COORD_RANGE:
            k = 0
            all_falldown = False
            while k < len(_range) and not all_falldown and not found_cd:
                delta = _range[k]

                # Modify the position of the block or remove
                if coord == 'rm':
                    list_pose_delta = list_pose[:-1]
                    list_angle_delta = list_angle[:-1]
                    list_sample_confouder_config_delta = [[x[:-1], y[:-1], z] for (x, y, z) in
                                                          list_sample_confouder_config]
                    list_colors_delta = list_colors[:-1]
                else:
                    list_pose_delta = move_one_block(list_pose, j, coord, delta)
                    list_angle_delta = list_angle
                    list_sample_confouder_config_delta = list_sample_confouder_config
                    list_colors_delta = list_colors

                # Run simulation for all mass config and get the stability dict
                confouderConfig2stab_cd = run_simulation_mutiple_confouders(list_pose_delta,
                                                                                list_angle_delta,
                                                                                list_sample_confouder_config_delta,
                                                                                list_colors_delta,
                                                                                nb_secs=NB_SECONDS_FAST,
                                                                                fps=FPS,
                                                                                nb_substep=NB_SUBSTEPS)

                list_stability_cd = list(confouderConfig2stab_cd.values())

                all_falldown = sum(list(confouderConfig2stab_cd.values())) == 0

                if len(set(list(confouderConfig2stab_cd.values()))) == 2 and list_stability_ab != list_stability_cd:
                    list_keys = list(confouderConfig2stab_cd.keys())
                    for idx, key in enumerate(list_keys):
                        if confouderConfig2stab_ab[key] == stab_ab and confouderConfig2stab_cd[key] == stab_cd:
                            found_cd = True
                k += 1

            if found_cd:
                break

        # Finally render AB and CD if found
        if found_cd:
            states_ab = run_simulation(list_pose, list_angle,
                                       list_sample_confouder_config[idx],
                                       list_colors,
                                       nb_secs=NB_SECONDS,
                                       fps=FPS,
                                       nb_substep=NB_SUBSTEPS,
                                       save=True,
                                       out_dir=os.path.join(args.dir_out, str(nb_ex), 'ab'))
            states_cd = run_simulation(list_pose_delta, list_angle_delta,
                                       list_sample_confouder_config_delta[idx],
                                       list_colors_delta,
                                       nb_secs=NB_SECONDS,
                                       fps=FPS,
                                       nb_substep=NB_SUBSTEPS,
                                       save=True,
                                       out_dir=os.path.join(args.dir_out, str(nb_ex), 'cd'))

            # Write metadata
            confounders = np.zeros((len(COLORS), 2))
            states_ab_up = np.zeros((states_ab.shape[0], len(COLORS), 13))
            states_cd_up = np.zeros((states_ab.shape[0], len(COLORS), 13))  # from bottom to top
            for i, col in enumerate(list_colors):  # from bottom to top
                idx_col = COLORS.index(col)
                states_ab_up[:, idx_col] = states_ab[:, i]
                states_cd_up[:, idx_col] = states_cd[:, i]
                confounders[idx_col, 0] = list_sample_confouder_config[idx][0][i]
                confounders[idx_col, 1] = list_sample_confouder_config[idx][1][i]

            np.save(os.path.join(args.dir_out, str(nb_ex), 'ab', 'states.npy'), states_ab)
            np.save(os.path.join(args.dir_out, str(nb_ex), 'cd', 'states.npy'), states_cd)
            np.save(os.path.join(args.dir_out, str(nb_ex), 'confounders.npy'), confounders)
            with open(os.path.join(args.dir_out, str(nb_ex), 'gravity.txt'), 'w') as f:
                f.write("gravity_x={} gravity_y={}".format(list_sample_confouder_config[idx][1][0],
                                                           list_sample_confouder_config[idx][1][1]))

            nb_ex += 1


if __name__ == "__main__":
    # Args
    args = parser.parse_args()
    print(args)
    print("")

    start = time.time()
    main(args)
    print(f"\nTotal time: {time.time() - start:.2f} sec\t")
