#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# Ozer Ozkahraman (ozero@kth.se)

import signal
import subprocess
import time
import sys
import os
import glob
import shutil
import json
import numpy as np
np.set_printoptions(formatter={'float': '{: 0.4f}'.format})

from termcolor import colored

from functools import reduce

from toolbox import geometry as geom

from auv import AUV
from auv_agent import AUVAgent, AUVAction
from pose_graph import PoseGraphOptimization, PGO_VertexIdStore
from mission_plan import construct_lawnmower_paths


def sigint_handler(sig, frame):
    print('Stopping')
    global STOP
    STOP = True


def get_next_data_folder(mission_type, data_folder_name='data'):
    cwd = os.getcwd()
    data_dir = os.path.join(cwd, data_folder_name)
    existing_dirs = os.listdir(data_dir)

    exp_folder = f'{mission_type}_{len(existing_dirs)}'
    i = 1
    while exp_folder in existing_dirs:
        exp_folder = f'{mission_type}_{len(existing_dirs)+i}'
        i+=1

    next_folder = os.path.join(data_dir, exp_folder)
    os.mkdir(next_folder)
    return next_folder



if __name__=="__main__":
    try:
        config_file = sys.argv[1]
        print(f"Using config file:{sys.argv[1]}")
    except:
        config_file = 'manual_config.json'
        print("Using manual_config.json")

    with open(config_file, 'r') as f:
        config = json.load(f)


    print("CONFIG:")
    print(config)

    MISSION_TYPE = config.get('mission_type', 'tri')
    seed = config.get('seed', 42)
    std_shift = config.get('std_shift', 1.0)
    std_heading = config.get('std_heading', 1.0)
    exp_folder = config.get('exp_folder', 'manual')
    swath = config.get('swath', 50)
    flower_type = config.get('flower_type', 'circle')
    num_petals = config.get("num_flower_petals", 2)
    num_flower_depth = config.get("num_flower_depth", 2)
    comms_range = config.get('comms_range', 50)
    max_timesteps = config.get('max_timesteps', None)
    enable_gps_when_lost = config.get('enable_gps_when_lost', False)
    BATCH_MODE = config.get('batch_mode', False)
    data_folder_name = config.get('data_folder_name', 'manual')
    enable_merging_pgos = config.get('enable_merging_pgos', False)
    enable_gossip = config.get('enable_gossip', False)
    enable_longrange_when_landed = config.get('enable_longrange_when_landed', False)
    enable_flower_loop_closures = config.get('enable_flower_loop_closures', False)
    enable_measure_all_agents = config.get('enable_measure_all_agents', False)
    num_lawn_auvs = config.get('num_lawn_auvs', 6)
    double_sided_lawn = config.get('double_sided_lawn', True)
    enable_consistent_drift = config.get('enable_consistent_drift', False)
    max_drift_magnitude = config.get('max_drift_magnitude', 0)
    do_triangles_between_flowers = config.get('do_triangles_between_flowers', True)

    try:
        __IPYTHON__
        IPYTHON = True
    except:
        IPYTHON = False



    try:
        full_exp_folder = config.get('exp_folder')
        os.makedirs(full_exp_folder, exist_ok=True)
    except:
        full_exp_folder = get_next_data_folder(MISSION_TYPE, data_folder_name)

    # clear out this debug folder everytime just in case some stuff arent over-written from
    # previous runs. 
    # this is not a problem for batch mode since those create new folders for each seed etc.
    if full_exp_folder == 'debug':
        files = glob.glob(f'{full_exp_folder}/*')
        for f in files:
            os.remove(f)
        print("Deleted stuff in debug/")







    # fix seed
    np.random.seed(seed)

    # so that we can stop early and still save and plot stuff
    STOP = False
    signal.signal(signal.SIGINT, sigint_handler)

    try:
        plt.cla()
    except:
        pass

    consistent_x_drift = 0
    consistent_y_drift = 0
    if enable_consistent_drift:
        consistent_x_drift = max_drift_magnitude*2*(np.random.rand() - 0.5)
        consistent_y_drift = max_drift_magnitude*2*(np.random.rand() - 0.5)


    # time duration in seconds per update
    dt = 0.5
    px_per_m = 0.1
    # in meters
    target_threshold = 3
    auv_length = 1.5
    auv_speed = 1.5
    landmark_detection_range = swath


    if max_timesteps is None:
        # total time to simulate, in seconds
        total_time_hours = 8
        total_time_mins = total_time_hours*60
        total_time = total_time_mins*60
        num_dts = int(total_time/dt)
    else:
        num_dts = max_timesteps

    optimize_pgo_live = True
    summarize_pgo = False


    ########################################
    # LAWNMOWER PARAMS
    ########################################
    if MISSION_TYPE == 'lawn':
        num_lawnmower_hooks = 20
        num_auvs = num_lawn_auvs
        lawnmower_hook_len = 100

        # set this because eh...
        num_auvs_per_coven = 1



    ########################################
    # TRIS PARAMS
    ########################################
    if MISSION_TYPE == 'tri':
        num_coverage_triangles = 10
        num_auvs_per_coven = 2
        num_auvs = 6

        # this is a function of swath and how many swaths
        # of coverage each coven will do
        coven_radius = num_flower_depth*swath






    ########################################
    # EKF PARAMS
    ########################################
    # std. deviations of
    # velocity measurement
    std_vel = 0.05
    # heading measurement
    std_steer = np.radians(.005)
    # range measurement to landmark
    std_range=0.01
    # bearing measurement to landmark
    std_bearing=0.01
    # update some things every N seconds
    periodic_update_period = 1
    periodic_update_period_dts = int(periodic_update_period / dt)





    # kalman filters for continous predictions
    ekf_params = {'wheelbase':auv_length,
                  'std_vel':std_vel,
                  'std_steer':std_steer,
                  'std_range':std_range,
                  'std_bearing':std_bearing,
                  'dt':dt}


    # a common store of vertex ids for the pose graphs.
    # this way, every agent can get unique ids all the time easily
    # makes merging graphs easier
    pgo_id_store = PGO_VertexIdStore()



    #  landmarks = [(pgo_id_store.get_new_id(),[0,0])]
    landmarks = []

    if MISSION_TYPE == 'lawn':
        lawnmower_paths = construct_lawnmower_paths(num_agents = num_auvs,
                                                    num_hooks = num_lawnmower_hooks,
                                                    hook_len = lawnmower_hook_len,
                                                    swath = swath,
                                                    double_sided = double_sided_lawn)
        init_positions = [p[0] for p in lawnmower_paths]

        stacked = np.vstack(lawnmower_paths)
        x_min, y_min = np.min(stacked, axis=0)
        x_max, y_max = np.max(stacked, axis=0)
        x_radius = max(abs(x_min), abs(x_max))
        y_radius = max(abs(y_min), abs(y_max))
        radius = max(x_radius, y_radius)

        map_size = radius*2 + swath*5
        map_xsize = map_size
        map_ysize = map_size




    if MISSION_TYPE == 'tri':
        # create the main mission controller
        # the radius in which we assume a single coven will cover
        warlock = Warlock(coven_radius = coven_radius,
                          num_coverage_triangles = num_coverage_triangles,
                          do_triangles_between_flowers = do_triangles_between_flowers)

        init_positions = warlock.init_coords
        covens = []

        map_size = warlock.mission_radius * 2 + swath*5
        map_xsize = map_size
        map_ysize = map_size


    map_params = {'xsize':map_xsize,
                  'ysize':map_ysize,
                  'px_per_m':px_per_m}

    # a map with some features in it
    env_map = ScalarMap(**map_params)


    if std_shift <= 0.00001:
        optimize_pgo_live = False

    all_auvs = []
    all_agents = []
    for i, init_pos in enumerate(init_positions):
        # the physical, real auvs to be simulated
        auvs = []
        for _ in range(num_auvs_per_coven):
            auv = AUV(auv_id = len(all_auvs),
                      init_pos = np.array(init_pos),
                      init_heading = 0.,
                      swath = swath,
                      forward_speed = auv_speed,
                      auv_length = auv_length,
                      max_turn_angle = 35.,
                      map_params = map_params)
            all_auvs.append(auv)
            auvs.append(auv)


        # the 'brain's of the operation
        agents = [AUVAgent(auv = auv.clone(),
                           landmarks = landmarks,
                           ekf_params = ekf_params,
                           pgo_id_store = pgo_id_store,
                           optimize_pgo_live = optimize_pgo_live,
                           summarize_pgo = summarize_pgo,
                           landmark_detection_range = landmark_detection_range,
                           target_threshold = target_threshold,
                           enable_gps_when_lost = enable_gps_when_lost,
                           enable_gossip = enable_gossip,
                           enable_longrange_when_landed = enable_longrange_when_landed,
                           enable_merging_pgos = enable_merging_pgos,
                           enable_measure_all_agents = enable_measure_all_agents,
                           disable_logs = BATCH_MODE,
                           comms_range = comms_range) for auv in auvs]
        all_agents.extend(agents)

        if MISSION_TYPE == 'tri':
            # create the group controller
            coven = Coven(coven_id = i,
                          agents = agents,
                          num_petals = num_petals,
                          coven_radius = coven_radius,
                          comms_range = comms_range,
                          enable_flower_loop_closures = enable_flower_loop_closures)

            covens.append(coven)


    if MISSION_TYPE == 'tri':
        # and _FINALLY_ a controller to control the multiple covens!
        # BAYYM
        warlock.init_covens(covens)

    if MISSION_TYPE == 'lawn':
        for agent, path in zip(all_agents, lawnmower_paths):
            action = AUVAction()
            for p in path[:-1]:
                action.add_wp(p, 'moving')
            action.add_wp(path[-1], 'done')
            agent.planned_actions.append(action)


    # keep track of the pose error on every tick
    # for all agents
    pose_diffs = [[] for i in all_agents]



    timings = []
    last_agent_states = ''

    t0 = time.time()

    last_update_time = 0
    total_simulation_steps = 0
    for t in range(0,num_dts):
        total_simulation_steps += 1
        if STOP:
            print(colored(f'Stopped manually at {t*dt}s', 'red'))
            break

        t00 = time.time()
        new_agent_states = f'agents:{[a.state for a in all_agents]}====='
        if new_agent_states != last_agent_states:
            print(colored(f'======={t}:{new_agent_states}=======', 'green'))
            last_agent_states = new_agent_states
            last_update_time = time.time()

        if time.time() - last_update_time > 10:
            print(colored(f'======={t}:{new_agent_states}=======', 'green'))
            waiting_times = [a.waiting_clock for a in all_agents]
            print(f'Waiting times:{waiting_times}')
            gps_states = [a.got_gps_when_landed for a in all_agents]
            print(f'Got gps when landed:{gps_states}')
            last_update_time = time.time()

        for agent, auv, pose_diff in zip(all_agents, all_auvs, pose_diffs):

            # update from the compass before anything
            compass_drift = np.random.normal(0,1) # in degrees
            read_heading = auv.heading + compass_drift # in degrees
            agent.read_compass(read_heading)

            # single time step update
            # decide where to turn
            turn = agent.update(dt,
                                env_map = env_map,
                                real_auv = auv,
                                all_agents = all_agents,
                                all_auvs = all_auvs)

            # physically do the move
            # with some errors of course, the agent doesnt know these
            drift_x = np.random.normal(0, std_shift) + consistent_x_drift
            drift_y = np.random.normal(0, std_shift) + consistent_y_drift
            drift_heading = np.random.normal(0, std_heading)
            auv.update(dt,
                       turn_direction=turn,
                       env_map=env_map,
                       drift_x = drift_x,
                       drift_y = drift_y,
                       drift_heading = drift_heading)


            # update the pose graph with odom
            agent.add_odom_edge_to_pgo()


            # periodic updates
            if t % periodic_update_period_dts == 0:
                # external sensing
                agent.add_landmark_edge_to_pgo(auv)
                #  agent.update_ekf()
                #  agent.update_ekf_landmarks(auv)

                # inter-agent interactions
                agent.interact_with_other_agents(all_agents, all_auvs, auv)


            # error collection
            pos_diff = auv.pos - agent.auv.pos
            heading_diff = auv.heading - agent.auv.heading
            pose_diff.append((pos_diff[0], pos_diff[1], heading_diff))



        if MISSION_TYPE == 'tri':
            # also update the group controllers
            # this should not cause any physical state changes
            for coven in covens:
                coven.update()

            warlock.update()

        t11 = time.time()
        timings.append(t11-t00)


        if MISSION_TYPE == 'tri':
            # stop sim if all agents completed their plans
            if warlock.state in ['done', 'lost'] or \
               all([c.state == 'done' for c in covens]):
                break

        if MISSION_TYPE == 'lawn':
            if all([a.state == 'done' for a in all_agents]):
                break


    # report some stats
    total_travel = reduce(lambda x,y: x+y, [auv.total_distance_traveled() for auv in all_auvs])
    average_travel_duration = (total_travel/num_auvs)/auv_speed
    average_travel_ticks = average_travel_duration/dt
    total_travel_ticks = average_travel_ticks * num_auvs

    t1 = time.time()
    print('='*30)
    if MISSION_TYPE == 'tri':
        print(f'====={t}:covens:{[c.state for c in covens]}=====')
    if MISSION_TYPE == 'lawn':
        print(f'====={t}:agents:{[a.state for a in all_agents]}=====')

    print(f'AUVs traveled for {average_travel_duration} seconds = {average_travel_ticks} ticks on average')
    print(colored(f'TOTAL SIMULATED\n\tTICKS:{t}\n\tMINUTES:{t*dt/60.}\n\tTRAVEL:{total_travel}', 'green'))
    print(colored(f'AVERAGE TRAVEL\n\tTICKS:{average_travel_ticks}\n\tMINUTES:{average_travel_duration/60.}\n\tTRAVEL:{total_travel/num_auvs}', 'green'))

    num_buckets = 8
    bucket_size = int(len(timings)/8)
    s = ''
    for bucket_idx in range(num_buckets):
        avg = np.mean(timings[bucket_idx*bucket_size:(bucket_idx+1)*bucket_size])
        s += str(avg)
        s += ','
    print(f'Avg. update timings: {s}')





    ######################
    # PLOT ALL THE THINGS
    ######################
    plotting = True

    #  fig, (ax1, ax2) = plt.subplots(1,2)
    try:
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(1,1)
    except:
        try:
            import matplotlib as mpl
            mpl.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax1 = plt.subplots(1,1)
        except:
            plotting = False

    plt.rcParams['pdf.fonttype'] = 42

    if IPYTHON:
        print("PLT ION")
        plt.ion()

    # to make plots look cute later
    all_plotted_points = []

    # create an image to show the coverage done
    combined_coverage_im = np.zeros_like(all_auvs[0].marked_map._padded_map)


    colors = ['red', 'green', 'cyan', 'purple', 'orange', 'blue']

    # plot the auv paths
    for color, idx, agent, auv in zip(colors, range(len(all_agents)), all_agents, all_auvs):
        pgo = agent.pgo

        print(f'\nAgent_id:{agent.agent_id}, index in list:{idx}')

        p = os.path.join(full_exp_folder, f'{agent.agent_id}_pgo.g2o')
        pgo.save(p)
        print(f'Optimizing pgo with {len(pgo.vertices())} verts')
        pgo.optimize(max_iterations=200)


        print(f'Colored: {color}')
        pos_trace = np.array(auv.pos_trace)
        agent_pos_trace = np.array(agent.auv.pos_trace)
        if plotting:
            ax1.plot(pos_trace[:,0], pos_trace[:,1], alpha=0.5, color=color)
            ax1.plot(agent_pos_trace[:,0], agent_pos_trace[:,1], linestyle='--', color=color, alpha=0.5)

        all_plotted_points.append(pos_trace[:,:2])
        all_plotted_points.append(agent_pos_trace[:,:2])

        p = os.path.join(full_exp_folder, f'{agent.agent_id}_auv_pos_trace.npy')
        np.save(p, pos_trace)
        p = os.path.join(full_exp_folder, f'{agent.agent_id}_agent_pos_trace.npy')
        np.save(p, agent_pos_trace)
        p = os.path.join(full_exp_folder, f'{agent.agent_id}_pgo_ids.npy')
        np.save(p, np.array(agent.odometry_pgo_vertex_ids))


        if not summarize_pgo:
            pgo_points = pgo.get_all_poses(agent.odometry_pgo_vertex_ids)
            #  pgo_heading_vecs = np.array([(np.cos(p[2]), np.sin(p[2])) for p in pgo_points])
            if plotting:
                ax1.plot(pgo_points[:,0], pgo_points[:,1], alpha=0.7, linestyle=':', color=color)
                #  plt.quiver(pgo_points[:,0], pgo_points[:,1],
                           #  pgo_heading_vecs[:,0], pgo_heading_vecs[:,1],
                           #  color=color, alpha=0.7)
            all_plotted_points.append(pgo_points[:,:2])

            # measurement lines
            # just for one agent please
            for edge in agent.measurement_pgo_edges:
                if edge is None:
                    # dafaq?
                    continue
                p1, p2 = edge['poses']
                points = np.array([p1[:2], p2[:2]])
                # dont want a line between disjoint sets of lines
                ax1.plot(points[:,0], points[:,1], alpha=0.5, linestyle='-.', color='k', linewidth='0.2')



        # paint in the coverage map
        combined_coverage_im += auv.marked_map._padded_map
        p = os.path.join(full_exp_folder, f'{agent.agent_id}_auv_padded_map')
        np.save(p, auv.marked_map._padded_map)
        p = os.path.join(full_exp_folder, f'{agent.agent_id}_agent_padded_map')
        np.save(p, agent.auv.marked_map._padded_map)




    if MISSION_TYPE == 'tri':
        low = -map_xsize/2
        high = map_xsize/2
        lowx = low
        lowy = low
        highx = high
        highy = high

    if MISSION_TYPE == 'lawn':
        lowx = -map_xsize/2
        lowy = -map_ysize/2
        highx = map_xsize/2
        highy = map_ysize/2
        plt.xlim(-swath/2, highx)
        plt.ylim(-swath/2, highy)

    if plotting:
        # plot the coverage images
        ax1.imshow(combined_coverage_im.T, extent=(lowx,highx,lowy,highy), origin='lower')



    if plotting:
        # make things square
        plt.title("Trajectories and Comb. coverage")
        ax1.set_aspect('equal', adjustable='box')
        # plt.tight_layout()
        # move the fig to monitor on the right
        #  fig_manager = plt.get_current_fig_manager()
        #  fig_manager.window.move(4000,700)
        #  fig_manager.window.move(100,100)

    # zoom into the action
    all_plotted_points = np.vstack(all_plotted_points)
    minx, miny = np.min(all_plotted_points, axis=0)
    maxx, maxy = np.max(all_plotted_points, axis=0)
    xlim_min = minx-swath
    xlim_max = maxx+swath
    ylim_min = miny-swath
    ylim_max = maxy+swath
    if plotting:
        ax1.set_xlim(minx-swath, maxx+swath)
        ax1.set_ylim(miny-swath, maxy+swath)

    zoom_window = {'xlim_min':xlim_min,
                   'xlim_max':xlim_max,
                   'ylim_min':ylim_min,
                   'ylim_max':ylim_max,
                   'lowx':lowx,
                   'highx':highx,
                   'lowy':lowy,
                   'highy':highy}






    # plot the pose differences in a different graph
    #  plt.title("Errors in X(r), Y(b)")
    #  pose_diffs = np.abs(np.array(pose_diffs))
    #  colors = ['r', 'b', 'g']
    #  for diff in pose_diffs:
        #  for i in range(2):
            #  ax2.plot(range(len(diff[:,i])), diff[:,i], alpha=0.2, c=colors[i])
    #  ax2.set_aspect('auto', adjustable='box')


    #######################
    # SAVE ALL THE THINGS
    #######################

    if plotting:
        p = os.path.join(full_exp_folder, 'plot.pdf')
        plt.savefig(p, dpi=150, bbox_inches='tight')

    p = os.path.join(full_exp_folder, 'map_params.pickle')
    pickle.dump(map_params, open(p, 'wb'))
    p = os.path.join(full_exp_folder, 'zoom_window.pickle')
    pickle.dump(zoom_window, open(p, 'wb'))

    all_gps_points = {}
    for agent in all_agents:
        all_gps_points[agent.agent_id] = agent.gps_points
    p = os.path.join(full_exp_folder, 'gps_points.pickle')
    pickle.dump(all_gps_points, open(p, 'wb'))

    p = os.path.join(full_exp_folder, 'sys_args.pickle')
    pickle.dump(sys.argv, open(p, 'wb'))

    runtime_stats = {'total_simulation_steps':total_simulation_steps}
    runtime_stats['total_travel'] = total_travel
    runtime_stats['total_travel_ticks'] = total_travel_ticks
    runtime_stats['average_travel_ticks'] = average_travel_ticks
    runtime_stats['average_travel_duration'] = average_travel_duration
    for k,v in runtime_stats.items():
        print(colored(f'{k}:{v}', 'blue'))
    p = os.path.join(full_exp_folder, 'runtime_stats.pickle')
    pickle.dump(runtime_stats, open(p, 'wb'))

    shutil.copy('main.py', full_exp_folder)

    if not BATCH_MODE:
        try:
            shutil.copy(config_file, full_exp_folder)
        except shutil.SameFileError:
            # this file is alredy there, no need to copy it over
            pass

    print(colored(f"Saved stuff into:{full_exp_folder}", 'red'))

    # run the measurement stuff
    # subprocess.run(['python3', 'measurements.py', full_exp_folder])
    measure(exp_dir = full_exp_folder)

    if plotting and not BATCH_MODE:
        plt.show()




