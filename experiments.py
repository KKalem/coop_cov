#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# Ozer Ozkahraman (ozero@kth.se)


import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
import numpy as np

import yaml, sys, time, uuid

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from dataclasses import dataclass

from mission_plan import MissionPlan
from drift_model import DriftModel
from auv_agent import RunnableMission


@dataclass
class Experiment:
    # exp configs
    seed: int = None
    num_agents: int = None
    num_landmarks: int = 0
    plan_type: int = None
    area_width: float = None
    area_height: float = None
    swath: int = 50
    k: float = 0.1
    kept_uncertainty: float = 0.5
    heading_noise_degrees: float = 10
    comm_range: int = 50
    landmark_detection_range: int = 50
    lane_overlap: int = 5
    row_overlap: int = 5
    drift_spirals: int = 10
    use_summary: bool = True
    # exp results, should come from mission.results
    missed_area: float = None
    total_covered_area: float = None
    intended_area: float = None
    intended_covered_area: float = None
    plan_is_complete: bool = None
    missed_lenwidths: list = None
    total_travel: float = None
    total_agent_time: float = None
    final_translational_errors: list = None
    # meta
    experiment_finished: bool = False
    experiment_attempted: bool = False
    runtime: int = -1
    finished_time: int = -1
    filename: str = None
    experiment_number: int = 0

    def save(self):
        if self.filename is None:
            random_str = str(uuid.uuid4().hex)
            self.filename = f"experiments/exp_config_{self.finished_time}_{self.seed}_{self.num_agents}_{random_str}.yaml"
        with open(self.filename, 'w') as f:
            yaml.dump(self.__dict__, f)


    def load(self, filename):
        self.filename = filename
        with open(filename, 'rb') as f:
            d = yaml.load(f, Loader=Loader)
        self.__dict__.update(d)


    def as_row(self):
        return zip(*self.__dict__.items())


    def run_and_save(self):
        mplan = MissionPlan(
            plan_type = self.plan_type,
            num_agents = self.num_agents,
            swath = self.swath,
            rect_width = self.area_width,
            rect_height = self.area_height,
            speed = 1.5,
            uncertainty_accumulation_rate_k = self.k,
            kept_uncertainty_ratio_after_loop = self.kept_uncertainty,
            heading_noise_degrees = self.heading_noise_degrees,
            turning_rad = 5,
            comm_range = self.comm_range,
            landmark_range = self.landmark_detection_range,
            overlap_between_lanes = self.lane_overlap,
            overlap_between_rows = self.row_overlap
        )

        drift_model = DriftModel(
            num_spirals = self.drift_spirals,
            num_ripples = 0,
            area_xsize = self.area_width,
            area_ysize = self.area_height,
            xbias = 0,
            ybias = 0,
            scale_size = 1,
            seed = self.seed
        )


        mission = RunnableMission(
            dt = 0.05,
            seed = self.seed,
            mplan = mplan,
            drift_model = drift_model,
            use_summary = self.use_summary
        )


        xs = np.random.randint(0, self.area_height, self.num_landmarks)
        ys = np.random.randint(0, self.area_width, self.num_landmarks)
        landmarks = list(zip(xs,ys))
        mission.add_landmarks(landmarks)


        t0 = time.time()
        self.experiment_attempted = True
        try:
            mission.run()
            t1 = time.time()
            self.runtime = int(t1-t0)
            self.finished_time = int(t1)
            self.experiment_finished = True
            self.__dict__.update(mission.results)
            print(f"Experiment {self.experiment_number} DONE")
        except:
            self.experiment_finished = False
            print(f"Experiment {self.experiment_number} FAILED")

        self.save()





if __name__ == '__main__':

    try:
        __IPYTHON__
        plt.ion()
    except:
        pass

    # seeds = np.random.randint(0,100000,25)
    # seeds = np.random.randint(0,100000,2)
    # seeds used in the previous batch for comm_range=50.
    # initially randomly selected with above
    seeds = [6722, 12565, 17368, 19998, 30095, 30311, 32014,
             39063, 40507, 41451, 44693, 49771, 51286, 60028, 72701,
             72998, 77084, 79373, 80817, 81332, 94483, 94925, 95033,
             98081, 99982]
    sides = [100, 600, 1200]
    # sides = [100, 600]
    # nums_agents = [1, 3, 6]
    nums_agents = [3, 6]
    ks = [0.01, 0.05, 0.1]
    # ks = [0.05, 0.1]
    plan_types = [MissionPlan.PLAN_TYPE_SIMPLE, MissionPlan.PLAN_TYPE_DUBINS]
    # plan_types = [MissionPlan.PLAN_TYPE_DUBINS]
    # kept_uncertainties = [0.25, 0.5, 0.75, 1.0]
    # kept_uncertainties = [0.25, 0.5, 1.0]
    kept_uncertainties = [1.0]
    # nums_landmarks = [0, 1, 5]
    # nums_landmarks = [0, 1]
    nums_landmarks = [0]


    exps = []
    i = 0
    for w_side in sides:
        for h_side in sides:
            for num_agents in nums_agents:
                for k in ks:
                    for num_landmarks in nums_landmarks:
                        for plan_type in plan_types:
                            if plan_type == MissionPlan.PLAN_TYPE_DUBINS:
                                for kept_uncertainty in kept_uncertainties:
                                    for seed in seeds:
                                        e = Experiment(seed = seed,
                                                       comm_range = 0,
                                                       num_agents = num_agents,
                                                       num_landmarks = num_landmarks,
                                                       plan_type = plan_type,
                                                       area_width = w_side,
                                                       area_height = h_side,
                                                       k = k,
                                                       kept_uncertainty = kept_uncertainty,
                                                       experiment_number = len(exps))
                                        exps.append(e)
                            else:
                                for seed in seeds:
                                        e = Experiment(seed = seed,
                                                       comm_range = 0,
                                                       num_agents = num_agents,
                                                       num_landmarks = num_landmarks,
                                                       plan_type = plan_type,
                                                       area_width = w_side,
                                                       area_height = h_side,
                                                       k = k,
                                                       experiment_number = len(exps))
                                        exps.append(e)


    avg_runtime = 300
    num_cores = 12
    estimated_runtime = np.round(len(exps) * avg_runtime / 60 / 60 /  num_cores, decimals=2)
    print(f"Num experiments to run={len(exps)}")
    if estimated_runtime < 1:
        print(f"Estimated runtime:{estimated_runtime*60}mins = {estimated_runtime}hrs")
    else:
        print(f"Estimated runtime:{estimated_runtime}hrs = {estimated_runtime/24}days")


    i = input("Continue? [Y/n]")
    if i == 'n':
        sys.exit(0)

    def run(exp):
        exp.run_and_save()

    from multiprocessing import Pool
    with Pool(processes=12) as p:
        p.map(run, exps)



