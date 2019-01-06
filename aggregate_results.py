#!/usr/bin/env python

import os
import argparse

from pathlib import Path
import numpy as np


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


def main(logdir_path):
    logdir = Path(logdir_path)
    results = dict()
    for c in sorted(listdir_nohidden(logdir)):
        min_sol = np.inf
        min_perf = np.inf
        max_sol = -np.inf
        max_perf = -np.inf
        min_avg = list()
        max_avg = list()

        for r in listdir_nohidden(logdir / c):
            solutions = np.load(logdir / c / r / "solutions.npy")
            performances = np.load(logdir / c / r / "performances.npy")
            performances[performances == np.inf] = np.nan
            best = np.nanargmin(performances)
            worst = np.nanargmax(performances)
            idx = np.unravel_index(best, performances.shape)
            idx_worst = np.unravel_index(worst, performances.shape)
            best_perf = performances[idx]
            worst_perf = performances[idx_worst]
            best_ind = solutions[idx]

            min_avg.append(best_perf)
            max_avg.append(worst_perf)

            if best_perf < min_perf:
                min_perf = best_perf
                min_sol = best_ind
            if best_perf > max_perf:
                max_perf = best_perf
                max_sol = best_ind
        results[c] = {
            "min_sol": min_sol,
            "min_perf": min_perf,
            "max_sol": max_sol,
            "max_perf": max_perf,
            "min_avg": np.mean(min_avg),
            "max_avg": np.mean(max_avg)
        }

    print("Function &Min &Max &MinAvg &MaxAvg\\\\")
    print("\\midrule")
    for k, v in results.items():
        print("{0} &{1:.2f} &{2:.2f} &{3:.2f} &{4:.2f}\\\\".format(k, v['min_perf'], v['max_perf'], v['min_avg'],
                                                                   v['max_avg']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MAP-Elites results aggregation')
    parser.add_argument('--logdir', type=str, help='Log directory', default="log/")
    args = parser.parse_args()
    main(args.logdir)
