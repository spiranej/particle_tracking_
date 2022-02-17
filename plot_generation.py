#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import optimize
import scipy as sc
from scipy.signal import savgol_filter
from matplotlib.collections import LineCollection
import pprint
from trackpy import compute_drift, subtract_drift
from scipy.ndimage.filters import gaussian_filter1d

def x_y_coordinates_to_msd(x, y, frames, micron_pixel_ratio, time, unit_time):
    msds = []
    distances = []
    end_distances = []

    end_vector = np.array([x[-1], y[-1]])
    # print(time, len(x))
    for i in range(time, len(x)):
        # if frames[i] - frames[i-time] == time:
        new_vec = np.array([x[i], y[i]])
        prev_vec = np.array([x[i-time], y[i-time]])
        distance = np.linalg.norm(new_vec - prev_vec)*micron_pixel_ratio
        # distance = np.power((x[i] - x[i-time])**2 + (y[i] - y[i-time])**2, 0.5)*micron_pixel_ratio
        end_distance = np.linalg.norm(new_vec - end_vector) * micron_pixel_ratio
        distances.append(distance)

        result = np.power(distance, 2)
        msds.append(result)
        end_distances.append(end_distance)
        # else:
        #     msds.append(np.nan)

    distances = np.array(distances)
    distances = distances[~np.isnan(msds)]
    msds = np.array(msds)
    msds = msds[~np.isnan(msds)]

    velocities = distances/(time*unit_time)
    velocities_per_minute = velocities*60

    return time*unit_time, msds.mean(), velocities_per_minute, distances.sum(), end_distances


def x_y_coordinates_to_msd_all_time(particle_coordinates, time_max, unit_time, micron_pixel_ratio, samples=50):
    lines = []
    times = []
    velocities = []
    distances = []
    all_distances = []

    for i in np.logspace(0, np.log10(time_max), num=samples, endpoint=False):
    # for i in range(1, time_max):
        i = int(i)
        # print("Initial time", i)
        time, msd, velocity, distance, all_distance = x_y_coordinates_to_msd(
            particle_coordinates[0],
            particle_coordinates[1],
            particle_coordinates[2],
            micron_pixel_ratio,
            i,
            unit_time
        )
        lines.append(msd)
        times.append(time)
        velocities.append(velocity)
        distances.append(distance)
        all_distances.append(all_distance)

    lines = np.array(lines)
    times = np.array(times)
    velocities = np.array(velocities, dtype=object)
    distances = np.array(distances)
    all_distances = np.array(all_distances, dtype=object)

    times = times[~np.isnan(lines)]
    velocities = velocities[~np.isnan(lines)]
    distances = distances[~np.isnan(lines)]
    lines = lines[~np.isnan(lines)]
    return times, lines, velocities, distances, all_distances[0]


def file_to_coordinates(file_location, max_particles, frames_minimum, percentage_jump_frames, drift_correction, exclude):
    # Trajectory, Frame, x, y

    file = pd.read_csv(file_location, header=0)

    if drift_correction:
        file.rename(columns={'Trajectory': 'particle', 'Frame': 'frame'}, inplace=True)
        drift = compute_drift(file, 10)
        plt.figure()
        drift.plot()
        file = subtract_drift(file, drift)
        file.rename(columns={'particle': 'Trajectory', 'frame': 'Frame'}, inplace=True)

    total_particles = file['Trajectory'].max()

    particle_coordinates = []

    index = 1
    while len(particle_coordinates) < max_particles and index <= total_particles:
        if index in exclude:
            index += 1
            continue
        current_particle_frames = file[file['Trajectory']==index]['Frame']
        # total_frames = len(current_particle_frames)
        max_frames = current_particle_frames.max()
        if max_frames is np.nan:
            max_frames = 0
        if max_frames < frames_minimum:
            print("Trajectory (particle) with less than minimum frames", index)
            index += 1
            continue

        truncated_frames = current_particle_frames[:frames_minimum]
        print("Truncated frames (max), (total)", truncated_frames.max(), len(truncated_frames))
        if truncated_frames.max() - len(truncated_frames) > percentage_jump_frames*truncated_frames.max():
            print("Trajectory (particle) missing too many frames", index)
            index += 1
            # for i in range(1, len(truncated_frames)):
                # if truncated_frames[i] > truncated_frames[i-1] + 1:
                    # print(truncated_frames[i])
            continue

        coordinates = file[file['Trajectory'] == index][['x', 'y']]
        x = coordinates['x'].values[:frames_minimum]
        y = coordinates['y'].values[:frames_minimum]
        frames = current_particle_frames.values[:frames_minimum]

        particle_coordinates.append((index, x, y, frames))
        index += 1

    print("Number of particles", len(particle_coordinates))
    return particle_coordinates


def power_fit(times, msds):
    func = lambda t, a, b: a * t ** b
    try:
        fit_val = sc.optimize.curve_fit(func, times, msds, maxfev=1000)
        best_fit_values = [func(x, fit_val[0][0], fit_val[0][1]) for x in times]
    except:
        print("Error fitting")
        return 0, 0, msds
    return fit_val[0][0], fit_val[0][1], best_fit_values


def generate_single_particle_plot(msds, times, title=None, best_fit=True, show_fit=False, color='red'):
    plt.xlabel("Time (seconds)")
    plt.ylabel("MSD $ ({μm}^2)$")
    # plt.title("")

    if best_fit:
        multiplier, power, best_fit_line = power_fit(times, msds)
        plt.loglog(times, best_fit_line, zorder=99999999999999999, linestyle='--', linewidth=1, color='black')
        if show_fit or power == 0:
            fit_explanation = "y={:.2E}".format(multiplier) + r"$*x^{" + "{0:.3f}".format(power) + "}$"
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.text(0.05, 0.95, fit_explanation, fontsize=10, verticalalignment='top', bbox=props,
                     transform=plt.gca().transAxes)

    plt.loglog(times, msds, marker='o', linestyle='-', linewidth=0.25, markersize=2, zorder=1, fillstyle='none', color=color)
    if title is not None:
        plt.title(title)
    plt.yticks(list(plt.yticks())[0])
    if power == 0 or show_fit:
        plt.show()
    return power


def generate_all_particles_plot(msds, times, from_particle, separate_figures=False, color=None, legend_offset=0):
    plt.figure(1)

    all_msds = np.concatenate(msds)
    all_times = np.concatenate(times)

    multiplier, power, best_fit_line = power_fit(all_times, all_msds)
    plt.loglog(all_times, best_fit_line, linestyle='-', linewidth=2, color='black')
    fit_explanation = "y={:.2E}".format(multiplier)+r"$*x^{" + "{0:.3f}".format(power) + "}$"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95-legend_offset, fit_explanation, fontsize=10, verticalalignment='top', bbox=props,
             transform=plt.gca().transAxes)

    figure_number = 1
    for i, (msd, time) in enumerate(zip(msds, times)):
        if separate_figures:
            figure_number += 1
            plt.figure(figure_number)
            generate_single_particle_plot(msd, time, "Particle " + str(from_particle + i + 1), best_fit=True, show_fit=True)
        plt.figure(1)
        generate_single_particle_plot(msd, time, best_fit=True, color=color)


def medusa_plot(all_coordinates, micron_pixel_ratio, file_name): 
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
        
    for coordinate in all_coordinates:
        trajectory_number = coordinate[0]
        x = coordinate[1]
        y = coordinate[2]
        frames = coordinate[3]
    #     print(y)
        y = y - y[0]
        x = x - x[0]
        
        x_1 = np.convolve(x, np.ones(5)/5, mode='valid') * micron_pixel_ratio
        y_1 = np.convolve(y, np.ones(5)/5, mode='valid') * micron_pixel_ratio
        #x_1 = savgol_filter(x, 5, 1) * micron_pixel_ratio
        #y_1 = savgol_filter(y, 5, 1) * micron_pixel_ratio

        dydx = frames

        points = np.array([x_1, y_1]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(dydx.min(), dydx.max())
        lc = LineCollection(segments, cmap='jet', norm=norm, linestyle='dotted', facecolor='none')
        # Set the values used for colormapping
        lc.set_array(dydx)
        lc.set_linewidth(3)
        line = ax1.add_collection(lc)
        #ax1.text(x_1[-1], y_1[-1], str(trajectory_number))
    fig.colorbar(line, ax=ax1)
    #ax1.set_title(file_name)
    ax1.set_xlim([-6, 6])
    ax1.set_ylim([-6, 6])
    ax1.grid()
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    # ax1.axis('equal')
    # ax1.autoscale()
    # plt.show()


def plot_net_displacement(all_coordinates, micron_pixel_ratio, file_name):
    plt.figure()

    net_displacements = []
    print("Net displacements below")
    for coordinate in all_coordinates:
        trajectory_number = coordinate[0]
        x = coordinate[1]
        y = coordinate[2]
        new_vec = np.array([x[0], y[0]])
        prev_vec = np.array([x[-1], y[-1]])
        distance = np.linalg.norm(new_vec - prev_vec)*micron_pixel_ratio
        net_displacements.append(distance)
        print(distance)

    plt.hist(net_displacements, label=file_name[:-5], alpha=0.5, edgecolor='black', color=file_color)
    plt.xlim(left=0, right=3)
    plt.xlabel("Net displacements (μm)")
    plt.ylabel("Counts (particle)")


if __name__ == "__main__":
    # execute only if run as a script

    list_of_files = [(".csv", "red")]
    exclude = []

    drift_correction = True
    frames_number = 180 
    seconds_per_frame = 5 
    micron_pixel_ratio = 0.55 
    percentage_jump_frames = .000005 
    max_particles = 300 
    from_particle = 0
    to_particle = 300
    separate_histograms = False  
    separate_msd_plots = False  
    mean_velocities_histogram = True 
    #per_particle_distances_plot = True


    legend_offset=0
    all_velocities_per_file = []
    for (file_name, file_color) in list_of_files:
        all_coordinates = file_to_coordinates(file_name, max_particles, frames_number, percentage_jump_frames, drift_correction, exclude)[from_particle:to_particle]

        all_velocities = []
        all_times = []
        all_msds = []
        all_distances_0 = []
        all_powers = []
        all_average_speeds = []
        all_average_velocities = []

        for coordinate in all_coordinates:
            x = coordinate[1]
            y = coordinate[2]
            frames = coordinate[3]

            times, msd, velocity, distances, all_distances = x_y_coordinates_to_msd_all_time(
                [x, y, frames],
                frames_number+1,
                unit_time=seconds_per_frame,
                micron_pixel_ratio=micron_pixel_ratio
            )

            power = generate_single_particle_plot(msd, times) #need this to get alpha values
            all_powers.append(power)
            all_average_speeds.append(velocity[0].mean())
            all_average_velocities.append(velocity[-1].mean())
            all_velocities.append(velocity[0])
            all_times.append(times)
            all_msds.append(msd)
            all_distances_0.append(all_distances)


        print("Average velocities below")
        for avg_velocity in all_average_velocities:
            print(avg_velocity)
        print("Average velocities ended")

        print("Average speeds below")
        for avg_speed in all_average_speeds:
            print(avg_speed)
        print("Average speeds ended")

        print("Powers below")
        for power in all_powers:
            print(power)
        print("Powers ended")

        all_msds = np.array(all_msds)
        all_times = np.array(all_times)
        all_velocities = np.array(all_velocities)
        all_velocities_per_file.append(all_velocities)
        medusa_plot(all_coordinates, micron_pixel_ratio, file_name)
        plot_net_displacement(all_coordinates, micron_pixel_ratio, file_name)
        #plot_movement_over_time(all_distances_0, file_name, per_particle_distances_plot)

        #print("Shape", all_msds.shape)
       # generate_all_particles_plot(all_msds, all_times, from_particle, separate_msd_plots, color=file_color, legend_offset=legend_offset)
        legend_offset += 0.075
      
    if len(list_of_files) > 1:
        bins = np.linspace(0, 3, 50)
        if mean_velocities_histogram:
            plt.figure()
            for (all_vel, (file_name, file_color)) in zip(all_velocities_per_file, [file_pair for file_pair in list_of_files]):
                mean_velocities = [vel.mean() for vel in all_vel]
                print(mean_velocities)
                plt.rcParams["axes.labelweight"] = "bold"
                plt.rcParams["font.weight"] = "bold"
                plt.hist(mean_velocities, bins=bins, label=file_name[:-5], alpha=0.5, edgecolor='black', color= file_color)
                plt.xlim(left=0, right=3)
                plt.xlabel("Velocity (μm/min)")
                plt.ylabel("Counts (particle)")
                # ax.tick_params(axis='both', which='major', labelsize=10)
        plt.legend(loc="upper right")
        plt.show()
        exit(0)

    weights = []
    for vel in all_velocities:
        # plt.figure()
        weights.append(np.zeros_like(vel) + 100. / vel.size)
    weights = np.array(weights)
    # print(weights.shape)
    # print(all_velocities.shape)

    plt.figure()
    plt.title("All particles")
    plt.hist(list(all_velocities), weights=list(weights))
    plt.xlim(left=0, right=5)
    plt.xlabel("Velocity (μm/min)")
    plt.ylabel("Frequency (percentage)")

    concatenated_velocities = np.concatenate(all_velocities)
    plt.figure()
    plt.title("All particles")
    plt.hist(concatenated_velocities, weights=np.zeros_like(concatenated_velocities) + 100. / concatenated_velocities.size)
    plt.xlabel("Velocity (μm/min)")
    plt.ylabel("Frequency (percentage)")
    
    if mean_velocities_histogram:
        mean_velocities = [vel.mean() for vel in all_velocities]
        plt.figure()
        plt.title("All particles mean velocities")
        plt.hist(mean_velocities, bins=200)
        plt.xlabel("Velocity (μm/min)")
        plt.ylabel("Counts (particle)")

    if separate_histograms:
        for i, vel in enumerate(all_velocities):
            print("Generating individual histograms")
            plt.figure()
            plt.title("Particle " + str(from_particle+i+1))
            plt.hist(vel, weights=np.zeros_like(vel) + 100. / vel.size)
            plt.xlabel("Velocity (μm/min)")
            plt.ylabel("Frequency (percentage)")
    plt.show()
