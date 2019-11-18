#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataPreparation.py:
"""
import os
import sys
import pandas as pd
import numpy as np
from scipy.signal import peak_widths
import matplotlib.pyplot as plt

from multi_grid.helper_functions.filtering import filter_clusters
from multi_grid.helper_functions.energy_calculation import calculate_energy
from multi_grid.helper_functions.fitting import get_hist, get_fit_parameters_guesses, fit_data
from multi_grid.helper_functions.misc import get_duration, mkdir_p, meV_to_A
from multi_grid.helper_functions.peak_finding import get_peaks
from multi_grid.plotting.analysis.lineshape import get_FoM, calculate_distance_borders
from multi_grid.plotting.analysis.efficiency import get_peak_area

from helium_tube.filtering_he3 import filter_He3
from helium_tube.energy_he3 import calculate_He3_energy


# =============================================================================
#                               PREPARE DATA
# =============================================================================

def prepare_data(origin_voxel, MG_filter_parameters, He3_filter_parameters):
    """
    Data is returned in following order:

    1. Multi-Grid Coated Radial Blades, beam
    2. Multi-Grid Non-Coated Radial Blades, beam
    3. Multi-Grid Coated Radial Blades, background
    4. Multi-Grid Non-Coated Radial Blades, background
    5. He-3, beam
    6. He-3, background

    Within each data-list, data is returned in following order

    1. DataFrame
    2. Energies
    3. Histogram
    4. Bin centers
   (5. Peaks)
   (6. Widths)
   (7. PileUp) <- Only for He-3
   (8. ADCs) <- Only for He-3

    """
    # Declare parameters, such as distance offset and He3 duration
    dirname = os.path.dirname(__file__)
    number_bins = 5000
    start = 0.8  # [meV]
    end = 80  # [meV]
    MG_distance_offsets = [1.5e-3, 0, 1.5e-3, 0]
    He3_distance_offset = 3e-3
    He3_durations = [54304, 58094]
    # Declare heights used as threshold in peak finding algorithm
    heights_MG_coated = [20000, 10000]
    heights_MG_non_coated = [12000, 1000]
    heights_He3 = [20000, 1000]
    heights_vec_MG = [heights_MG_coated, heights_MG_non_coated]
    # Declare file names
    MG_COATED = 'mvmelst_165_191002_111641_Det2_overnight3.h5'
    MG_COATED_BACKGROUND = 'mvmelst_169_191003_075039_Det2_He3InBeam_overnight4.h5'
    MG_NON_COATED = 'mvmelst_135_190930_141618_Det1_overnight2_30x80_14x60.h5'
    MG_NON_COATED_BACKGROUND = 'mvmelst_141_191001_120405_He3InBeam_overnight3.h5'
    HE_3 = '2019_09_HZB_He3InBeam54304s_overnight.h5'
    HE_3_BACKGROUND = '2019_09_HZB_out_of_beam_overnight_58094s.h5'
    MG_file_names = [MG_COATED, MG_NON_COATED, MG_COATED_BACKGROUND, MG_NON_COATED_BACKGROUND]
    He3_file_names = [HE_3, HE_3_BACKGROUND]
    # Declare list to store all data
    full_data = []
    # Store Multi-Grid data
    print('Multi-Grid...')
    for i, file_name in enumerate(MG_file_names):
        path = os.path.join(dirname, '../../../data/Lineshape/%s' % file_name)
        df = pd.read_hdf(path, 'ce')
        df_red = filter_clusters(df, MG_filter_parameters)
        duration = get_duration(df)
        energies = calculate_energy(df_red, origin_voxel, MG_distance_offsets[i])
        hist, bins = get_hist(energies, number_bins, start, end)
        data = [df_red, energies, hist, bins]
        if i < 2:
            # If it is a beam measurement, extract peaks
            peaks = get_peaks(hist, heights_vec_MG[i], number_bins)
            widths, *_ = peak_widths(hist, peaks)
            data.extend([peaks, widths])
        full_data.append(data)
    # Store He-3 data
    print('He-3...')
    for i, (file_name, duration) in enumerate(zip(He3_file_names, He3_durations)):
        path = os.path.join(dirname, '../../../data/Lineshape/%s' % file_name)
        df = pd.read_hdf(path, 'df')
        df_red = filter_He3(df, He3_filter_parameters)
        energies = calculate_He3_energy(df_red, He3_distance_offset)
        hist, bins = get_hist(energies, number_bins, start, end)
        data = [df_red, energies, hist, bins]
        if i < 1:
            # If it is a beam measurement, extract peaks and pile up info
            peaks = get_peaks(hist, heights_He3, number_bins)
            widths, *_ = peak_widths(hist, peaks)
            data.extend([peaks, widths, df_red.PileUp, df_red.ADC])
        full_data.append(data)
    return full_data


# =============================================================================
#                          EXTRACT KEY PARAMETERS
# =============================================================================

def plot_all_peaks(data, label, color, chopper_to_detector_distance):
    # Prepare output paths
    dirname = os.path.dirname(__file__)
    output_folder = os.path.join(dirname, '../../../output/%s/' % label)
    mkdir_p(output_folder)
    # Extract parameters
    energies, hist, bins, peaks, widths = data[1], data[2], data[3], data[4], data[5]
    number_bins = 100
    # Declarea vectors to store data
    peak_energies = []
    FoMs = []
    FoM_uncertainites = []
    peak_areas = []
    peak_area_uncertainties = []
    ADC_ratios = []
    shoulder_vector = []
    # Declare histogram weights, for He-3 we use PileUp + 1 as a weight, else None
    if label == 'He3':
        weights = np.ones(len(energies))#data[6] + 1
        ADCs = data[7]
    else:
        weights = None
    # Declare peak locations and widths
    peak_values = bins[peaks]
    width_values = widths
    # Add last two peaks
    if label == 'He3':
        peak_values = np.append(peak_values, [77, 104])
        width_values = np.append(width_values, [widths[-1]/2, widths[-1]/2])
    elif label == 'MG_Non_Coated':
        peak_values = np.append(peak_values, [77.5, 105])
        width_values = np.append(width_values, [widths[-1]/2, widths[-1]/2])
    # Iterate through all peaks
    for width, peak in zip(width_values, peak_values):
        # Extract fit guesses and peak borders
        left, right = peak-width/20, peak+width/20
        hist_peak, bins_peak = get_hist(energies, number_bins, left, right, weights)
        a_guess, x0_guess, sigma_guess = get_fit_parameters_guesses(hist_peak, bins_peak)
        # Prepare peak within +/- 7 of our estimated sigma
        if peak < 70:
            left_fit, right_fit = (x0_guess - (7 * sigma_guess)), (x0_guess + (7 * sigma_guess))
        else:
            left_fit, right_fit = (x0_guess - (4 * sigma_guess)), (x0_guess + (4 * sigma_guess))
        hist_fit, bins_fit = get_hist(energies, number_bins, left_fit, right_fit, weights)
        fig = plt.figure()
        # Fit data
        a, x0, sigma, x_fit, y_fit, *_ = fit_data(hist_fit, bins_fit, a_guess, x0_guess, sigma_guess)
        # Plot data
        if peak < 70:
            left_plot, right_plot = (x0 - (31 * sigma)), (x0 + (7 * sigma))
            plt.xlim(x0 - (31 * sigma), x0 + (7 * sigma))
        else:
            left_plot, right_plot = (x0 - (10 * sigma)), (x0 + (10 * sigma))
            plt.xlim(x0 - (10 * sigma), x0 + (10 * sigma))
        hist_plot, bins_plot = get_hist(energies, number_bins, left_plot, right_plot, weights)
        plt.errorbar(bins_plot, hist_plot, np.sqrt(hist_plot), fmt='.-', capsize=5, zorder=5, label=label, color=color)
        plt.plot(x_fit, y_fit*(max(hist_plot)/max(y_fit)), label='Gaussian fit', color='black')
        # Plot where the shoulder should be
        reduced_energies, distances = calculate_distance_borders(bins_plot, hist_plot, chopper_to_detector_distance)
        linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
        distances_to_plot = [0.05, 0.10, 0.20, 0.40]
        for distance, linestyle in zip(distances_to_plot, linestyles):
            E_new = reduced_energies[distance]
            plt.axvline(x=E_new, linewidth=2, zorder=10, color='black',
                        linestyle=linestyle,
                        label='Extra distance: %d cm' % (distance*100))
        # Extract FoM
        start, end = reduced_energies[0.20], reduced_energies[0.10]
        bin_width = bins_plot[1] - bins_plot[0]
        FoM, FoM_uncertainity = get_FoM(energies, x0, sigma, start, end, bin_width)
        # Extract FoM between limits 1 cm apart
        FoM_steps = []
        FoM_uncertainity_steps = []
        #for i in np.arange(10, 39, 1):
        #    # Declare distance
        #    distance = distances[i+1]
        #    # Declare start and stop
        #    start = reduced_energies[distances[i]]
        #    stop = reduced_energies[distances[i+1]]
        #    # Get FoM
        #    FoM_step, FoM_uncertainity_step = get_FoM(energies, x0, sigma, start, end, bin_width)
        #    FoM_steps.append(FoM_step)
        #    FoM_uncertainity_steps.append(FoM_uncertainity_step)
        # Extract area
        peak_area, peak_area_uncertainity = get_peak_area(energies, x0, sigma, bin_width, weights)
        # Extract ratio between single events and double events using ADC values
        if label == 'He3':
            peak_indexes = (energies >= (x0 - sigma)) & (energies <= (x0 + sigma))
            ADCs_peak = ADCs[peak_indexes]
            ADC_ratio = len(ADCs_peak[ADCs_peak > 25000])/len(ADCs_peak)
            ADC_ratios.append(ADC_ratio)
        # Save all important values
        peak_energies.append(peak)
        FoMs.append(FoM)
        FoM_uncertainites.append(FoM_uncertainity)
        peak_areas.append(peak_area)
        peak_area_uncertainties.append(peak_area_uncertainity)
        shoulder_vector.append(np.array([FoM_steps, FoM_uncertainity_steps]))
        # Stylise plot
        plt.grid(True, which='major', linestyle='--', zorder=0)
        plt.grid(True, which='minor', linestyle='--', zorder=0)
        plt.title('Peak at: %.2f meV (%.2f Å)' % (peak, meV_to_A(peak)))
        plt.xlabel('Energy [meV]')
        plt.ylabel('Counts')
        plt.yscale('log')
        plt.ylim(1, max(hist_plot)*1.3)
        plt.legend(loc=2)
        # Save plot
        file_name = '%s_Peak_at_%.2f_meV_(%.2f_Å).pdf' % (label, peak, meV_to_A(peak))
        output_path = output_folder + file_name
        fig.savefig(output_path, bbox_inches='tight')
        plt.close()
        # Save Shoulder Investigation
        #fig = plt.figure()
        #plt.errorbar(distances,
        #             FoM_steps,
        #             FoM_uncertainity_steps,
        #             fmt='.-', capsize=5, zorder=5,
        #             color=color,
        #             label=label)
        # Stylise plot
        #plt.grid(True, which='major', linestyle='--', zorder=0)
        #plt.grid(True, which='minor', linestyle='--', zorder=0)
        #plt.title('Peak at: %.2f meV (%.2f Å)' % (peak, meV_to_A(peak)))
        #plt.xlabel('Distance (m)')
        #plt.ylabel('FoM')
        #plt.legend()
        # Save plot
        #file_name = 'Shoulder_Investigation_%s_Peak_at_%.2f_meV_(%.2f_Å).pdf' % (label, peak, meV_to_A(peak))
        #output_path = output_folder + file_name
        #fig.savefig(output_path, bbox_inches='tight')
        #plt.close()

    return peak_energies, FoMs, FoM_uncertainites, peak_areas, peak_area_uncertainties, ADC_ratios, shoulder_vector



# =============================================================================
#                   PREPARE COMPARISON BETWEEN ALL DETECTORS
# =============================================================================

def plot_all_peaks_from_three_data_sets(data_1, label_1, color_1,
                                        data_2, label_2, color_2,
                                        data_3, label_3, color_3):
    # Prepare output paths
    dirname = os.path.dirname(__file__)
    output_folder = os.path.join(dirname, '../../../output/Comparison_%s_and_%s_and_%s/' % (label_1, label_2, label_3))
    mkdir_p(output_folder)
    # Extract parameters
    energies, hist, bins, peaks, widths = data_1[1], data_1[2], data_1[3], data_1[4], data_1[5]
    energies_2 = data_2[1]
    energies_3 = data_3[1]
    number_bins = 100
    # Iterate through all peaks
    for width, peak in zip(widths, peaks):
        # Extract fit guesses and peak borders
        left, right = bins[peak]-width/20, bins[peak]+width/20
        hist_peak, bins_peak = get_hist(energies, number_bins, left, right)
        a_guess, x0_guess, sigma_guess = get_fit_parameters_guesses(hist_peak, bins_peak)
        # Prepare peak within +/- 7 of our estimated sigma
        left_fit, right_fit = (x0_guess - (7 * sigma_guess)), (x0_guess + (7 * sigma_guess))
        hist_fit, bins_fit = get_hist(energies, number_bins, left_fit, right_fit)
        # Fit data
        a, x0, sigma, *_ = fit_data(hist_fit, bins_fit, a_guess, x0_guess, sigma_guess)
        fig = plt.figure()
        fig.set_figheight(5)
        fig.set_figwidth(15)
        plt.subplot(1, 3, 1)
        # Prepare data within +/- 25 of our estimated sigma, we'll use this to plot
        left_plot, right_plot = (x0 - (50 * sigma)), (x0 + (10 * sigma))
        hist_plot, bins_plot = get_hist(energies, number_bins, left_plot, right_plot)
        # Plot from main data
        norm_1 = 1/max(hist_plot)
        plt.errorbar(bins_plot, hist_plot*norm_1, np.sqrt(hist_plot)*norm_1, fmt='.-',
                     capsize=5, zorder=5, label=label_1, color=color_1)
        # Plot from second data
        hist_2, bins_2 = get_hist(energies_2, number_bins, left_plot, right_plot)
        norm_2 = 1/max(hist_2)
        plt.errorbar(bins_2, hist_2*norm_2, np.sqrt(hist_2)*norm_2, fmt='.-',
                     capsize=5, zorder=5, label=label_2, color=color_2)
        # Plot from third data
        hist_3, bins_3 = get_hist(energies_3, number_bins, left_plot, right_plot)
        norm_3 = 1/max(hist_3)
        plt.errorbar(bins_3, hist_3*norm_3, np.sqrt(hist_3)*norm_3, fmt='.-',
                     capsize=5, zorder=5, label=label_3, color=color_3)
        # Plot where the shoulder should be
        E_reduced, distances = calculate_distance_borders(bins_plot, hist_plot)
        linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
        distances_to_plot = [0.05, 0.10, 0.20, 0.40]
        for distance, linestyle in zip(distances_to_plot, linestyles):
            E_new = E_reduced[distance]
            plt.axvline(x=E_new, linewidth=2, zorder=10, color='black', linestyle=linestyle,
                        label='Extra distance: %d cm' % (distance*100))
        # Stylise plot
        plt.grid(True, which='major', linestyle='--', zorder=0)
        plt.grid(True, which='minor', linestyle='--', zorder=0)
        plt.title('Peak at: %.2f meV (%.2f Å)' % (bins[peak], meV_to_A(bins[peak])))
        plt.xlabel('Energy [meV]')
        plt.ylabel('Counts (Normalized to maximum)')
        plt.xlim(x0 - (50 * sigma), x0 + (10 * sigma))
        plt.yscale('log')
        plt.legend(loc=2)
        # Plot FoM as a function of distance
        plt.subplot(1, 3, 2)
        FoMs_1 = get_FoM_vector(energies, number_bins, left_fit, right_fit, distances, E_reduced)
        FoMs_2 = get_FoM_vector(energies_2, number_bins, left_fit, right_fit, distances, E_reduced)
        FoMs_3 = get_FoM_vector(energies_3, number_bins, left_fit, right_fit, distances, E_reduced)
        plt.grid(True, which='major', linestyle='--', zorder=0)
        plt.grid(True, which='minor', linestyle='--', zorder=0)
        plt.title('FoM vs additional travel distance')
        plt.xlabel('Distance (cm)')
        plt.ylabel('FoM')
        print(FoMs_1[0])
        print(FoMs_1[1])
        print(FoMs_1[2])
        print(len(FoMs_1[0]))
        print(len(FoMs_1[1]))
        print(len(FoMs_1[2]))
        plt.errorbar(FoMs_1[0], FoMs_1[1], FoMs_1[2], fmt='.-', label=label_1,
                     color=color_1, capsize=5, zorder=5)
        plt.errorbar(FoMs_2[0], FoMs_2[1], FoMs_2[2], label=label_2,
                     color=color_2, fmt='.-', capsize=5, zorder=5)
        plt.errorbar(FoMs_3[0], FoMs_3[1], FoMs_3[2], label=label_3,
                     color=color_3, fmt='.-', capsize=5, zorder=5)
        plt.legend(loc=1)
        # Plot relevant ratios
        plt.subplot(1, 3, 3)
        plt.plot(FoMs_1[0], FoMs_1[1]/FoMs_2[1],
                 label='%s/%s' % (label_1, label_2),
                 marker='o', linestyle='-', zorder=5, color='black')
        #plt.plot(FoMs_1[0], FoMs_1[1]/FoMs_3[1],
        #         label='%s/%s' % (label_1, label_3),
        #         marker='o', linestyle='-', zorder=5)
        #plt.plot(FoMs_1[0], FoMs_2[1]/FoMs_3[1],
        #         label='%s/%s' % (label_2, label_3),
        #         marker='o', linestyle='-', zorder=5)
        plt.grid(True, which='major', linestyle='--', zorder=0)
        plt.grid(True, which='minor', linestyle='--', zorder=0)
        plt.title('Fractional FoM vs additional travel distance')
        plt.xlabel('Distance (cm)')
        plt.ylabel('Fractional FoM')
        plt.legend(loc=1)
        # Save plot
        file_name = 'Peak_at_%.2f_meV_(%.2f_Å).pdf' % (bins[peak], meV_to_A(bins[peak]))
        output_path = output_folder + file_name
        fig.savefig(output_path, bbox_inches='tight')
        plt.close()



# =============================================================================
#                        GET FOM FOR DIFFERENT DISTANCES
# =============================================================================

def get_FoM_vector(energies, number_bins, left, right, distances, reduced_energies):
    hist, bins = get_hist(energies, number_bins, left, right)
    a_guess, x0_guess, sigma_guess = get_fit_parameters_guesses(hist, bins)
    left_fit, right_fit = (x0_guess - (7 * sigma_guess)), (x0_guess + (7 * sigma_guess))
    hist_fit, bins_fit = get_hist(energies, number_bins, left_fit, right_fit)
    a, x0, sigma, *_ = fit_data(hist_fit, bins_fit, a_guess, x0_guess, sigma_guess)
    bin_width = bins_fit[1] - bins_fit[0]
    # Extract FoM between limits 1 cm apart
    FoM_steps = []
    FoM_uncertainity_steps = []
    indices = np.arange(10, 39, 1)
    print(distances)
    for i in indices:
        # Declare distance
        distance = distances[i+1]
        # Declare start and stop
        start = reduced_energies[distances[i+1]]
        stop = reduced_energies[distances[i]]
        print('start: %f' % start)
        print('stop: %f' % stop)
        print('sigma: %f' % sigma)
        print('x0: %f' % x0)
        # Get FoM
        FoM_step, FoM_uncertainity_step = get_FoM(energies, x0, sigma, start, stop, bin_width)
        FoM_steps.append(FoM_step)
        FoM_uncertainity_steps.append(FoM_uncertainity_step)
    print(distances[indices]*100)
    print(np.array(FoM_steps))
    print(np.array(FoM_uncertainity_steps))
    return [distances[indices]*100, np.array(FoM_steps), np.array(FoM_uncertainity_steps)]
