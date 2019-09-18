#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" main.py: Module containing the GUI used for data analysis.
"""

# Standard library
import os
import sys
# QT
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic
# Data Analysis
import pandas as pd
import numpy as np
# File handling
from FileHandling.Import import unzip_data, import_data
from FileHandling.Cluster import cluster_data
from FileHandling.Storage import save_data, load_data
# Helper functions
from HelperFunctions.CreateMapping import create_full_mapping
from HelperFunctions.Filtering import filter_clusters, get_filter_parameters
# PHS
from Plotting.PHS.PHS_1D import PHS_1D_plot
from Plotting.PHS.PHS_2D import PHS_2D_plot
from Plotting.PHS.PHS_Wires_Vs_Grids import PHS_wires_vs_grids_plot
# Coincidences
from Plotting.Coincidences.Coincidences_2D import coincidences_2D_plot
from Plotting.Coincidences.Coincidences_3D import coincidences_3D_plot
from Plotting.Coincidences.Coincidences_Projections import coincidences_projections_plot
# Misc
from Plotting.Misc.Multiplicity import multiplicity_plot
from Plotting.Misc.ToF import ToF_histogram
from Plotting.Misc.Timestamp import timestamp_plot
# Analysis
from Plotting.Analysis.DeltaE import energy_transfer_plot
from Plotting.Analysis.CountRate import calculate_count_rate

# =============================================================================
# Windows
# =============================================================================

class MainWindow(QMainWindow):
    def __init__(self, app, parent=None):
        super(MainWindow, self).__init__(parent)
        dir_name = os.path.dirname(__file__)
        title_screen_path = os.path.join(dir_name, '../Windows/mainwindow.ui')
        self.ui = uic.loadUi(title_screen_path, self)
        self.app = app
        # Clustering attributes
        self.data_sets = ''
        self.ILL_buses = [-1, -1, -1]
        self.maximum_file_size_in_mb = 3000
        self.adc_threshold = 0
        # Cluster properties
        self.measurement_time = 0
        self.ce = pd.DataFrame()
        self.e = pd.DataFrame()
        self.fill_MG_information_window()
        # He3 attributes
        self.He3_data_sets = ''
        self.He3_measurement_time = 0
        self.He3_counts = 0
        self.fill_He3_information_window()
        self.show()
        self.refresh_window()

    # =========================================================================
    # File handling
    # =========================================================================

    def cluster_action(self):
        zip_paths = QFileDialog.getOpenFileNames(self, "", "../Data")[0]
        if len(zip_paths) > 0:
            self.set_clustering_parameters()
            # Import
            data = ()
            data_sets_temp = '<br/>'
            for i, zip_path in enumerate(zip_paths):
                file_name = zip_path.rsplit('/', 1)[-1]
                data_sets_temp += file_name + '<br/>'
                file_path = unzip_data(zip_path)
                data += import_data(file_path, self.maximum_file_size_in_mb)
                os.remove(file_path)
                print('%d/%d' % (i+1, len(zip_paths)))
            # Cluster
            clusters, events = cluster_data(data, self.ILL_buses, self.adc_threshold)
            # Write or append
            if self.write.isChecked():
                self.ce = clusters
                self.e = events
                self.data_sets = data_sets_temp
            else:
                self.ce = self.ce.append(clusters)
                self.e = self.e.append(events)
                self.data_sets += data_sets_temp
                # Reset index
                self.ce.reset_index(drop=True, inplace=True)
                self.e.reset_index(drop=True, inplace=True)
            # Update window
            self.measurement_time = get_duration(self.ce)
            self.fill_MG_information_window()
            self.refresh_window()
            # Print statements for debugging purposes
            print(self.ce)
            print(self.e)

    def save_action(self):
        path = QFileDialog.getSaveFileName()[0]
        if path != '':
            save_data(path, self.ce, self.e, self.data_sets, self.adc_threshold, self.ILL_buses)

    def load_action(self):
        path = QFileDialog.getOpenFileName(self, "", "../Data")[0]
        if path != '':
            clusters, events, data_sets_temp, adc_threshold_temp, ILL_buses_temp = load_data(path)
            # Write or append
            if self.write.isChecked():
                self.ce = clusters
                self.e = events
                self.data_sets = data_sets_temp
                self.adc_threshold = adc_threshold_temp
                self.ILL_buses = ILL_buses_temp
            else:
                self.ce = self.ce.append(clusters)
                self.e = self.e.append(events)
                self.data_sets += data_sets_temp
                # Reset index
                self.ce.reset_index(drop=True, inplace=True)
                self.e.reset_index(drop=True, inplace=True)
            # Update window
            self.measurement_time = get_duration(self.ce)
            self.fill_MG_information_window()
            self.refresh_window()


    # =========================================================================
    # Plotting
    # =========================================================================

    # ==== PHS ==== #

    def PHS_1D_action(self):
        if self.data_sets != '':
            filter_parameters = get_filter_parameters(self)
            ce_filtered = filter_clusters(self.ce, filter_parameters)
            number_bins = int(self.phsBins.text())
            fig = PHS_1D_plot(self.e, ce_filtered, number_bins)
            fig.show()


    def PHS_2D_action(self):
        if self.data_sets != '':
            fig = PHS_2D_plot(self.e)
            fig.show()

    def PHS_wires_vs_grids_action(self):
        if (self.data_sets != ''):
            filter_parameters = get_filter_parameters(self)
            ce_filtered = filter_clusters(self.ce, filter_parameters)
            fig = PHS_wires_vs_grids_plot(ce_filtered)
            fig.show()

    # ==== Coincidences ==== #

    def Coincidences_2D_action(self):
        if self.data_sets != '':
            filter_parameters = get_filter_parameters(self)
            ce_filtered = filter_clusters(self.ce, filter_parameters)
            fig, histograms = coincidences_2D_plot(ce_filtered, self.measurement_time)
            # Export histograms to text
            dir_name = os.path.dirname(__file__)
            output_path = os.path.join(dir_name, '../Output/')
            for bus, histogram in enumerate(histograms):
                path = output_path + '2D_Coincidences_Bus_%d.txt' % bus
                np.savetxt(path, histogram, fmt="%d", delimiter=",")
            # Plot figure
            fig.show()


    def Coincidences_3D_action(self):
        if self.data_sets != '':
            filter_parameters = get_filter_parameters(self)
            ce_filtered = filter_clusters(self.ce, filter_parameters)
            coincidences_3D_plot(ce_filtered)

    def Coincidences_Projections_action(self):
        if self.data_sets != '':
            filter_parameters = get_filter_parameters(self)
            ce_filtered = filter_clusters(self.ce, filter_parameters)
            fig, histograms = coincidences_projections_plot(ce_filtered)
            # Export histograms to text
            dir_name = os.path.dirname(__file__)
            output_path = os.path.join(dir_name, '../Output/')
            file_names = ['Front', 'Top', 'Side']
            for file_name, histogram in zip(file_names, histograms):
                path = output_path + '2D_Coincidences_Projections_%s.txt' % file_name
                np.savetxt(path, histogram, fmt="%d", delimiter=",")
            # Plot figure
            fig.show()

    # ==== Misc ==== #

    def Multiplicity_action(self):
        if (self.data_sets != ''):
            filter_parameters = get_filter_parameters(self)
            ce_filtered = filter_clusters(self.ce, filter_parameters)
            fig = multiplicity_plot(ce_filtered)
            fig.show()

    def ToF_action(self):
        if self.data_sets != '':
            filter_parameters = get_filter_parameters(self)
            ce_filtered = filter_clusters(self.ce, filter_parameters)
            number_bins = int(self.tofBins.text())
            fig = ToF_histogram(ce_filtered, number_bins)
            fig.show()

    def Timestamp_action(self):
        if (self.data_sets != ''):
            filter_parameters = get_filter_parameters(self)
            ce_filtered = filter_clusters(self.ce, filter_parameters)
            fig = timestamp_plot(ce_filtered)
            fig.show()

    # ==== Analysis ==== #

    def Energy_Transfer_action(self):
        if (self.data_sets != '') and (self.Ei_value != -1):
            filter_parameters = get_filter_parameters(self)
            ce_filtered = filter_clusters(self.ce, filter_parameters)
            number_bins = int(window.dE_bins.text())
            fig = energy_transfer_plot(ce_filtered, self.Ei, number_bins)
            fig.show()

    def Count_Rate_action(self):
        if (self.data_sets != ''):
            filter_parameters = get_filter_parameters(self)
            ce_filtered = filter_clusters(self.ce, filter_parameters)
            ToF_values = ce_filtered.ToF * 62.5e-9
            count_rate = calculate_count_rate(ToF_values, self.measurement_time)
            print('Count rate: %.1f [Hz]' % count_rate)




    # ========================================================================
    # Helper Functions
    # ========================================================================

    def setup_buttons(self):
        # File handling
        self.cluster_button.clicked.connect(self.cluster_action)
        self.save_button.clicked.connect(self.save_action)
        self.load_button.clicked.connect(self.load_action)
        # PHS
        self.PHS_1D_button.clicked.connect(self.PHS_1D_action)
        self.PHS_2D_button.clicked.connect(self.PHS_2D_action)
        self.PHS_wires_vs_grids_button.clicked.connect(self.PHS_wires_vs_grids_action)
        # Misc
        self.multiplicity_button.clicked.connect(self.Multiplicity_action)
        self.ToF_button.clicked.connect(self.ToF_action)
        self.timestamp_button.clicked.connect(self.Timestamp_action)
        # Coincidences
        self.Coincidences_2D_button.clicked.connect(self.Coincidences_2D_action)
        self.Coincidences_3D_button.clicked.connect(self.Coincidences_3D_action)
        self.Coincidences_Projections_button.clicked.connect(self.Coincidences_Projections_action)
        # Analysis
        self.dE_button.clicked.connect(self.Energy_Transfer_action)
        self.count_rate_button.clicked.connect(self.Count_Rate_action)
        # Button toogle
        self.toogle_VMM_MG()

    def refresh_window(self):
        self.app.processEvents()
        self.update()
        self.app.processEvents()
        self.update()
        self.app.processEvents()
        self.app.processEvents()
        self.app.processEvents()

    def fill_MG_information_window(self):
        information_text = '<b>Measurement time:</b> %d [s]' % int(self.measurement_time)
        information_text += "<br/><b>ADC Threshold:</b> %d [ADC Ch's]" % self.adc_threshold
        information_text += '<br/><b>ILL buses:</b> ' + str(self.ILL_buses)
        information_text += '<br/><b>Data sets:</b> ' + self.data_sets
        self.information_window.setText(information_text)

    def fill_He3_information_window(self):
        information_text = '<b>Measurement time:</b> %d [s]' % int(self.He3_measurement_time)
        information_text += "<br/><b>Counts:</b> %d [Counts]" % self.He3_counts
        information_text += '<br/><b>Data sets:</b> ' + self.He3_data_sets
        self.He3_information_window.setText(information_text)

    def set_clustering_parameters(self):
        self.ILL_buses = [self.ILL_bus_1.value(), self.ILL_bus_2.value(), self.ILL_bus_3.value()]
        self.maximum_file_size_in_mb = float(self.maximum_file_size_in_mb_value.text())
        self.adc_threshold = float(self.adc_threshold_value.text())

    def toogle_VMM_MG(self):
        self.ESS_button.toggled.connect(
            lambda checked: checked and self.ILL_button.setChecked(False))
        self.ILL_button.toggled.connect(
            lambda checked: checked and self.ESS_button.setChecked(False))



# =============================================================================
# Helper Functions
# =============================================================================

def append_folder_and_files(folder, files):
    folder_vec = np.array(len(files)*[folder])
    return np.core.defchararray.add(folder_vec, files)

def get_duration(df):
    times = df.Time.values
    diff = np.diff(times)
    resets = np.where(diff < 0)
    duration_in_TDC_channels = sum(times[resets]) + times[-1]
    duration_in_seconds = duration_in_TDC_channels * 62.5e-9
    return duration_in_seconds


# =============================================================================
# Start GUI
# =============================================================================

app = QApplication(sys.argv)
main_window = MainWindow(app)
main_window.setAttribute(Qt.WA_DeleteOnClose, True)
main_window.setup_buttons()
sys.exit(app.exec_())
