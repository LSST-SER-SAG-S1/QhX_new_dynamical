"""
This module contains unit tests for the ParallelSolver class.
"""

import os
import gc
import threading
from io import StringIO

import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

from QhX.parallelization_solver import ParallelSolver
from QhX import DataManagerDynamical, process1_new_dyn
import unittest

class TestParallelSolver(unittest.TestCase):
    """Unit tests for the ParallelSolver class."""

    def setUp(self):
        """Set up the data manager and synthetic data for testing."""
        print("Running setUp...")  # Debugging print
        # Set up the data manager with the required configuration
        agn_dc_mapping = {
            'column_mapping': {'flux': 'psMag', 'time': 'mjd', 'band': 'filter'},
            'group_by_key': 'objectId',
            'filter_mapping': {0: 0, 1: 1, 2: 2, 3: 3}
        }
        self.data_manager = DataManagerDynamical(
            column_mapping=agn_dc_mapping['column_mapping'],
            group_by_key=agn_dc_mapping['group_by_key'],
            filter_mapping=agn_dc_mapping['filter_mapping']
        )
        # Generate synthetic data directly within the test
        self.synthetic_data = self.create_synthetic_data()
        self.synthetic_data_file = 'synthetic_test_data.parquet'
        self.synthetic_data.to_parquet(self.synthetic_data_file)
        
        # Load and group the synthetic data
        print("Loading synthetic data...")  # Debugging print
        self.data_manager.load_data(self.synthetic_data_file)
        self.data_manager.group_data()

        # Initialize the solver
        self.solver = ParallelSolver(
            delta_seconds=12.0,
            num_workers=2,
            data_manager=self.data_manager,
            log_time=True,
            log_files=False,
            save_results=True,
            process_function=process1_new_dyn,
            parallel_arithmetic=True,
            ntau=80,
            ngrid=100,
            provided_minfq=500,
            provided_maxfq=10,
            mode='dynamical'
        )
        self.setids = ['1']  # Set test set IDs

    def create_synthetic_data(self):
        """Generate synthetic data for testing."""
        np.random.seed(42)
        object_id = '1'
        num_measurements = 50
        mean_magnitude = 20.0

        mjd_values = np.linspace(50000, 50500, num=num_measurements)
        psMag_values = np.random.normal(loc=mean_magnitude, scale=0.5, size=num_measurements)
        psMagErr_values = np.random.uniform(0.02, 0.1, size=num_measurements)
        filter_values = np.tile([0, 1, 2, 3], int(num_measurements / 4) + 1)[:num_measurements]

        data = {
            'objectId': [object_id] * num_measurements,
            'mjd': mjd_values,
            'psMag': psMag_values,
            'psMagErr': psMagErr_values,
            'filter': filter_values
        }
        return pd.DataFrame(data)

    def test_parallel_solver_process_and_merge(self):
        """Test the processing and merging functionality of the solver."""
        print("Running test_parallel_solver_process_and_merge...")  # Debugging print
        try:
            self.solver.process_ids(set_ids=self.setids, results_file='mock_lc.csv')
            print("Solver processed IDs successfully.")  # Debugging print
        except Exception as e:
            self.fail(f"Error processing/saving data: {e}")

        # Check the current working directory
        print("Current Working Directory:", os.getcwd())

        # Read the processing result from the file
        actual_df = None
        if os.path.exists('mock_lc.csv'):
            actual_df = pd.read_csv('mock_lc.csv')
            print("Actual DataFrame read successfully.")  # Debugging print
        else:
            self.fail("Processed result file missing or cannot be read")

        # Define the expected format of the output (simplified for testing)
        expected_result = (
            "ID,Sampling_1,Sampling_2,Common period (Band1 & Band2),Upper error bound,Lower error bound,Significance,Band1-Band2\n"
            "1,40.81632653061206,49.886621315192315,nan,nan,nan,nan,0-1\n"
            "1,40.81632653061206,44.89795918367381,nan,nan,nan,nan,0-2\n"
            "1,40.81632653061206,40.81632653061256,nan,nan,nan,nan,0-3\n"
            "1,49.886621315192315,44.89795918367381,59.880239520958085,2.0581448337963977,4.27550777826837,0.92,1-2\n"
            "1,49.886621315192315,40.81632653061256,nan,nan,nan,nan,1-3\n"
            "1,44.89795918367381,40.81632653061256,nan,nan,nan,nan,2-3\n"
        )

        # Convert expected result to a pandas DataFrame
        expected_df = pd.read_csv(StringIO(expected_result))

        # Compare the actual and expected results using pandas DataFrames
        try:
            assert_frame_equal(actual_df, expected_df, rtol=1e-2, atol=1e-2)
            print("DataFrames are equal.")  # Debugging print
        except AssertionError as e:
            self.fail(f"Merged result does not match expected result: {e}")

    def tearDown(self):
        """Clean up after tests."""
        print("Cleaning up...")  # Debugging print
        # Attempt to shut down any remaining threads if needed
        if hasattr(self.solver, 'executor') and self.solver.executor:
            try:
                self.solver.executor.shutdown(wait=True)
                print("Executor shutdown successfully.")  # Debugging print
            except Exception as e:
                print(f"Error during executor shutdown: {e}")

        if os.path.isfile(self.synthetic_data_file):
            os.remove(self.synthetic_data_file)
        if os.path.isfile('mock_lc.csv'):
            os.remove('mock_lc.csv')

        # Force garbage collection to help release any remaining resources
        gc.collect()

        # Debugging: Check if there are any active threads
        for thread in threading.enumerate():
            if thread.name != "MainThread":
                print(f"Thread {thread.name} is still active.")  # Debugging print

if __name__ == '__main__':
    unittest.main()
