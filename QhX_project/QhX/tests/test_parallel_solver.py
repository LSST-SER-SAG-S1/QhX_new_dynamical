import unittest
import os
import pandas as pd
import numpy as np
from QhX.parallelization_solver import ParallelSolver
from QhX import DataManagerDynamical, process1_new_dyn

class TestParallelSolver(unittest.TestCase):
    def setUp(self):
        # Set up the data manager with the required configuration
        agn_dc_mapping = {
            'column_mapping': {'flux': 'psMag', 'time': 'mjd', 'band': 'filter'},
            'group_by_key': 'objectId',
            'filter_mapping': {0: 0, 1: 1, 2: 2, 3: 3}  # Map AGN DC filters (0, 1, 2, 3)
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
        self.data_manager.load_data(self.synthetic_data_file)
        self.data_manager.group_data()  # Ensure data is grouped

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

        # Set test set IDs (must match grouped objectId from the synthetic data)
        self.setids = ['1']  # Using integer as object ID to match grouped data

    def create_synthetic_data(self):
        # Create synthetic data for one object with 50 measurements across multiple filters
        np.random.seed(42)  # For reproducibility
        object_id = '1'
        num_measurements = 50
        mean_magnitude = 20.0

        # Generate random MJD values (timestamps) for the measurements
        mjd_values = np.linspace(50000, 50500, num=num_measurements)

        # Generate magnitudes around the mean value with some noise
        psMag_values = np.random.normal(loc=mean_magnitude, scale=0.5, size=num_measurements)

        # Generate random errors for the magnitudes
        psMagErr_values = np.random.uniform(0.02, 0.1, size=num_measurements)

        # Assign filters (0, 1, 2, 3) in a repeating pattern
        filter_values = np.tile([0, 1, 2, 3], int(num_measurements / 4) + 1)[:num_measurements]

        # Create a DataFrame with the generated data
        data = {
            'objectId': [object_id] * num_measurements,
            'mjd': mjd_values,
            'psMag': psMag_values,
            'psMagErr': psMagErr_values,
            'filter': filter_values
        }

        return pd.DataFrame(data)

    def test_parallel_solver_process_and_merge(self):
        # Run the solver with the test set IDs
        try:
            self.solver.process_ids(set_ids=self.setids, results_file='mock_results_file.csv')
        except Exception as e:
            print(f"Error processing/saving data: {e}")
        
        # Read the processing result from the file
        if os.path.exists('mock_results_file.csv'):
            with open('mock_results_file.csv') as f:
                process_result = f.read()
        else:
            process_result = ""

        # Print the actual process result
        print("\nActual Process Result:\n", process_result)
    
        # Define the expected format of the output (simplified for testing)
        expected_result = (
            "ID,Sampling_1,Sampling_2,Common period (Band1 & Band2),Upper error bound,Lower error bound,Significance,Band1-Band2\n"
            "1,40.81632653061206,49.886621315192315,nan,nan,nan,nan,0-1\n"
            "1,40.81632653061206,44.89795918367381,nan,nan,nan,nan,0-2\n"
            "1,40.81632653061206,40.81632653061256,nan,nan,nan,nan,0-3\n"
            "1,49.886621315192315,44.89795918367381,59.880239520958085,2.0581448337963977,4.27550777826837,0.92,1-2\n"
            "1,49.886621315192315,40.81632653061256,nan,nan,nan,nan,1-3\n"
            "1,44.89795918367381,40.81632653061256,nan,nan,nan,nan,2-3\n"  # Simplified expected values for this test
        )
    
        # Print the expected result for comparison
        print("\nExpected Result:\n", expected_result)
    
        # Assert the presence of the processed result file and its content
        self.assertIsNotNone(process_result, "Merged file missing or cannot be read")
    
        # Compare the actual and expected results
        self.assertEqual(process_result.strip(), expected_result.strip(), "Merged result does not match expected result")

    def tearDown(self):
        # Clean up any created files after the test
        if os.path.isfile(self.synthetic_data_file):
            os.remove(self.synthetic_data_file)
        if os.path.isfile('mock_results_file.csv'):
            os.remove('mock_results_file.csv')

if __name__ == '__main__':
    unittest.main()
