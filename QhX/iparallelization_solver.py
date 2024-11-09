"""
The `iparallelization_solver` interface is designed for parallel execution of an input function.

This module defines the `IParallelSolver` class, which orchestrates the parallel execution of a
general processing function on a dataset consisting of multiple independent data subsets, referred to as set IDs.
It includes a method for logging (intended to start a separate logging thread) and methods to manage result saving.

Attributes:
-----------
- `num_workers`: Specifies the number of worker processes to spawn.
- `results_queue`: A queue for storing aggregated results if the unified saving mode is enabled.
- `set_ids_queue`: A queue holding the IDs of the datasets to be processed.
- `unified_save_mode`: A boolean flag indicating whether results should be saved in a unified file.

Author:
-------
Momcilo Tosic
Astroinformatics student
Faculty of Mathematics, University of Belgrade
"""

from multiprocessing import Process, Queue

# Default number of processes to spawn
DEFAULT_NUM_WORKERS = 4

class IParallelSolver:
    """
    Manages the parallel execution of data processing functions.

    Attributes:
        num_workers (int): Number of worker processes to spawn.
        results_queue (Queue): Queue to store aggregated results if `unified_save_mode` is enabled.
        set_ids_queue (Queue): Queue holding the set IDs for processing.
        unified_save_mode (bool): When set to `True`, results are saved to the specified file.
    """

    def __init__(self, num_workers=DEFAULT_NUM_WORKERS):
        """
        Initialize the ParallelSolver with the specified configuration.

        Parameters:
        -----------
        num_workers (int): Number of worker processes to spawn.
        """
        print(f"Initializing IParallelSolver with {num_workers} worker(s).")
        self.num_workers = num_workers
        self.results_ = Queue()            # Initialize the results queue
        self.set_ids_ = Queue()             # Initialize the set IDs queue
        self.save_all_results_ = False      # Initialize the save results flag

    def process_wrapper(self):
        """
        Wrapper for the process function to integrate logging and result handling.

        Retrieves set IDs from the queue, processes each ID, and manages logging and error handling.
        Results can be saved to the unified results queue if `save_all_results_` is True.
        """
        print("Starting process_wrapper...")

        while not self.set_ids_.empty():
            try:
                set_id = self.set_ids_.get()
                print(f"Processing set ID: {set_id}")
            except Exception as e:
                print(f"Error retrieving set ID from queue: {e}")
                break

            res_string = ""
            try:
                self.maybe_begin_logging(set_id)
                result = self.get_process_function_result(set_id)
                res_string = self.aggregate_process_function_result(result)

                if self.save_all_results_:
                    self.results_.put(res_string)
                    print(f"Result for set ID {set_id} added to results queue.")
            except Exception as e:
                print(f"Error processing/saving data for set ID {set_id}: {str(e)}")
            finally:
                try:
                    self.maybe_stop_logging()
                    self.maybe_save_local_results(set_id, res_string)
                    print(f"Local results saved for set ID {set_id}.")
                except Exception as e:
                    print(f"Error stopping logs or saving local results for set ID {set_id}: {str(e)}")

    def process_ids(self, set_ids, results_file=None):
        """
        Processes a list of set IDs using the configured process function in parallel.

        Parameters:
        -----------
        set_ids (list of str): List of set IDs to process.
        results_file (str, optional): Path to save aggregated results if `save_all_results_` is True.
        """
        print(f"Starting process_ids with {len(set_ids)} set ID(s). Results file: {results_file}")
        self.save_all_results_ = results_file is not None

        # Fill the set IDs queue with provided IDs
        for id in set_ids:
            self.set_ids_.put(id)
            print(f"Set ID {id} added to the queue.")

        # Generate and start worker processes
        processes = [Process(target=self.process_wrapper) for _ in range(self.num_workers)]
        for p in processes:
            p.start()
            print(f"Started process with PID {p.pid}")
        for p in processes:
            p.join()
            print(f"Process with PID {p.pid} completed.")

        # Save results to the specified results file if needed
        self.maybe_save_results(results_file)
        print("All results have been processed and saved.")

    def aggregate_process_function_result(self, result):
        """
        Aggregate the result of the process function into a formatted string.

        Parameters:
        -----------
        result: Result data to aggregate.

        Returns:
        --------
        str: Aggregated result as a formatted string.
        """
        print("Aggregating process function result.")
        # Placeholder for aggregation logic
        return "Aggregated result string."

    def get_process_function_result(self, set_id):
        """
        Get the result from processing a single set ID.

        Parameters:
        -----------
        set_id (str): The ID of the set to process.

        Returns:
        --------
        Result data from processing the set ID.
        """
        print(f"Getting process function result for set ID {set_id}.")
        # Placeholder for actual processing function
        return {"result": "sample result"}

    def maybe_begin_logging(self, set_id):
        """
        Start logging for a specific set ID if logging is enabled.

        Parameters:
        -----------
        set_id (str): The ID of the set for which logging might be started.
        """
        print(f"Starting logging for set ID {set_id}.")

    def maybe_stop_logging(self):
        """
        Stop logging for the current process if logging is enabled.
        """
        print("Stopping logging.")

    def maybe_save_local_results(self, set_id, res_string):
        """
        Save local results for a specific set ID if necessary.

        Parameters:
        -----------
        set_id (str): The ID of the set.
        res_string (str): The result string to save.
        """
        print(f"Saving local results for set ID {set_id}. Result string: {res_string}")

    def maybe_save_results(self, results_file):
        """
        Save all results in the unified results queue to a specified file.

        Parameters:
        -----------
        results_file (str): Path to the file where results will be saved.
        """
        if results_file:
            print(f"Saving all results to {results_file}.")
            # Placeholder for saving logic
        else:
            print("No results file specified, skipping save.")
