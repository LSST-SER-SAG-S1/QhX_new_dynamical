.. _dynamical_mode:

Dynamical Mode Module
=====================

This module provides functionality for dynamically managing and processing light curve data with various filters.

.. currentmodule:: QhX.dynamical_mode

Classes and Functions
---------------------

.. automodule:: QhX.dynamical_mode
    :members:
    :undoc-members:
    :show-inheritance:
    :no-index:

.. autoclass:: DataManagerDynamical
    :members:
    :undoc-members:
    :show-inheritance:
    :no-index:

.. autofunction:: get_lc_dyn
    :no-index:

.. autofunction:: process1_new_dyn
    :no-index:

.. seealso::

   For an example of using the `DataManagerDynamical` and `process1_new_dyn` functions with parallel processing, refer to the example in the `Parallelization Solver module` documentation.



Example Usage
-------------

This example demonstrates how to use the `DataManagerDynamical` class and the `process1_new_dyn` function to process light curve data dynamically.

.. code-block:: python

    import QhX
    import pandas as pd
    import numpy as np
    from QhX import DataManagerDynamical, get_lc_dyn, process1_new_dyn
    from QhX.output import classify_periods, classify_period

    # Define the mappings for Gaia dataset
    gaia_mapping = {
        'column_mapping': {'mag': 'psMag', 'flux_error': 'psMagErr', 'time': 'mjd', 'band': 'filter'},  # Correct mappings
        'group_by_key': 'source_id',  # Group by 'source_id' for Gaia
        'filter_mapping': {'BP': 0, 'G': 1, 'RP': 2}  # Map Gaia filters to numeric values
    }

    # Initialize the DataManager with Gaia mappings
    data_manager_gaia = DataManagerDynamical(
        column_mapping=gaia_mapping['column_mapping'],
        group_by_key=gaia_mapping['group_by_key'],
        filter_mapping=gaia_mapping['filter_mapping']
    )

    # Load the dataset from the data folder of the package
    data_manager_gaia.load_data('QhX/data/GaiaQSOcandsLCNobsGgt900.pqt')

    # Group the data by the specified key
    data_manager_gaia.group_data()

    # Process light curve data for a specific object ID
    set_id = 382600737609553280  # Example object ID
    process1_results = process1_new_dyn(
        data_manager_gaia, set_id, ntau=80, ngrid=800, provided_minfq=2000, provided_maxfq=10, include_errors=False
    )

    # Classify the periods obtained from the results
    output = classify_periods([process1_results])
    output['classification'] = output.apply(classify_period, axis=1)
    print(output)

.. note::
    Ensure the file `GaiaQSOcandsLCNobsGgt900.pqt` is placed correctly in the `data` folder of the package and is accessible for loading.