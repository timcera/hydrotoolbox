.. image:: https://github.com/timcera/hydrotoolbox/actions/workflows/pypi-package.yml/badge.svg
    :alt: Tests
    :target: https://github.com/timcera/hydrotoolbox/actions/workflows/pypi-package.yml
    :height: 20

.. image:: https://img.shields.io/coveralls/github/timcera/hydrotoolbox
    :alt: Test Coverage
    :target: https://coveralls.io/r/timcera/hydrotoolbox?branch=master
    :height: 20

.. image:: https://img.shields.io/pypi/v/hydrotoolbox.svg
    :alt: Latest release
    :target: https://pypi.python.org/pypi/hydrotoolbox/
    :height: 20

.. image:: https://img.shields.io/pypi/l/hydrotoolbox.svg
    :alt: BSD-3 clause license
    :target: https://pypi.python.org/pypi/hydrotoolbox/
    :height: 20

.. image:: https://img.shields.io/pypi/dd/hydrotoolbox.svg
    :alt: hydrotoolbox downloads
    :target: https://pypi.python.org/pypi/hydrotoolbox/
    :height: 20

.. image:: https://img.shields.io/pypi/pyversions/hydrotoolbox
    :alt: PyPI - Python Version
    :target: https://pypi.org/project/hydrotoolbox/
    :height: 20

hydrotoolbox - Quick Guide
==========================
The hydrotoolbox is a Python script for hydrologic calculations and analysis
or by function calls within Python.  Uses pandas (http://pandas.pydata.org/)
or numpy (http://numpy.scipy.org) for any heavy lifting.

Requirements
------------
* python 3.7 or higher

Installation
------------
pip
~~~
.. code-block:: bash

    pip install hydrotoolbox

conda
~~~~~
.. code-block:: bash

    conda install -c conda-forge hydrotoolbox

Usage - API
-----------
All functions return a PANDAS DataFrame.  Input can be a CSV or TAB separated
file, or a PANDAS DataFrame and is supplied to the function via the 'input_ts'
keyword.

Simply import hydrotoolbox::

    from hydrotoolbox import hydrotoolbox

    # Then you could call the functions
    ntsd = hydrotoolbox.baseflow_sep(method='boughton', input_ts='tests/test_fill_01.csv')

Usage - Command Line
--------------------
All functions are available from the command line.  The command line
interface is a wrapper around the functions in the hydrotoolbox module.  The
command line arguments match one-to-one with the function arguments.

To get help for the command line interface, run::

    hydrotoolbox --help

Which gives you the following output::

    usage: hydrotoolbox [-h] [-v]
                        {baseflow_sep, recession, flow_duration, storm_events,
                        indices, exceedance_time, about} ...

    positional arguments:
      {baseflow_sep,recession,flow_duration,storm_events,indices,exceedance_time,about}
        baseflow_sep        baseflow_sep subcommand
        recession           Recession coefficient.
        flow_duration       Flow duration.
        storm_events        Storm events.
        indices             Calculate hydrologic indices.
        exceedance_time     Calculate the time that a time series exceeds (or is
                            below) a threshold.
        about               Display version number and system information.

    options:
      -h, --help            show this help message and exit
      -v, --version         show program's version number and exit

To get help for a specific subcommand, run::

    hydrotoolbox <subcommand> --help

The default for all of the subcommands is to accept data from stdin (typically
a pipe).  If a subcommand accepts an input file for an argument, you can use
"--input_ts=input_file_name.csv", or to explicitly specify from stdin (the
default) "--input_ts='-'".

For the subcommands that output data it is printed to the screen and you can
then redirect to a file.
