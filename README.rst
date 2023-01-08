.. image:: https://github.com/timcera/hydrotoolbox/actions/workflows/python-package.yml/badge.svg
    :alt: Tests
    :target: https://github.com/timcera/hydrotoolbox/actions/workflows/python-package.yml
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
Should be as easy as running ``pip install hydrotoolbox``
at any command line.

Usage - Command Line
--------------------
Just run 'hydrotoolbox --help' to get a list of subcommands::


    usage: hydrotoolbox [-h] [-v] {baseflow_sep,recession,about} ...

    positional arguments:
      {baseflow_sep,recession,about}
        baseflow_sep        baseflow_sep subcommand
        recession           Recession coefficient.
        about               Display version number and system information.

    optional arguments:
      -h, --help            show this help message and exit
      -v, --version         show program's version number and exit

The default for all of the subcommands is to accept data from stdin (typically
a pipe).  If a subcommand accepts an input file for an argument, you can use
"--input_ts=input_file_name.csv", or to explicitly specify from stdin (the
default) "--input_ts='-'".

For the subcommands that output data it is printed to the screen and you can
then redirect to a file.

Usage - API
-----------
You can use all of the command line subcommands as functions.  The function
signature is identical to the command line subcommands.  The return is always
a PANDAS DataFrame.  Input can be a CSV or TAB separated file, or a PANDAS
DataFrame and is supplied to the function via the 'input_ts' keyword.

Simply import hydrotoolbox::

    from hydrotoolbox import hydrotoolbox

    # Then you could call the functions
    ntsd = hydrotoolbox.baseflow_sep(method='broughton', input_ts='tests/test_fill_01.csv')
