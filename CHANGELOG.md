## v2.0.12 (2025-04-29)

### Refactor

- ruff format

## v2.0.11 (2024-06-24)

### Fix

- fixed exceedance_time function where numpy.bool was not treated as a bool

## v2.0.10 (2024-03-31)

## v2.0.9 (2023-11-17)

### Fix

- "var is True" is not the same as "var == True"
- since working with a series must use items not iteritems

### Refactor

- small refactors for readability

## v2.0.8 (2023-08-08)

## v2.0.7 (2023-07-24)

## v2.0.6 (2023-04-29)

## v2.0.5 (2023-02-14)

### Refactor

- small refactors

## 2.0.4 (2023-01-16)

### Fix

- forgot cltoolbox dependency

## 2.0.3 (2023-01-16)

### Refactor

- refactored code using refurb and pylint

## 2.0.2 (2023-01-08)

## 2.0.1 (2022-10-18)

### Fix

- import Literal in python 3.7 by importing from typing_extensions

## 2.0.0 (2022-10-16)

### BREAKING CHANGE

- move from typic to pydantic

### Fix

- update to pydantic toolbox_utils which is a breaking change

## 1.3.1 (2022-09-28)

## 1.3.0 (2022-09-05)

### Feat

- adopt process to deal with negative and missing flow values

### Fix

- completed all unit support in exceedance_time and fixed rise_lag and fall_lag to ints in storm_events

## 1.2.0 (2022-08-05)

### Feat

- **storm_events**: added the storm_events function

## 1.1.0 (2022-07-29)

### Feat

- added flow_duration and exceedance_time to support tsblender

### Refactor

- removed __future__ since only use 3,7+
- moved to f strings

## 1.0.1 (2022-04-23)

### Fix

- **baseflow_sep**: fixed the import location of the baseflow separation techniques

## 1.0.0 (2022-02-14)

## 0.1.0 (2022-02-07)

### Feat

- **baseflow**: added several baseflow techniques

### Refactor

- docsrc -> docs
