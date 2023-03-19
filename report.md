---
title: Multi-Layer Perceptron
subtitle: Report
author: F121584 - Thomas Woolhouse
date: 22/03/2023
numbersections: true
documentclass: report
papersize: A4
fontsize: 12pt
geometry: margin=2cm
toc: true
links-as-notes: true
colorlinks: blue
header-includes: |
	\usepackage{sectsty}
	\usepackage{pgffor}
	\usepackage{minted}
---

# Overview

***Gooooooood morning Vietnam***
- Monolithic structure from processing to report.
- Language?
- Variants

![alt](graph/model/std_dev_3.lin1-9.year_2_1_1.H05.sigmoid.momentum.png)

# Data Preprocessing

The data preprocessing has been carried out programmatically using Python to clean the data using statistical analysis.
Initially I converted the given Excel dataset into a CSV format.
This is can be easily manipulated by the builtin standard library module: `csv`.
The preprocessing of the raw dataset is abstracted into several distinct sections:

1. [Parsing](#parsing) the Raw dataset.
2. [Cleaning](#cleaning).
3. [Standardising](#standardising).
4. [Splitting](#splitting).

## Parsing

Parsing entails loading the raw dataset into the main Python scriptThe data is streamed into the program line by line.
This is parsed into a [`Record`](#recordpy) object using a static method which will raise an `Exception` if the row contains any invalid data.
For our dataset, this means any rows with non-floating point values are rejected.
This leads to a reduction in the size of the final dataset, however, there are 1461 in the raw set and after removing these exception raising values, 1456 rows remain.
It is an insignificant amount of data to lose and therefore, I did not deem it necessary to impute values back into the dataset.

## Cleaning

Inline with programmatic preprocessing, for cleaning the data, I experimented with different methods of statistical analysis to remove outliers from the dataset.
I explored using the standard deviation and the inter-quartile range to provide empirical method of removing the anomalous data.

The [standard deviation method](#filterpy) would compute the mean $\bar{x}$ and standard deviation $\sigma$ for each column (both the predictors and predictand) of the dataset.
For every row, if any field's value lay outside of $\bar{x} \pm R\sigma$ for it's respective column, it would be rejected from the final dataset.
The variable $R$ controls the number of standard deviations the value must be within.

The process for filtering the data using inter-quartile range is broadly the same, changing the the mean to the median, and $\sigma$ to the titular IQR.

To get the best of both worlds, the final dataset would undergo both of these filtering methods.

| Method                  | Records |      % |
| ----------------------- | ------: | -----: |
| No Cleaning             |    1456 | 100.0% |
| 3 Standard Deviations   |    1415 | 97.18% |
| 2 Standard Deviations   |    1225 | 84.13% |
| 1 Standard Deviation    |     358 | 24.59% |
| 3 Inter-Quartile Ranges |    1403 | 96.36% |
| 3 Inter-Quartile Ranges |    1232 | 84.62% |
| 3 Inter-Quartile Ranges |     769 | 52.82% |
| 3 Deviations & Ranges   |    1389 | 95.40% |
| 2 Deviations & Ranges   |    1172 | 80.49% |
| 1 Deviation & Range     |     342 | 23.49% |

The amount of data left in the final datasets can be seen in the table above.
As we can see, a combined approach removes more data points than either of the individual ones, meaning neither strategy would be wholey effective on its own.

## Standardising

## Splitting

Amount of data left

# Implementation of MLP

WHY DO WE RANDOMISE.

GRAPH different modifications with each other.

- How it can be modified to change it to another dataset w/ different inputs (change record.py & record.h).
- Reasons for cpp
- Data structure and its optimisations
- Code structure
- Reasons for template hell - Extensibility to any structure

## Activation Functions

## Modifications

# Training and Network Selection

# Evaluation of Final Model

# Comparison with Other Models

# Appendix Code

## record.py

PY_CODE_FILE{Python}{./src/process/record.py}

## filter.py

PY_CODE_FILE{Python}{./src/process/filter.py}
