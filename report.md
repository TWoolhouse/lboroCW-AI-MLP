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
	\sectionfont{\clearpage}
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

1. Parsing the Raw dataset.
2. Cleaning.
3. Standardising.
4. Splitting.

## Parsing

The data is streamed into the program line by line.
This is parsed into a [`Record`](#record) object using a static method which will raise an `Exception` if the row contains any invalid data.
For our dataset, this means any rows with non-floating point values are rejected.
This leads to a reduction in the size of the final dataset, however, there are 1461 in the raw set and after removing these exception raising values, 1456 rows remain.
It is an insignificant amount of data to lose and therefore, I did not deem it necessary to impute values back into the dataset.

## Cleaning

Inline with programmatic preprocessing, for cleaning the data, I experimented with different methods of statistical analysis to remove outlier from the dataset.


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

## Record

PY_CODE_FILE{Python}{./src/process/record.py}
