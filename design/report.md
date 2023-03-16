---
title: Multi-Layer Perceptron
subtitle: Report
author: F121584 - Thomas Woolhouse
date: 22/03/2023
numbersections: true
documentclass: report
papersize: A4
fontsize: 11pt
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

- Monolithic structure from processing to report.
- Language?
- Variants

# Data Preprocessing

1. Cleaning (Removing the invalid lines)
2. Filter out the outliers
3. Standardisation
4. Split the dataset into different sets

Amount of data left

# Implementation of MLP

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
