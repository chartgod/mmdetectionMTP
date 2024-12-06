# Performance Metrics Evaluation Script

## Overview
This script is designed to evaluate segmentation performance metrics for image pairs in specified `label` and `predict` directories. The metrics calculated include:
- IoU (Intersection over Union)
- F1-score
- Precision
- Recall

Results are sorted by IoU in descending order and saved to an output file for further analysis.

## Requirements
Before running the script, ensure that Python is installed along with the following libraries:
- `numpy`
- `pillow`
- `scikit-learn`

If the required libraries are not already installed, you can add them using pip:
```bash
pip install numpy pillow scikit-learn

Performance Results (Sorted by IoU)
======================================================================
Rank    Filename        IoU (%)    F1 (%)     Precision (%)     Recall (%)
======================================================================
1       image1.png      90.23      89.45      92.34            88.12
2       image2.png      85.34      84.00      88.12            80.45
3       image3.png      75.56      74.00      78.56            70.45
...
