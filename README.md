# Biclustering research algorithm implementation

## Context

This project aims at implementing Biclustering methods to link genes with heterogeneous data using Mosek Python API.

## Prerequisites

Before you begin, ensure you have met the following requirements: 
<!--- These are just example requirements. Add, duplicate or remove as required --->
* You have installed a version of Python > 2.6.

## Algorithms implemented

- SDP resolution by Mosek
- K-means constrained from "Data Clustering with Cluster Size Constraints Usinga Modifiedk-means Algorithm" article by Nuwan Ganganath†, Chi-Tsun Cheng, and Chi K. Tse.

## How to use 

To use the SDP biclustering algorithm, follow these steps:

To load your data:

Replace ```I=Instance.LoadInstance("save3.txt")```  with your text file representing your clusters matrixes in SDP_21.py

To run the algorithm on your data:

```
python SDP_21.py 
```

## Contributors

Authors : 
 - Alexis Le Glaunec
 - Geoffroy Zardi

This project was realized in collaboration with researchers from the university of Evry :
 - Farida Zehraoui
 - Eric Angel
