# This sucks!!!
import numpy as np 
import matplotlib.pyplot as plt 

TESTS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
Precision = [0.59, 0.98, 0.96, 0.98, 0.97, 0.91, 0.99, 0.98, 0.95, 0.98, 0.98, 0.97, 0.98, 0.92, 0.99, 0.95, 0.95, 0.94, 0.96, 0.94, 0.95, 0.97, 0.98, 0.98, 0.98, 0.99, 0.95, 0.99, 0.99, 0.99, 0.98, 0.97, 0.98, 0.98, 0.97, 0.96]
Recall = [0.86, 0.99, 0.96, 0.98, 0.97, 0.95, 0.98, 0.98, 0.97, 0.97, 0.98, 0.97, 0.98, 0.98, 0.98, 0.99, 0.98, 0.97, 0.98, 0.97, 0.97, 0.99, 0.99, 0.98, 0.91, 0.99, 0.98, 0.97, 0.97, 0.99, 0.98, 0.99, 0.97, 0.98, 0.98, 0.97]
f1 = [0.7, 0.98, 0.96, 0.98, 0.97, 0.93, 0.99, 0.98, 0.96, 0.97, 0.98, 0.97, 0.98, 0.95, 0.98, 0.97, 0.97, 0.96, 0.97, 0.96, 0.96, 0.98, 0.99, 0.98, 0.94, 0.99, 0.97, 0.98, 0.98, 0.99, 0.98, 0.98, 0.98, 0.98, 0.98, 0.97]

X_axis = np.arange(len(TESTS))
  
plt.bar(X_axis - 0.2, Precision, 0.4, label = 'Precision')
plt.bar(X_axis - 0.0, Precision, 0.4, label = 'Recall')
plt.bar(X_axis + 0.2, Precision, 0.4, label = 'f1')
  
plt.xticks(X_axis, TESTS)
plt.title("Testing Results per Character")
plt.xlabel("Results")
plt.ylabel("Percentage")
plt.legend()
plt.figure(figsize=(3000, 100))
plt.show()


# Classification, Precision, Recall, F1-Score, Support
# 0, 0.59, 0.86, 0.70, 1381
# 1, 0.98, 0.99, 0.98, 1575
# 2, 0.96, 0.96, 0.96, 1398
# 3, 0.98, 0.98, 0.98, 1428
# 4, 0.97, 0.97, 0.97, 1365
# 5, 0.91, 0.95, 0.93, 1263
# 6, 0.99, 0.98, 0.99, 1375
# 7, 0.98, 0.98, 0.98, 1459
# 8, 0.95, 0.97, 0.96, 1365
# 9, 0.98, 0.97, 0.97, 1392
# A, 0.98, 0.98, 0.98, 2774
# B, 0.97, 0.97, 0.97, 1734
# C, 0.98, 0.98, 0.98, 4682
# D, 0.92, 0.98, 0.95, 2027
# E, 0.99, 0.98, 0.98, 2288
# F, 0.95, 0.99, 0.97, 232
# G, 0.95, 0.98, 0.97, 1152
# H, 0.94, 0.97, 0.96, 1444
# I, 0.96, 0.98, 0.97, 224
# J, 0.94, 0.97, 0.96, 1699
# K, 0.95, 0.97, 0.96, 1121
# L, 0.97, 0.99, 0.98, 2317
# M, 0.98, 0.99, 0.99, 2467
# N, 0.98, 0.98, 0.98, 3802
# O, 0.98, 0.91, 0.94, 11565
# P, 0.99, 0.99, 0.99, 3868
# Q, 0.95, 0.98, 0.97, 1162
# R, 0.99, 0.97, 0.98, 2313
# S, 0.99, 0.97, 0.98, 9684
# T, 0.99, 0.99, 0.99, 4499
# U, 0.98, 0.98, 0.98, 5802
# V, 0.97, 0.99, 0.98, 836
# W, 0.98, 0.97, 0.98, 2157
# X, 0.98, 0.98, 0.98, 1254
# Y, 0.97, 0.98, 0.98, 2172
# Z, 0.96, 0.97, 0.97, 1215