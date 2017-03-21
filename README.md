Solves the Traveling Salesman Problem with CUDA.

Every thread calculates the minimum distance for a group of combinations of 2D points. The minimum for each block is calculated with the reduction of a binary tree.


OpenMP was used to employ as many GPUs as were available in the test machine.
