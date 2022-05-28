# Odometry Estimation

We'd like to play around with different algorithms for estimating odometry from
inertial (and perhaps other) sensors.

TODO:

* Add information about conda environment used to run/test...
* Add how to run Jupyter notebooks...
* Notebook on least-squares filtering, show cost surface updates with EKF/InEKF

Dependencies

* `conda`
* NOT USED [`Sophus`](https://github.com/strasdat/Sophus) C++ library.
    * Requires g++/clang and Eigen3.
    * Requires sympy
    * After building, run `conda-develop /path/to/Sophus/sympy/` to add to conda environment.
