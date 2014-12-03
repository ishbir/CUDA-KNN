CUDA-KNN
========

Implementation of k nearest neighbours algorithm using CUDA and C++. Originally developed in University of Iowa with Dr. Suely Oliveiar in Summer, 2012. Code was then substantially improved and optimized for writing an extended essay in Computer Science for the IB program.

Requires [thrust](https://github.com/thrust/thrust) library and [moderngpu](https://github.com/NVlabs/moderngpu) library. For generating the testing times, add the compiled executable in the Testing folder, generate the test case data using dataGenerator.py and start timing using testTimes.py.

You might need to slightly modify the code to get it to work since it was coded a long time ago. 

The paper was not published in any journal but is available online at: http://www.ishbir.com/static/ee.pdf
