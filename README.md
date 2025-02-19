# Two View 3D Reconstruction

Two view reconstruction using pre-defined camera matrix, SIFT, RANSAC.

Visualization doesn't work too well.

1. Find camera matrix from [checkerboard camera calibration](https://github.com/SimonKim4100/Camera_Calibration_and_Visualization)
2. Keypoint detection with SIFT
3. Essential matrix with SVD
4. Find outliers with RANSAC
5. Visualize

Credits to [3D Reconstruction From Two Views Repository](https://github.com/diegobonilla98/3D-Reconstruction-From-Two-Views/blob/master/scene3D.py), providing many parts of this code.
