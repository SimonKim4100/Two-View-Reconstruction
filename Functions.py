import cv2
import numpy as np

class SceneReconstruction3D:

    def __init__(self, K, dist):
        self.K = K
        self.K_inv = np.linalg.inv(K)  # store inverse for fast access
        self.d = dist

    def load_image_pair(
            self,
            img1: np.array,
            img2: np.array) -> None:

        # Simply assign the images to the instance variables
        self.img1, self.img2 = img1, img2

    def plot_optic_flow(self):
        self._extract_keypoints_sift()

        img = np.copy(self.img1)
        for pt1, pt2 in zip(self.match_pts1, self.match_pts2):
            pt1 = tuple(map(int, pt1))
            pt2 = tuple(map(int, pt2))
            cv2.arrowedLine(img, pt1, pt2, color=(255, 0, 0), thickness = 2, line_type = cv2.LINE_AA)

        return img

    def plot_optic_flow_ransac(self, scale=10):
        # Run the RANSAC pipeline
        self._find_fundamental_matrix()
        self._find_essential_matrix()
        self._find_camera_matrices_rt()
        
        # Use the RANSAC mask to filter the original pixel coordinates:
        inlier_pts1 = self.match_pts1[self.Fmask.ravel() == 1]
        inlier_pts2 = self.match_pts2[self.Fmask.ravel() == 1]
        
        print("Number of inlier matches (from mask):", len(inlier_pts1))
        
        img = np.copy(self.img1)
        for pt1, pt2 in zip(inlier_pts1, inlier_pts2):
            # Compute difference in pixel space
            diff = np.array(pt2) - np.array(pt1)
            print("Flow vector in pixel coordinates:", diff)
            
            # Scale the difference for visualization
            scaled_diff = diff * scale
            pt2_scaled = np.array(pt1) + scaled_diff
            
            pt1_int = tuple(map(int, pt1))
            pt2_int = tuple(map(int, pt2_scaled))
        
            cv2.arrowedLine(img, pt1_int, pt2_int, color=(0, 255, 0),
                            thickness=2, line_type=cv2.LINE_AA)
            cv2.circle(img, pt1_int, 4, (0, 0, 255), -1)
            cv2.circle(img, pt2_int, 4, (0, 0, 255), -1)
        
        return img
    
    def plot_point_cloud(self):
        # Ensure necessary computations are performed
        self._find_fundamental_matrix()
        self._find_essential_matrix()
        self._find_camera_matrices_rt()

        # Triangulate points
        first_inliers = np.array(self.match_inliers1)[:, :2]
        second_inliers = np.array(self.match_inliers2)[:, :2]
        pts4D = cv2.triangulatePoints(self.Rt1, self.Rt2, first_inliers.T, second_inliers.T).T

        # Convert from homogeneous coordinates to 3D
        pts3D = pts4D[:, :3] / pts4D[:, 3, None]

        # Extract colors for each 3D point from the first image
        colors = []
        for pt in first_inliers:
            x, y = int(pt[0]), int(pt[1])  # Pixel coordinates
            color = self.img1[y, x]  # Get the color (BGR format) from the first image
            colors.append(color)

        # Convert BGR to RGB for matplotlib
        colors = np.array(colors)[:, ::-1] / 255.0  # Normalize to [0, 1] for plotting

        # Prepare data for 3D plot
        Xs, Zs, Ys = [pts3D[:, i] for i in range(3)]

        return Xs, Zs, Ys, colors

    def _extract_keypoints_sift(self):
        """Extracts keypoints via sift descriptors"""
        # extract keypoints and descriptors from both images
        # detector = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.11, edgeThreshold=10)
        detector = cv2.SIFT_create(
            contrastThreshold=0.1,
            edgeThreshold=10
        )
        first_key_points, first_desc = detector.detectAndCompute(self.img1,
                                                                 None)
        second_key_points, second_desc = detector.detectAndCompute(self.img2,
                                                                   None)
        
        if first_desc is None or second_desc is None:
            raise ValueError("SIFT failed to find descriptors in one or both images.")
        
        # match descriptors
        # matcher = cv2.BFMatcher(cv2.NORM_L1, True)
        # matches = matcher.match(first_desc, second_desc)

        matcher = cv2.BFMatcher(cv2.NORM_L2)
        knn_matches = matcher.knnMatch(first_desc, second_desc, k=2)

        # # generate lists of point correspondences
        # self.match_pts1 = np.array(
        #     [first_key_points[match.queryIdx].pt for match in matches])
        # self.match_pts2 = np.array(
        #     [second_key_points[match.trainIdx].pt for match in matches])
        # Apply Lowe's ratio test to filter matches
        self.match_pts1 = []
        self.match_pts2 = []
        for m, n in knn_matches:
            if m.distance < 0.75 * n.distance:
                self.match_pts1.append(first_key_points[m.queryIdx].pt)
                self.match_pts2.append(second_key_points[m.trainIdx].pt)

        self.match_pts1 = np.array(self.match_pts1)
        self.match_pts2 = np.array(self.match_pts2)

    def _find_fundamental_matrix(self):
        """Estimates fundamental matrix """
        self.F, self.Fmask = cv2.findFundamentalMat(self.match_pts1,
                                                    self.match_pts2,
                                                    cv2.FM_RANSAC, 0.5, 0.95)

    def _find_essential_matrix(self):
        """Estimates essential matrix based on fundamental matrix """
        self.E = self.K.T.dot(self.F).dot(self.K)

    def _find_camera_matrices_rt(self):
        """Finds the [R|t] camera matrix"""
        # decompose essential matrix into R, t (See Hartley and Zisserman 9.13)
        U, S, Vt = np.linalg.svd(self.E)
        W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                      1.0]).reshape(3, 3)

        # iterate over all point correspondences used in the estimation of the
        # fundamental matrix
        first_inliers = []
        second_inliers = []
        for pt1, pt2, mask in zip(
                self.match_pts1, self.match_pts2, self.Fmask):
            if mask:
                # normalize and homogenize the image coordinates
                first_inliers.append(self.K_inv.dot([pt1[0], pt1[1], 1.0]))
                second_inliers.append(self.K_inv.dot([pt2[0], pt2[1], 1.0]))

        # Determine the correct choice of second camera matrix
        # only in one of the four configurations will all the points be in
        # front of both cameras

        R = T = None
        R = U.dot(W.T).dot(Vt)
        T = U[:, 2]
        for r in (U.dot(W).dot(Vt), U.dot(W.T).dot(Vt)):
            for t in (U[:, 2], -U[:, 2]):
                if self._in_front_of_both_cameras(
                        first_inliers, second_inliers, r, t):
                    R, T = r, t

        assert R is not None, "Camera matricies were never found"

        self.match_inliers1 = first_inliers
        self.match_inliers2 = second_inliers
        self.Rt1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        self.Rt2 = np.hstack((R, T.reshape(3, 1)))

    def _in_front_of_both_cameras(self, first_points, second_points, rot,
                                  trans):
        for first, second in zip(first_points, second_points):
            first_z = np.dot(rot[0, :] - second[0] * rot[2, :],
                             trans) / np.dot(rot[0, :] - second[0] * rot[2, :],
                                             second)
            first_3d_point = np.array([first[0] * first_z,
                                       second[0] * first_z, first_z])
            second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T,
                                                                     trans)

            if first_3d_point[2] < 0 or second_3d_point[2] < 0:
                return False

        return True
