import numpy as np
import cv2
from Functions import SceneReconstruction3D
import matplotlib.pyplot as plt

def main():
    # K = np.array([[2759.48 / 4, 0, 1520.69 / 4, 0, 2764.16 / 4,
    #                1006.81 / 4, 0, 0, 1]]).reshape(3, 3)
    # d = np.array([0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, 5)
    K = np.array([[1.47131676e+03, 0.00000000e+00, 6.91598193e+02],
    [0.00000000e+00, 1.49059139e+03, 5.13388257e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype = np.float32)
    d = np.array([0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, 5)
    scene = SceneReconstruction3D(K, d)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    img1 = cv2.imread('img0.jpg')
    # img1 = cv2.pyrUp(img1)
    # img1 = cv2.pyrDown(img1)
    img2 = cv2.imread('img1.jpg')
    # img2 = cv2.pyrUp(img2)
    # img2 = cv2.pyrDown(img2)

    # # Get original dimensions of the images
    # height1, width1 = img1.shape[:2]
    # height2, width2 = img2.shape[:2]

    resize_percentage = 30

    # # Calculate new dimensions
    # new_width1 = int(width1 * resize_percentage / 100)
    # new_height1 = int(height1 * resize_percentage / 100)
    # new_width2 = int(width2 * resize_percentage / 100)
    # new_height2 = int(height2 * resize_percentage / 100)

    # # Create named windows
    # cv2.namedWindow("Image Right", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("Image Left", cv2.WINDOW_NORMAL)

    # # Resize windows using calculated dimensions
    # cv2.resizeWindow("Image Right", new_width1, new_height1)
    # cv2.resizeWindow("Image Left", new_width2, new_height2)

    # cv2.moveWindow("Image Right", 600, 100)
    # cv2.moveWindow("Image Left", 600, 100)

    # Display the images
    # cv2.imshow("Image Right", img1)
    # cv2.imshow("Image Left", img2)
    # cv2.waitKey()

    scene.load_image_pair(img1, img2)

    opt_flow_img = scene.plot_optic_flow()

    # cv2.imshow("imgFlow", opt_flow_img)
    optic_h, optic_w = opt_flow_img.shape[:2]
    new_optic_w = int(optic_w * resize_percentage / 100)
    new_optic_h = int(optic_h * resize_percentage / 100)
    cv2.namedWindow("Optic Flow", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Optic Flow", new_optic_w, new_optic_h)
    cv2.imshow("Optic Flow", opt_flow_img)
    cv2.waitKey()

    opt_flow_img_ransac = scene.plot_optic_flow_ransac(scale=1)

    # cv2.imshow("imgFlow", opt_flow_img)
    # optic_hr, optic_wr = opt_flow_img_ransac.shape[:2]
    # new_optic_wr = int(optic_wr * resize_percentage / 100)
    # new_optic_hr = int(optic_hr * resize_percentage / 100)
    cv2.namedWindow("Optic Flow After RANSAC", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Optic Flow After RANSAC", new_optic_wr, new_optic_hr)
    cv2.imshow("Optic Flow After RANSAC", opt_flow_img_ransac)
    cv2.waitKey()

    Xs, Zs, Ys, colors = scene.plot_point_cloud()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d') 

    ax.scatter(Xs, Ys, Zs, c=colors, marker='o')
    
    # plt.savefig(f"output/scatter_plot.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    main()