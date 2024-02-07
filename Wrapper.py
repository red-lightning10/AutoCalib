import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

def find_and_plot_corners(image):
    
    img = image.copy()
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, corner = cv2.findChessboardCorners(grayscale_img, (9, 6), None)
    cv2.drawChessboardCorners(img, (9,6), corner, True)
    corner_int = np.intp(corner).reshape(-1, 2)
    for j in corner_int:
        x, y = j.ravel()
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

    return img, corner

def calculate_homography_using_svd(corners1, corners2): #from ros-developer article in 2017

    A = []
    if len(corners1) < 4 or len(corners2) < 4: 
        print("Operation needs more sets of points. Aborting.")
        return A
    else: # assuming checkerboard corners in order and are matched to the other set in the same order

        corners1_array = np.array(corners1)
        corners2_array = np.array(corners2)
        x = corners1_array[:, 0, 0]
        y = corners1_array[:, 0, 1]
        x_p = corners2_array[:, 0, 0]
        y_p = corners2_array[:, 0, 1]
        
        for i in range(len(corners1)):

            A.append([x[i], y[i], 1, 0, 0, 0, -x[i] * x_p[i], -y[i] * x_p[i], -x_p[i]])
            A.append([0, 0, 0, x[i], y[i], 1, -x[i] * y_p[i], -y[i] * y_p[i], -y_p[i]])

        _, _, V = np.linalg.svd(A)
        H = (np.transpose(V)[:, -1]).reshape(3, 3)
        return H / H[2, 2]
    

def calculate_v_from_homography(H, i, j):

    v_ij = [H[0, i] * H[0, j], H[0, i] * H[1, j] + H[1, i] * H[0, j], \
            H[1, i] * H[1, j], H[2, i] * H[0, j] + H[0, i] * H[2, j], \
            H[2, i] * H[1, j] + H[1, i] * H[2, j], H[2, i] * H[2, j]]
    
    return np.transpose(v_ij)

def calculate_intrinsic_parameters(b):

    v_0 = (b[1] * b[3] - b[0] * b[4]) / (b[0] * b[2] - np.square(b[1]))
    lmda = b[5] - (np.square(b[3]) + v_0 * (b[1] * b[3] - b[0] * b[4])) / b[0]
    alpha = np.sqrt(lmda / b[0])
    beta = np.sqrt(lmda * b[0] / (b[0] * b[2] - np.square(b[1])))
    gamma = -b[1] * np.square(alpha) * beta / lmda
    u_0 = gamma * v_0 / beta - b[3] * np.square(alpha) / lmda

    return [[alpha, gamma, u_0], [0, beta, v_0], [0, 0, 1]]

def calculate_R_and_t(A, H):

    lmda = (np.linalg.norm(np.matmul(np.linalg.inv(A), H[:, 0])) + np.linalg.norm(np.matmul(np.linalg.inv(A), H[:, 1]))) / 2
    
    r1 = np.matmul(np.linalg.inv(A), H[:, 0]) / lmda
    r2 = np.matmul(np.linalg.inv(A), H[:, 1]) / lmda
    r3 = np.cross(r1, r2)
    t = np.matmul(np.linalg.inv(A), H[:, 2]) / lmda
    
    R_ = np.transpose([r1, r2, r3])
    
    R = estimate_better_rotation_matrix(R_)
    
    return R, t.reshape(3,1)

def estimate_better_rotation_matrix(Q):

    U, _, V_t = np.linalg.svd(Q)

    return np.matmul(U, V_t)

def projection_matrix(A, R, t):

    return np.matmul(A, np.hstack((R, t)))

def projection_points(P, XY):

    points = []
    M = np.array(augment_corner_vector(XY)).reshape(-1, 4, 1)
    for i in range(len(XY)):

        points.append(np.matmul(P, M[i]).reshape(3))

    return np.array(points)

def augment_corner_vector(vector):

    vector_np = [[x, y, 0, 1] for x, y in vector[:, 0]]

    return vector_np

def convert_to_params(A, k):

    alpha = A[0][0]
    beta = A[1][1]
    gamma = A[0][1]
    u_0 = A[0][2]
    v_0 = A[1][2]
    return [alpha, gamma, u_0, beta, v_0, k[0], k[1]]

def convert_from_params(params):

    k1 = params[5]
    k2 = params[6]
    A = np.zeros((3, 3))
    A[0, :] = params[:3]
    A[1, :] = [0, params[3], params[4]]
    A[2, :] = [0, 0, 1]

    return A, k1, k2

def reprojection_with_intrinsics_and_extrinsics(params, H_matrices, corners, target_corner):

    A, k1, k2 = convert_from_params(params)
    u_0 = A[0, 2]
    v_0 = A[1, 2]

    errors = []
    reprojected_corners = []

    for i in range(len(corners)):
        
        R, t = calculate_R_and_t(A, H_matrices[i])
        xy_coords = projection_points(np.hstack((R, t)), target_corner)
        pixel_coords = projection_points(projection_matrix(A, R, t), target_corner)
        
        x = xy_coords[:, 0] / xy_coords[:, 2]
        y = xy_coords[:, 1] / xy_coords[:, 2]
        u = pixel_coords[:, 0] / pixel_coords[:, 2]
        v = pixel_coords[:, 1] / pixel_coords[:, 2]
        
        r_sqr = (x**2 + y**2)

        u_real = u + (u - u_0) * (k1 * r_sqr + k2 * r_sqr**2)
        v_real = v + (v - v_0) * (k1 * r_sqr + k2 * r_sqr**2)

        projected_corner = np.array(corners[i]).reshape(-1, 2)
        reprojected_corner = np.transpose(np.array([u_real, v_real]))

        error = np.linalg.norm(np.subtract(projected_corner, reprojected_corner), axis = 1)
        errors = np.append(errors, error)
        reprojected_corners.append(reprojected_corner.reshape(-1, 1, 2))

    return reprojected_corners, errors

def least_square_error_fn(params, H_matrices, corners, target_corner):

    _, error = reprojection_with_intrinsics_and_extrinsics(params, H_matrices, corners, target_corner)

    return error


def main():

    image = []
    path = os.getcwd()
    filepath = os.path.join(path, "Calibration_Imgs") 
    print("\nAssuming that the Calibration images are placed in ", filepath)
    print("Also assuming Checkerboard calibration target - target.png placed in ", path)

    results_path = os.path.join(path, "Results")
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    print("Results will be stored in ", results_path)
    for i in os.listdir(filepath):
        im = cv2.imread(os.path.join(filepath, i))
        image.append(im)
        plt.imsave(os.path.join(results_path, i), im)

    corners = []
    H_matrices = []
    V = []
    
    target = cv2.imread("target.png")
    _, target_corner = find_and_plot_corners(target)
    
    for i, img in enumerate(image):
        
        img, corner = find_and_plot_corners(img)
        plt.imsave(os.path.join(results_path, "corners_of_" + os.listdir(filepath)[i]), img)
        corners.append(corner)
        # H, _ = cv2.findHomography(target_corner, corner, cv2.RANSAC)
        H = calculate_homography_using_svd(target_corner, corner)
        H_matrices.append(H)

        v_12 = calculate_v_from_homography(H, 0, 1)
        v_11 = calculate_v_from_homography(H, 0, 0)
        v_22 = calculate_v_from_homography(H, 1, 1)
        
        if not i:
            V = np.vstack((np.transpose(v_12), np.transpose(v_11 - v_22)))

        else:
            V = np.vstack((V, np.transpose(v_12), np.transpose(v_11 - v_22)))

    V_sqr = np.matmul(np.transpose(V), V)
    eigenvalues, eigenvectors = np.linalg.eig(V_sqr)
    min_eig_index = np.argmin(eigenvalues)
    b = eigenvectors[:, min_eig_index]
    
    np.set_printoptions(suppress=True)
    A = calculate_intrinsic_parameters(b)
    print("Initial intrisics matrix: \n", np.reshape(A, (3,3)))
    k0 = [0, 0]
    print("Initial Distortion coefficients: ", k0)
    params = convert_to_params(A, k0)
    reprojected_corners, error = reprojection_with_intrinsics_and_extrinsics(params, H_matrices, corners, target_corner)
    print("Initial mean error: ", np.mean(error))

    print("\nOptimization of current parameters ongoing \n")
    result = least_squares(least_square_error_fn, params, args = (H_matrices, corners, target_corner))

    new_params = result.x

    reprojected_corners, error = reprojection_with_intrinsics_and_extrinsics(new_params, H_matrices, corners, target_corner)
    A, k1, k2 = convert_from_params(new_params)
    distortion = np.array([k1, k2, 0, 0, 0], dtype = float).reshape(1, 5)
    K = np.array(A).reshape(3, 3)
    print("Intrisics matrix after optimization: \n", K)
    print("Distortion coefficients after optimization: ", [k1, k2])
        
    print("Final mean error: ", np.mean(error))



    for i, img in enumerate(image):

        im_undistorted = cv2.undistort(img, K, distortion)
        for j in reprojected_corners[i]:
            
            x, y = np.intp(j).ravel()
            cv2.circle(im_undistorted, (x, y), 5, (255, 0, 0), -1)
        
        plt.imsave(os.path.join(results_path, "Undistorted_" + os.listdir(filepath)[i]), im_undistorted)

if __name__ == "__main__":    
    main()
