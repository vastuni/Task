import cv2
import numpy as np
import matplotlib.pyplot as plt

#왜인지는 모르겠으나 이미지와 bin 경로를 로컬로 바꾸지 않으면 결과값이 나오지 않아서 그 부분 코드 수정했습니다

# Set up the figure and axis
fig1 = plt.figure(1)
ax1 = fig1.subplots(1, 1)

# C0 -> LiDAR Extrinsic Parameter
R = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04],
              [1.480249e-02,  7.280733e-04, -9.998902e-01],
              [9.998621e-01,  7.523790e-03,  1.480755e-02]])

t = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01])

C02L = np.hstack((R, t.reshape(-1, 1)))
C02L = np.insert(C02L, 3, values=[0, 0, 0, 1], axis=0)

# C0 -> C2 Extrinsic Parameter
R_02 = np.array([[9.999758e-01, -5.267463e-03, -4.552439e-03],
                 [5.251945e-03,  9.999804e-01, -3.413835e-03],
                 [4.570332e-03,  3.389843e-03,  9.999838e-01]])

T_02 = np.array([5.956621e-02, 2.900141e-04, 2.577209e-03])
C02C2 = np.hstack((R_02, T_02.reshape(-1, 1)))
C02C2 = np.insert(C02C2, 3, values=[0, 0, 0, 1], axis=0)

# C2 -> L Extrinsic Parameter
C22L = np.linalg.inv(C02C2) @ C02L

# Intrinsic Parameter
K = np.array([[9.597910e+02, 0.000000e+00, 6.960217e+02],
              [0.000000e+00, 9.569251e+02, 2.241806e+02],
              [0.000000e+00, 0.000000e+00, 1.000000e+00]])

P = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
              [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
              [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]])

D = np.array([-3.691481e-01,
              1.968681e-01,
              1.353473e-03,
              5.677587e-04,
              -6.770705e-02])

# Load image
image_path = 'c:/Users/JeongJoonHee/Desktop/problem_1/0000000000.png' 
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Cannot open image file: {image_path}")

h, w, _ = image.shape

# Load point cloud data
lidar_dtype = [('x', np.float32),
               ('y', np.float32),
               ('z', np.float32),
               ('intensity', np.float32)]

lidar_path = 'c:/Users/JeongJoonHee/Desktop/problem_1/0000000000.bin' 
scan = np.fromfile(lidar_path, dtype=lidar_dtype)
points = np.stack((scan['x'], scan['y'], scan['z']), axis=-1)
ptcloud = np.insert(points, 3, 1, axis=1).T

# Transform LiDAR point cloud to Camera 2 coordinate system
ptcloud_cam2 = C22L @ ptcloud

# Remove points behind the camera
ptcloud_cam2 = ptcloud_cam2[:, ptcloud_cam2[2, :] > 0]

# Normalize points
ptcloud_cam2_normalized = K @ ptcloud_cam2[:3, :]

# Remove points outside the image bounds
u = ptcloud_cam2_normalized[0, :] / ptcloud_cam2_normalized[2, :]
v = ptcloud_cam2_normalized[1, :] / ptcloud_cam2_normalized[2, :]
valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)

u = u[valid]
v = v[valid]
z = ptcloud_cam2[2, valid]

# Plot the points on the image
ax1.scatter([u],[v], c=[z], cmap='rainbow_r', alpha=0.5, s=2)
ax1.imshow(image)
ax1.set_title('Projection Image')
ax1.axis("off")

plt.show()