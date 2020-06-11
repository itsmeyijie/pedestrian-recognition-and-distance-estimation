import cv2
import glob
import numpy as np

# From matlab
# 左边相机的内参矩阵
mtx = 1.0e+02*np.array([[8.243991411668103, 0.0, 0.0],
                [0.027400612297573  , 8.234152132657007, 0.0],
                [5.953498585081239  , 3.562487032848879  , 0.010000000000000]]).T
mtx_r = 1.0e+02 *np.array([[8.177134140521405, 0.0, 0.0],
                  [-0.044279416675194   ,8.176511422717468, 0.0],
                  [6.589423417658030,   3.152494714988066,   0.010000000000000]]).T
# 右边相机的内参矩阵
R = np.array([[0.999343217346116  , 0.009605024909863  ,-0.034941056663968],
              [-0.010397538347202 ,  0.999691175350544  ,-0.022570891045005],
              [0.034713472033667 ,  0.022919367851839 ,  0.999134464142159]])
# R = np.array([[0.898066037373019, 0.0213408993926283, -0.439342643650985],
#               [-0.0214251464935368, 0.999759088048663, 0.00476749009820505],
#               [0.439338543283939, 0.00513145956036998, 0.898306914427317]])  # 旋转矩阵
T = 1.0e+02 *np.array([1.736265280119826   ,0.045278304441770  , 0.026552225706383])
# T = np.array([-264.886066592313, -1.77392898927413, 46.7689011903979])  # 平移矩阵

# K [-0.404240685876510,0.679412484741657,-4.804320406255162]
# P [0.003558952648167,-1.745438223176216e-04]
# [K1 K2 P1 P2 K3]
# distCoeffs1 = np.array([-0.404240685876510, 0.679412484741657,
#                         0.003558952648167, -1.745438223176216e-04, -4.804320406255162])  # 左边相机的畸变参数
distCoeffs1 = np.array([-0.276143528381186 , -0.013796742147836,
                        -0.002444991133973 ,  0.008694876339845, 0.111539050029845])
# K [-0.426243385650803,0.630959102496925,-1.016065433982113]
# P [0.002912859612620,0.001469594828381]
distCoeffs2 = np.array([-0.363675899898510  , 0.129399782307601,
                        0.005737642620431   ,0.004472487457913, 0.011881211743883])
# distCoeffs2 = np.array([-0.426243385650803, 0.630959102496925,
#                         0.002912859612620, 0.001469594828381, -1.016065433982113])  # 右边相机的畸变参数
cameraMatrix1 = mtx
cameraMatrix2 = mtx_r

P1 = None  # project matrix
P2 = None

rt1 = None  # R|T matrix
rt2 = None

'''
# some ERROR 
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
    (640, 480), R, T)
'''


def getrtMtx(rvec, tvec):
    rmtx, _ = cv2.Rodrigues(rvec)  # 旋转向量和旋转矩阵的转换
    return np.hstack((rmtx, tvec))  # 垂直排布


def computeProjectMtx(undistort=False):  # 计算投影矩阵p1、p2
    global P1, P2, cameraMatrix1, cameraMatrix2, rt1, rt2
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2) * 25

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    objpoints_r = []
    imgpoints_r = []

    images = glob.glob('../stereo512/L1.jpg')  # 返回所有匹配的文件路径列表
    images_r = glob.glob('../stereo512/R1.jpg')

    for fname, fname_r in zip(images, images_r):  # 将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表

        img = cv2.imread(fname)
        img_r = cv2.imread(fname_r)

        if undistort:
            img = undistortImage(img, cameraMatrix1, distCoeffs1)
            img_r = undistortImage(img_r, cameraMatrix2, distCoeffs2)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 7), None)
        # print('corners',corners)
        ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9, 7), None)

        # If found, add object points, image points (after refining them)
        if ret == True and ret_r == True:
            objpoints.append(objp)
            objpoints_r.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        criteria)
            corners2_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1),
                                          criteria)
            imgpoints.append(corners2)
            imgpoints_r.append(corners2_r)

    ret, rotation, translation = cv2.solvePnP(objpoints[0], imgpoints[0],
                                              cameraMatrix1, distCoeffs1)

    ret, rotation_r, translation_r = cv2.solvePnP(objpoints[0], imgpoints_r[0],
                                                  cameraMatrix2, distCoeffs2)

    rt1 = getrtMtx(rotation, translation)
    rt2 = getrtMtx(rotation_r, translation_r)

    P1 = np.dot(cameraMatrix1, rt1)
    P2 = np.dot(cameraMatrix2, rt2)

    l = imgpoints[0].reshape(63, 2).T
    r = imgpoints_r[0].reshape(63, 2).T

    print("left cam pixel\n", l.shape, l)
    print("right cam pixel\n", r.shape, r)

    print("P1\n", P1, "\nP2:\n", P2)
    p4d = cv2.triangulatePoints(P1, P2, l, r)
    print("left camear p4d\n", p4d / p4d[-1])

    # check rt1
    rtmtxl_homo = np.vstack((rt1, np.array([0, 0, 0, 1])))
    obj_homo = cv2.convertPointsToHomogeneous(objpoints[0]).reshape(63, 4).T
    print("P*RT:\n", np.dot(rtmtxl_homo, obj_homo))


# def getImagePoints(undistort=False):  # 获得棋盘格格点像素坐标
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#
#     # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
#     objp = np.zeros((7 * 9, 3), np.float32)
#     objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2)
#
#     # Arrays to store object points and image points from all the images.
#     objpoints = []  # 3d point in real world space
#     imgpoints = []  # 2d points in image plane.
#
#     objpoints_r = []
#     imgpoints_r = []
#
#     images = glob.glob('../stereo512/left.bmp')
#     images_r = glob.glob('../stereo512/right.bmp')
#
#     images.sort()
#     images_r.sort()
#
#     for fname, fname_r in zip(images, images_r):
#         img = cv2.imread(fname)
#         img_r = cv2.imread(fname_r)
#
#         if undistort:
#             img = undistortImage(img, cameraMatrix1, distCoeffs1)
#             img_r = undistortImage(img_r, cameraMatrix2, distCoeffs2)
#
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
#
#         # Find the chess board corners
#         ret, corners = cv2.findChessboardCorners(gray, (9, 7), None)
#         # print('corners',corners)
#         ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9, 7), None)
#
#         # If found, add object points, image points (after refining them)
#         if ret == True and ret_r == True:
#             objpoints.append(objp)
#             objpoints_r.append(objp)
#
#             corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
#                                         criteria)
#             corners2_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1),
#                                           criteria)
#             imgpoints.append(corners2)
#             imgpoints_r.append(corners2_r)
#     l = imgpoints[0].reshape(63, 2).T  # 变为63行2列的矩阵
#     r = imgpoints_r[0].reshape(63, 2).T
#
#     return l, r


def undistortImage(img, _cam_mtx, _cam_dis):
    new_image = cv2.undistort(img, _cam_mtx, _cam_dis)
    return new_image


def getp3d(imgpoints_l, imgpoints_r):  # 得到三维坐标
    '''
    l : left  cam imgpoints  2 * N [[x1, x2,...xn], [y1, y2,...yn]]
    r : right cam imgpoints  2 * N
    return : 3 * N  [[x1...xn], [y1...yn], [z1...zn]]
    '''
    #平均值
    P1 = 1.0e+05*np.array([[-0.007775058586607 , -0.000515403607656 , -0.000001275869494],
           [0.000121877576532  ,-0.008000676123069  , 0.000000292385319],
           [0.004025588636983   ,0.003282327279465  , 0.000008550833505],
           [3.484749746604872 ,  3.560839727843039  , 0.006425585737809]]).T
    P2 =1.0e+05*np.array([[-0.007631545808921 , -0.000458781066530 , -0.000000975763736],
                  [0.000450122072459 , -0.007877909331349  , 0.000000515365562],
                  [0.004836740182155 ,  0.003089981957156  , 0.000008582138746],
                  [5.474013174103684   ,3.442491548083373  , 0.006426336918450]]).T
    #第一张图片的
    # P1 = 1.0e+05 * np.array([[-0.008320343342106  ,-0.000614528976069 , -0.000000218017179],
    #                          [ 0.000561890988905  ,-0.008154804657427 ,  0.000000126943571],
    #                          [0.005686105185318  , 0.003724825571603  , 0.000009996817185],
    #                          [2.628083796082778  , 2.650713289933748  , 0.004783131227986]]).T
    # P2 = 1.0e+05 * np.array([[-0.008246287660131  ,-0.000645337656814  ,-0.000000146986894],
    #                          [0.000849256212752  ,-0.008060574157125  , 0.000000307739694],
    #                          [0.006610626927769,   0.003491423501173  , 0.000009994182865],
    #                          [4.504538538422722 ,  2.590744195446289  , 0.004836873598297]]).T


    l = imgpoints_l
    r = imgpoints_r

    p4d = cv2.triangulatePoints(P1, P2, l, r)
    X = p4d / p4d[-1]  # 3d in chessboard coor 结果为四维的齐次坐标，除以最后一维变成非齐次坐标

    return X[:-1]


if __name__ == '__main__':
    objp = np.zeros((7 * 9, 3), np.float32)  # 生成7行9列每个元素都为0的矩阵
    objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1,
                                               2) * 25  # 输出至少是一个三维的向量 z[0]行数由0：9决定，列数为0：7（每一行相同） z[1]列数由0：7决定，行数由0：9决定（每一列相同）
    objp = objp.T  # 转置

    # computeProjectMtx(undistort=False)
    # check for cheeseboard
    # l, r = getImagePoints(undistort=False)  # 获得左右相机像素中的坐标
    #     # p3d = getp3d(l, r)  # 获得三维
    #     # print("cheese board corners p3d:\n", p3d)
    #     # print("MSE: ", np.sqrt(np.sum(np.square(p3d - objp) / 63, axis=1)))

    # experiments  ../stereo512/left.bmp
    #第三个为5G
    l = np.array([[530.0], [417.0]])
    r = np.array([[690.0], [423.5]])
    p3d = getp3d(l,r)
    print("p3d:")
    print(p3d)