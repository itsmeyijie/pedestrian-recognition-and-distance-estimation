import sys

sys.path.append("../")
import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from scipy import misc
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
import cv2
import distance111
from sklearn.svm import SVC
import time
import multiprocessing as mp

def add_person(name, video_src, path='./face_database/'):
    folder = os.path.exists(path + name)
    if not folder:
        os.makedirs(path + name)
    else:
        print('Directory existed.')
        exit(1)

    test_mode = "onet"
    thresh = [0.9, 0.6, 0.7]
    min_face_size = 24
    stride = 2
    slide_window = False
    shuffle = False
    # vis = True
    detectors = [None, None, None]
    prefix = ['../data/MTCNN_model/PNet_landmark/PNet', '../data/MTCNN_model/RNet_landmark/RNet',
              '../data/MTCNN_model/ONet_landmark/ONet']
    epoch = [18, 14, 16]
    model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
    PNet = FcnDetector(P_Net, model_path[0])
    detectors[0] = PNet
    RNet = Detector(R_Net, 24, 1, model_path[1])
    detectors[1] = RNet
    ONet = Detector(O_Net, 48, 1, model_path[2])
    detectors[2] = ONet
    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                   stride=stride, threshold=thresh, slide_window=slide_window)
    video_capture = cv2.VideoCapture(video_src)
    # video_capture.set(3, 340)
    # video_capture.set(4, 480)
    video_capture.set(3, 800)
    video_capture.set(4, 800)
    corpbbox = None
    count = 0
    frame_cut = 0
    while True:
        ret, frame = video_capture.read()
        if ret:
            image = np.array(frame)
            img_size = np.array(image.shape)[0:2]
            boxes_c, landmarks = mtcnn_detector.detect(image)
            if boxes_c.shape[0] > 0:
                bbox = boxes_c[0, :4]  # 检测出的人脸区域，左上x，左上y，右下x，右下y
                score = boxes_c[0, 4]  # 检测出人脸区域的得分
                corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]

                x1 = np.maximum(int(bbox[0]) - 16, 0)
                y1 = np.maximum(int(bbox[1]) - 16, 0)
                x2 = np.minimum(int(bbox[2]) + 16, img_size[1])
                y2 = np.minimum(int(bbox[3]) + 16, img_size[0])
                crop_img = image[y1:y2, x1:x2]
                scaled = misc.imresize(crop_img, (160, 160), interp='bilinear')

                img_name = '%s/%d.jpg' % (path + name, count)
                if frame_cut % 5 == 0:
                    cv2.imwrite(img_name, scaled)
                    count += 1

                cv2.rectangle(frame, (corpbbox[0], corpbbox[1]),
                              (corpbbox[2], corpbbox[3]), (0, 0, 255), 1)
                cv2.putText(frame, '{}/10 Remain'.format(count), (corpbbox[0], corpbbox[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (50, 255, 50), 2)

            cv2.imshow('Video', frame)
            if cv2.waitKey(20) == ord('q') or count >= 20:
                break
        frame_cut += 1

    video_capture.release()
    cv2.destroyAllWindows()


def face2database(picture_path, model_path, database_path, batch_size=90, image_size=160):
    # 提取特征到数据库
    # picture_path为人脸文件夹的所在路径
    # model_path为facenet模型路径
    # database_path为人脸数据库路径
    with tf.Graph().as_default():
        with tf.Session() as sess:
            dataset = facenet.get_dataset(picture_path)
            paths, labels = facenet.get_image_paths_and_labels(dataset)
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(model_path)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
            np.savez(database_path, emb=emb_array, lab=labels)
            print("数据库特征提取完毕！")
            # emb_array里存放的是图片特征，labels为对应的标签


def ClassifyTrainSVC(database_path, SVCpath):
    # database_path为人脸数据库
    # SVCpath为分类器储存的位置
    Database = np.load(database_path)
    name_lables = Database['lab']
    embeddings = Database['emb']
    name_unique = np.unique(name_lables)
    labels = []
    for i in range(len(name_lables)):
        for j in range(len(name_unique)):
            if name_lables[i] == name_unique[j]:
                labels.append(j)
    print('Training classifier')
    model = SVC(kernel='linear', probability=True)
    model.fit(embeddings, labels)
    with open(SVCpath, 'wb') as outfile:
        pickle.dump((model, name_unique), outfile)
        print('Saved classifier model to file "%s"' % SVCpath)


# 图片预处理阶段
def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def crop(image, random_crop, image_size):
    if image.shape[1] > image_size:
        sz1 = int(image.shape[1] // 2)
        sz2 = int(image_size // 2)
        if random_crop:
            diff = sz1 - sz2
            (h, v) = (np.random.randint(-diff, diff + 1), np.random.randint(-diff, diff + 1))
        else:
            (h, v) = (0, 0)
        image = image[(sz1 - sz2 + v):(sz1 + sz2 + v), (sz1 - sz2 + h):(sz1 + sz2 + h), :]
    return image


def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image


def load_image(image_old, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    if image_old.ndim == 2:
        image_old = to_rgb(image_old)
    if do_prewhiten:
        image_old = prewhiten(image_old)
    image_old = crop(image_old, do_random_crop, image_size)
    image_old = flip(image_old, do_random_flip)
    return image_old


def RTrecognization(facenet_model_path, SVCpath, database_path):
    # facenet_model_path为facenet模型路径
    # SVCpath为SVM分类模型路径
    # database_path为人脸库数据
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(facenet_model_path)
            with open(SVCpath, 'rb') as infile:
                (classifymodel, class_names) = pickle.load(infile)
            print('Loaded classifier model from file "%s"' % SVCpath)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            Database = np.load(database_path)

            test_mode = "onet"
            thresh = [0.9, 0.6, 0.7]
            min_face_size = 24
            stride = 2
            slide_window = False
            shuffle = False
            # vis = True
            detectors = [None, None, None]
            prefix = ['../data/MTCNN_model/PNet_landmark/PNet', '../data/MTCNN_model/RNet_landmark/RNet',
                      '../data/MTCNN_model/ONet_landmark/ONet']
            epoch = [18, 14, 16]
            model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
            PNet = FcnDetector(P_Net, model_path[0])
            detectors[0] = PNet
            RNet = Detector(R_Net, 24, 1, model_path[1])
            detectors[1] = RNet
            ONet = Detector(O_Net, 48, 1, model_path[2])
            detectors[2] = ONet
            mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                           stride=stride, threshold=thresh, slide_window=slide_window)

            # video_capture = cv2.VideoCapture('./video1.avi')
            # ret, frame = video_capture.read()
            video_capture1 = cv2.VideoCapture('rtsp://admin:zhang12345678@192.168.3.14:554/Streaming/Channels/1')
            # video_capture1 = cv2.VideoCapture('./test_video.avi')

            video_capture1.set(3, 340)
            video_capture1.set(4, 480)
            # video_writer = cv2.VideoWriter('./face_recognition.avi', cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'),
            #                                10, (frame.shape[1], frame.shape[0]))
            persons = os.listdir('./face_database/')
            persons.sort()
            corpbbox = None
            while True:
                t1 = cv2.getTickCount()
                ret1, frame1 = video_capture1.read()
                if ret1:
                    image = np.array(frame1)
                    img_size = np.array(image.shape)[0:2]
                    boxes_c, landmarks = mtcnn_detector.detect(image)
                    # print(boxes_c.shape)
                    # print(boxes_c)
                    # print(img_size)
                    for i in range(boxes_c.shape[0]):
                        bbox = boxes_c[i, :4]  # 检测出的人脸区域，左上x，左上y，右下x，右下y
                        score = boxes_c[i, 4]  # 检测出人脸区域的得分
                        corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]

                        x1 = np.maximum(int(bbox[0]) - 16, 0)
                        y1 = np.maximum(int(bbox[1]) - 16, 0)
                        x2 = np.minimum(int(bbox[2]) + 16, img_size[1])
                        y2 = np.minimum(int(bbox[3]) + 16, img_size[0])
                        crop_img = image[y1:y2, x1:x2]
                        scaled = misc.imresize(crop_img, (160, 160), interp='bilinear')
                        img = load_image(scaled, False, False, 160)
                        img = np.reshape(img, (-1, 160, 160, 3))
                        feed_dict = {images_placeholder: img, phase_train_placeholder: False}
                        embvecor = sess.run(embeddings, feed_dict=feed_dict)
                        embvecor = np.array(embvecor)
                        # 利用人脸特征与数据库中所有人脸进行一一比较的方法
                        # tmp=np.sqrt(np.sum(np.square(embvecor-Database['emb'][0])))
                        # tmp_lable=Database['lab'][0]
                        # for j in range(len(Database['emb'])):
                        #     t=np.sqrt(np.sum(np.square(embvecor-Database['emb'][j])))
                        #     if t<tmp:
                        #         tmp=t
                        #         tmp_lable=Database['lab'][j]
                        # print(tmp)

                        # 利用SVM对人脸特征进行分类
                        predictions = classifymodel.predict_proba(embvecor)
                        best_class_indices = np.argmax(predictions, axis=1)
                        tmp_lable = class_names[best_class_indices]
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        print(best_class_probabilities)
                        cv2.rectangle(frame1, (corpbbox[0], corpbbox[1]),
                                      (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)

                        if best_class_probabilities < 0.5:
                            tmp_lable = "others"
                            cv2.putText(frame1, '{0}'.format(tmp_lable), (corpbbox[0], corpbbox[1] - 2),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        else:
                            cv2.putText(frame1, '{0}'.format(persons[tmp_lable[0]]), (corpbbox[0], corpbbox[1] - 2),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    t2 = cv2.getTickCount()
                    t = (t2 - t1) / cv2.getTickFrequency()
                    fps = 1.0 / t
                    cv2.putText(frame1, 'Time_cost: {:.4f}s'.format(t) + '  ' + '{:.3f}fps'.format(fps), (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    print('fps: {:.3f}'.format(fps))
                    # cv2.putText(frame, 'Confidence: {:.2f}'.format(best_class_probabilities[0]), (10, 40),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                    for i in range(landmarks.shape[0]):
                        for j in range(len(landmarks[i]) // 2):
                            cv2.circle(frame1, (int(landmarks[i][2 * j]), int(int(landmarks[i][2 * j + 1]))), 2,
                                       (0, 0, 255))
                            # time end
                    cv2.namedWindow('face1', 0)
                    cv2.resizeWindow('face1', 1000, 1000)
                    # video_writer.write(frame)
                    cv2.imshow('face1', frame1)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:

                    print('device not find')
                    break
            video_capture1.release()
            cv2.destroyAllWindows()


def RTrecognization1(facenet_model_path, SVCpath, database_path):
    # facenet_model_path为facenet模型路径
    # SVCpath为SVM分类模型路径
    # database_path为人脸库数据
   # t1 = time.monotonic()
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model

            print('Loading feature extraction model')
            facenet.load_model(facenet_model_path)
            with open(SVCpath, 'rb') as infile:
                (classifymodel, class_names) = pickle.load(infile)
            print('Loaded classifier model from file "%s"' % SVCpath)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            Database = np.load(database_path)

            test_mode = "onet"
            thresh = [0.9, 0.6, 0.7]
            min_face_size = 24
            stride = 2
            slide_window = False
            shuffle = False
            # vis = True
            detectors = [None, None, None]
            prefix = ['../data/MTCNN_model/PNet_landmark/PNet', '../data/MTCNN_model/RNet_landmark/RNet',
                      '../data/MTCNN_model/ONet_landmark/ONet']
            epoch = [18, 14, 16]
            model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
            PNet = FcnDetector(P_Net, model_path[0])
            detectors[0] = PNet
            RNet = Detector(R_Net, 24, 1, model_path[1])
            detectors[1] = RNet
            ONet = Detector(O_Net, 48, 1, model_path[2])
            detectors[2] = ONet
            mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                           stride=stride, threshold=thresh, slide_window=slide_window)

            # video_capture = cv2.VideoCapture('./test_video.avi')
            flag1 = False
            flag2 = False
            trackingbox1  = []
            trackingbox2 = []
            trackframe1 = None
            trackframe2 = None
            personname = None
            initial1 = False
            initial2 = False
            tracker1 = cv2.TrackerMedianFlow_create()
            tracker2 = cv2.TrackerMedianFlow_create()

            video_capture1 = cv2.VideoCapture('rtsp://admin:zhang12345678@192.168.3.14:554/Streaming/Channels/1')
            video_capture2 = cv2.VideoCapture('rtsp://admin:zhang12345678@192.168.3.9:554/Streaming/Channels/1')


            # video_capture.set(3, 340)
            # video_capture.set(4, 480)
            video_capture1.set(3, 800)
            video_capture1.set(4, 800)
            video_capture2.set(3, 800)
            video_capture2.set(4, 800)
            persons = os.listdir('./face_database/')
            persons.sort()
            corpbbox = None
            t2 = time.monotonic()
           # print(t2 - t1)
            while True:
                t1 = cv2.getTickCount()

                ret1, frame1 = video_capture1.read()
                ret2, frame2 = video_capture2.read()


                if ret1 and ret2:
                    t00 = time.monotonic()
                    image1 = np.array(frame1)
                    image2 = np.array(frame2)
                    img_size = np.array(image1.shape)[0:2]
                    boxes_c1, landmarks1 = mtcnn_detector.detect(image1)
                    img_size = np.array(image2.shape)[0:2]
                    boxes_c2, landmarks2 = mtcnn_detector.detect(image2)
                    #t22 = time.monotonic()
                    #print(t22 - t11)
                    if boxes_c1.shape[0] > 0 and boxes_c2.shape[0] > 0 and flag1 == False and flag2 == False:
                        print("检测到了人脸")
                        # print(boxes_c.shape)
                        # print(boxes_c)
                        # print(img_size)
                        flag1 = True
                        flag2 = True
                        t2 = cv2.getTickCount()
                        t = (t2 - t1) / cv2.getTickFrequency()
                        fps = 1.0 / t
                        bbox1 = boxes_c1[0, :4]
                        bbox2 = boxes_c2[0, :4]
                        #t11 = time.monotonic()
                        trackingbox1 =  (bbox1[0],bbox1[1],bbox1[2]-bbox1[0],bbox1[3]-bbox1[1])
                        trackingbox2 =  (bbox2[0],bbox2[1],bbox2[2]-bbox2[0],bbox2[3]-bbox2[1])
                        trackframe1 = frame1
                        trackframe2 = frame2

                        # print(bbox1)
                        # print(bbox2)
                        l = np.array([[float(bbox1[0]+bbox1[2])/2.0], [float(bbox1[1]+bbox1[3])/2.0]])
                        r = np.array([[float(bbox2[0]+bbox2[2])/2.0], [float(bbox2[1]+bbox2[3])/2.0]])
                        t11 = time.monotonic()
                        p3d = distance111.getp3d(l, r)
                        t22 = time.monotonic()
                        print("1:")
                        print(t22-t11)
                        t33 = time.monotonic()
                        corpbbox = [int(bbox1[0]), int(bbox1[1]), int(bbox1[2]), int(bbox1[3])]
                        x1 = np.maximum(int(bbox1[0]) - 16, 0)
                        y1 = np.maximum(int(bbox1[1]) - 16, 0)
                        x2 = np.minimum(int(bbox1[2]) + 16, img_size[1])
                        y2 = np.minimum(int(bbox1[3]) + 16, img_size[0])
                        crop_img = image1[y1:y2, x1:x2]
                        scaled = misc.imresize(crop_img, (160, 160), interp='bilinear')
                        img = load_image(scaled, False, False, 160)
                        img = np.reshape(img, (-1, 160, 160, 3))
                        feed_dict = {images_placeholder: img, phase_train_placeholder: False}
                        embvecor = sess.run(embeddings, feed_dict=feed_dict)
                        embvecor = np.array(embvecor)
                        # 利用人脸特征与数据库中所有人脸进行一一比较的方法
                        # tmp=np.sqrt(np.sum(np.square(embvecor-Database['emb'][0])))
                        # tmp_lable=Database['lab'][0]
                        # for j in range(len(Database['emb'])):
                        #     t=np.sqrt(np.sum(np.square(embvecor-Database['emb'][j])))
                        #     if t<tmp:
                        #         tmp=t
                        #         tmp_lable=Database['lab'][j]
                        # print(tmp)

                        # 利用SVM对人脸特征进行分类
                        predictions = classifymodel.predict_proba(embvecor)
                        best_class_indices = np.argmax(predictions, axis=1)
                        tmp_lable = class_names[best_class_indices]
                        best_class_probabilities = predictions[
                            np.arange(len(best_class_indices)), best_class_indices]
                        # print(best_class_probabilities)
                        cv2.rectangle(frame1, (corpbbox[0], corpbbox[1]),
                                      (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)

                        if best_class_probabilities < 0.2:
                            tmp_lable = "others"
                            personname = "others"
                            cv2.putText(frame1, '{0}'.format(tmp_lable), (corpbbox[0], corpbbox[1] - 2),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        else:
                            cv2.putText(frame1, '{0}'.format(persons[tmp_lable[0]]), (corpbbox[0], corpbbox[1] - 2),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            print(persons[tmp_lable[0]])
                            personname = persons[tmp_lable[0]]
                        t44 = time.monotonic()
                        print("2:" )
                        print(t44 - t33)
                        #print("1:" + str(t22 - t11))
                        #print(p3d)


                        cv2.putText(frame1, '{:.4f} {:.3f}'.format(t, fps) +
                                    ' recognition result:The distance from K1335 is {:.3f}m'.format(-p3d[2][0] / 1000),
                                    (200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2)

                        print('The distance from K1335 is {:.3f}m'.format(-p3d[2][0]/1000))

                    # cv2.putText(frame, 'Confidence: {:.2f}'.format(best_class_probabilities[0]), (10, 40),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                        for i in range(landmarks1.shape[0]):
                            for j in range(len(landmarks1[i]) // 2):
                                cv2.circle(frame1, (int(landmarks1[i][2 * j]), int(int(landmarks1[i][2 * j + 1]))), 2,
                                       (0, 0, 255))
                            # time end
                    elif(flag1 == True and flag2==True):

                        if(initial1 ==False and initial2 == False ):
                            t55=time.monotonic()
                            ok1 = tracker1.init(trackframe1, trackingbox1)
                            ok2 = tracker2.init(trackframe2, trackingbox2)
                            initial1 = True
                            initial2 = True
                            t66=time.monotonic()
                            print("3")
                            print(t66-t55)
                        else:
                            t77 = time.monotonic()
                            ok1, bbbox1 = tracker1.update(frame1)
                            ok2, bbbox2 = tracker2.update(frame2)
                            t88=time.monotonic()
                            print("4")
                            print(t88 - t77)
                            if ok1:
                                # Tracking success 跟踪成功
                                #print("追踪成功")
                                p1 = (int(bbbox1[0])-16, int(bbbox1[1]-16))
                                p2 = (int(bbbox1[0] + bbbox1[2])+16, int(bbbox1[1] + bbbox1[3])+16)
                                cv2.rectangle(frame1, p1, p2, (255, 0, 0), 2, 1)
                                cv2.putText(frame1, '{0}'.format(personname), p1,
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                if ok2:
                                 #   l = np.array(
                                 #       [[float(bbbox1[0] + bbbox1[2]) / 2.0], [float(bbbox1[1] + bbbox1[3]) / 2.0]])
                                    l = np.array([[bbbox1[0] + float(bbbox1[2])/2.0], [bbbox1[1] + float(bbbox1[3] / 2.0)]])
                                    r = np.array([[bbbox2[0] + float(bbbox2[2]) / 2.0], [bbbox2[1] + float(bbbox2[3] / 2.0)]])
                                 #   r = np.array(
                                 #       [[float(bbbox2[0] + bbbox2[2]) / 2.0], [float(bbbox2[1] + bbbox2[3]) / 2.0]])
                                    p3d = distance111.getp3d(l, r)
                                    cv2.putText(frame1, '{:.4f} {:.3f}'.format(t, fps) +
                                                ' Tracking result:The distance from K1335 is {:.3f}m'.format(abs(p3d[2][0]) / 1000),
                                                (200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2)
                                    print('The distance from K1335 is {:.3f}m'.format(-p3d[2][0] / 1000))

                            else:  # 跟踪失败
                                # Tracking failure
                                print("追踪失败")
                                cv2.putText(frame1, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                            (0, 0, 255),
                                            2)
                                flag1 = False
                                flag2 = False
                                initial1 = False
                                initial2 = False


                    cv2.namedWindow('face', 0)
                    cv2.resizeWindow('face', 1000, 1000)
                    cv2.imshow('face', frame1)
                    t99 = time.monotonic()
               #     cv2.imshow('face', frame2)
                    print('Time_cost: %s' % (t99-t00))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:

                    print('device not find')
                    break
            video_capture1.release()
            video_capture2.release()
            cv2.destroyAllWindows()

#获得视频放入队列中
def image_put(q, user, pwd, ip, channel=1):
    cap = cv2.VideoCapture("rtsp://%s:%s@%s//Streaming/Channels/%d" % (user, pwd, ip, channel))
    if cap.isOpened():
        print('HIKVISION')
    else:
        cap = cv2.VideoCapture("rtsp://%s:%s@%s/cam/realmonitor?channel=%d&subtype=0" % (user, pwd, ip, channel))
        print('DaHua')

    while True:
        q.put(cap.read()[1])
        q.get() if q.qsize() > 1 else time.sleep(0.01)
#取出队列中的数据进行处理将结果放入输出队列中
def image_process(q, r_q, facenet_model_path, SVCpath, database_path):
    global personname
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model

            print('Loading feature extraction model')
            facenet.load_model(facenet_model_path)
            with open(SVCpath, 'rb') as infile:
                (classifymodel, class_names) = pickle.load(infile)
            print('Loaded classifier model from file "%s"' % SVCpath)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            Database = np.load(database_path)

            test_mode = "onet"
            thresh = [0.9, 0.6, 0.7]
            min_face_size = 24
            stride = 2
            slide_window = False
            shuffle = False
            # vis = True
            detectors = [None, None, None]
            prefix = ['../data/MTCNN_model/PNet_landmark/PNet', '../data/MTCNN_model/RNet_landmark/RNet',
                      '../data/MTCNN_model/ONet_landmark/ONet']
            epoch = [18, 14, 16]
            model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
            PNet = FcnDetector(P_Net, model_path[0])
            detectors[0] = PNet
            RNet = Detector(R_Net, 24, 1, model_path[1])
            detectors[1] = RNet
            ONet = Detector(O_Net, 48, 1, model_path[2])
            detectors[2] = ONet
            mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                           stride=stride, threshold=thresh, slide_window=slide_window)

            flag1 = False
            trackingbox1  = []
            trackframe1 = None
            personname = None
            initial1 = False
            tracker1 = cv2.TrackerMedianFlow_create()
            video_capture1 = q.get()
            # video_capture.set(3, 340)
            # video_capture.set(4, 480)
            video_capture1.set(3, 800)
            video_capture1.set(4, 800)

            persons = os.listdir('./face_database/')
            persons.sort()
            corpbbox = None
            while True:
                t1 = cv2.getTickCount()
                frame1 = video_capture1
                if True:
                    t00 = time.monotonic()
                    image1 = np.array(frame1)

                    img_size = np.array(image1.shape)[0:2]
                    boxes_c1, landmarks1 = mtcnn_detector.detect(image1)

                    if boxes_c1.shape[0] > 0 and flag1 == False:
                        print("检测到了人脸")
                        # print(boxes_c.shape)
                        # print(boxes_c)
                        # print(img_size)
                        flag1 = True
                        t2 = cv2.getTickCount()
                        t = (t2 - t1) / cv2.getTickFrequency()
                        fps = 1.0 / t
                        bbox1 = boxes_c1[0, :4]
                        #t11 = time.monotonic()
                        trackingbox1 =  (bbox1[0],bbox1[1],bbox1[2]-bbox1[0],bbox1[3]-bbox1[1])
                        trackframe1 = frame1


                        # print(bbox1)
                        # print(bbox2)
                        l = np.array([[float(bbox1[0]+bbox1[2])/2.0], [float(bbox1[1]+bbox1[3])/2.0]])
                        # r = np.array([[float(bbox2[0]+bbox2[2])/2.0], [float(bbox2[1]+bbox2[3])/2.0]])
                        #
                        # p3d = distance111.getp3d(l, r)

                        corpbbox = [int(bbox1[0]), int(bbox1[1]), int(bbox1[2]), int(bbox1[3])]
                        x1 = np.maximum(int(bbox1[0]) - 16, 0)
                        y1 = np.maximum(int(bbox1[1]) - 16, 0)
                        x2 = np.minimum(int(bbox1[2]) + 16, img_size[1])
                        y2 = np.minimum(int(bbox1[3]) + 16, img_size[0])
                        crop_img = image1[y1:y2, x1:x2]
                        scaled = misc.imresize(crop_img, (160, 160), interp='bilinear')
                        img = load_image(scaled, False, False, 160)
                        img = np.reshape(img, (-1, 160, 160, 3))
                        feed_dict = {images_placeholder: img, phase_train_placeholder: False}
                        embvecor = sess.run(embeddings, feed_dict=feed_dict)
                        embvecor = np.array(embvecor)
                        # 利用人脸特征与数据库中所有人脸进行一一比较的方法
                        # tmp=np.sqrt(np.sum(np.square(embvecor-Database['emb'][0])))
                        # tmp_lable=Database['lab'][0]
                        # for j in range(len(Database['emb'])):
                        #     t=np.sqrt(np.sum(np.square(embvecor-Database['emb'][j])))
                        #     if t<tmp:
                        #         tmp=t
                        #         tmp_lable=Database['lab'][j]
                        # print(tmp)

                        # 利用SVM对人脸特征进行分类
                        predictions = classifymodel.predict_proba(embvecor)
                        best_class_indices = np.argmax(predictions, axis=1)
                        tmp_lable = class_names[best_class_indices]
                        best_class_probabilities = predictions[
                            np.arange(len(best_class_indices)), best_class_indices]
                        # print(best_class_probabilities)
                        cv2.rectangle(frame1, (corpbbox[0], corpbbox[1]),
                                      (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)

                        if best_class_probabilities < 0.2:
                            tmp_lable = "others"
                            personname = "others"
                            cv2.putText(frame1, '{0}'.format(tmp_lable), (corpbbox[0], corpbbox[1] - 2),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        else:
                            cv2.putText(frame1, '{0}'.format(persons[tmp_lable[0]]), (corpbbox[0], corpbbox[1] - 2),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            print(persons[tmp_lable[0]])
                            personname = persons[tmp_lable[0]]
                        r_q.put((video_capture1, corpbbox, personname))


                        # cv2.putText(frame1, '{:.4f} {:.3f}'.format(t, fps) +
                        #             ' recognition result:The distance from K1335 is {:.3f}m'.format(-p3d[2][0] / 1000),
                        #             (200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2)
                        #
                        # print('The distance from K1335 is {:.3f}m'.format(-p3d[2][0]/1000))

                    # cv2.putText(frame, 'Confidence: {:.2f}'.format(best_class_probabilities[0]), (10, 40),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                        for i in range(landmarks1.shape[0]):
                            for j in range(len(landmarks1[i]) // 2):
                                cv2.circle(frame1, (int(landmarks1[i][2 * j]), int(int(landmarks1[i][2 * j + 1]))), 2,
                                       (0, 0, 255))
                            # time end
                    elif(flag1 == True and flag2==True):

                        if(initial1 ==False and initial2 == False ):

                            ok1 = tracker1.init(trackframe1, trackingbox1)

                            initial1 = True

                        else:
                            ok1, bbbox1 = tracker1.update(frame1)
                            if ok1:
                                # Tracking success 跟踪成功
                                #print("追踪成功")
                                p1 = (int(bbbox1[0])-16, int(bbbox1[1]-16))
                                p2 = (int(bbbox1[0] + bbbox1[2])+16, int(bbbox1[1] + bbbox1[3])+16)
                                box = [int(bbox1[0]), int(bbox1[1]), int(bbox1[2]), int(bbox1[3])]
                                r_q.put((video_capture1, box, personname))
                                # cv2.rectangle(frame1, p1, p2, (255, 0, 0), 2, 1)
                                # cv2.putText(frame1, '{0}'.format(personname), p1,
                                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                # if ok2:
                                #  #   l = np.array(
                                #  #       [[float(bbbox1[0] + bbbox1[2]) / 2.0], [float(bbbox1[1] + bbbox1[3]) / 2.0]])
                                #     l = np.array([[bbbox1[0] + float(bbbox1[2])/2.0], [bbbox1[1] + float(bbbox1[3] / 2.0)]])
                                #     r = np.array([[bbbox2[0] + float(bbbox2[2]) / 2.0], [bbbox2[1] + float(bbbox2[3] / 2.0)]])
                                #  #   r = np.array(
                                #  #       [[float(bbbox2[0] + bbbox2[2]) / 2.0], [float(bbbox2[1] + bbbox2[3]) / 2.0]])
                                #     p3d = distance111.getp3d(l, r)
                                #     cv2.putText(frame1, '{:.4f} {:.3f}'.format(t, fps) +
                                #                 ' Tracking result:The distance from K1335 is {:.3f}m'.format(abs(p3d[2][0]) / 1000),
                                #                 (200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 2)
                                #     print('The distance from K1335 is {:.3f}m'.format(-p3d[2][0] / 1000))

                            else:  # 跟踪失败
                                # Tracking failure
                                print("追踪失败")
                                r_q.put(video_capture1,False,False)
                                flag1 = False
                                flag2 = False
                                initial1 = False
                                initial2 = False



                    cv2.namedWindow('face', 0)
                    cv2.resizeWindow('face', 1000, 1000)
                    cv2.imshow('face', frame1)

               #     cv2.imshow('face', frame2)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:

                    print('device not find')
                    break
            video_capture1.release()

            cv2.destroyAllWindows()


def image_get(q, window_name):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    while True:
        (frame,box,personname) = q.get()
        if box == False:
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 0, 255),
                        2)
            return
        cv2.rectangle(frame, (box[0], box[1]),
                      (box[2], box[3]), (255, 0, 0), 1)
        cv2.putText(frame, '{0}'.format(personname), (box[0], box[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow(window_name, frame)
        cv2.waitKey(1)
#运行
def run_multi_camera():
    # user_name, user_pwd = "admin", "password"
    user_name, user_pwd = "admin", "zhang12345678"
    camera_ip_l = [
        "192.168.3.14",  # ipv4
        "192.168.3.9",  # ipv6
    ]

    mp.set_start_method(method='spawn')  # init
    origin_queues = [mp.Queue(maxsize=4) for _ in camera_ip_l] #原始帧队列
    result_queues = [mp.Queue(maxsize=4) for _ in camera_ip_l] #处理之后的队列

    processes = [
        mp.Process(target=image_put, args=(origin_queues, user_name, user_pwd, camera_ip_l)),
        mp.Process(target=image_process, args=(origin_queues,result_queues, model_path, SVCpath, database_path)),
        mp.Process(target=image_get, args=(result_queues, camera_ip_l)),
    ]
    #mp.Process(target=image_get, args=(origin_queues, camera_ip))

    for process in processes:
        process.daemon = True
        process.start()
    for process in processes:
        process.join()
if __name__ == "__main__":
    picture_path = "./face_database"
    model_path = "./face_models/20180402-114759"
    database_path = "./Database.npz"
    SVCpath = "./face_models/SVCmodel.pkl"

    # # 添加一个人到数据库
    # name = 'li_zhijun'
    # video_src = '../../project1/lizhijun.avi'
    # add_person(name, video_src)

    # face2database(picture_path, model_path, database_path)#第一步 将图像信息embedding
    # ClassifyTrainSVC(database_path, SVCpath)  # 第二步 将第一步的结果进行训练SVC
    #RTrecognization(model_path, SVCpath, database_path)  # 第三步 实时检测
    run_multi_camera()
    # RTrecognization1(model_path, SVCpath, database_path)

