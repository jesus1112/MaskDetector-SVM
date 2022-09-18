import cv2
from mtcnn import MTCNN
from os import path
import json
import numpy as np
from feature_extraction import cfq, getFaceRegionsofInterest, faceDetector

dir_SAVE = './'
save_image_results = False
drawResults = False
save_json = True
totaldetectedfaces = 0

def area(xmin, ymin, xmax, ymax):
    dx = xmax - xmin
    dy = ymax - ymin
    if (dx >= 0) and (dy >= 0):
        return dx * dy
    else:
        return False


def verify_face(res):
    if len(res) == 0:
        return None
    else:
        # Selects only the face with the greatest bounding box area
        A_max = 0
        GT_index = 0
        for j in range(len(res)):
            bndbox_fd = res[j]['box']
            A_j = area(bndbox_fd[0], bndbox_fd[1],
                       bndbox_fd[0] + bndbox_fd[2], bndbox_fd[1] + bndbox_fd[3])
            if A_j > A_max and A_j > 2200:
                A_max = A_j
                GT_index = j
        if A_max == 0:
            return None
        else:
            return res[GT_index]


def CelebA_feature_extraction(detector, dir_CelebA):
    j = 0
    totaldetectedfaces = 0
    dirImage1 = "{0}{1:06d}{2}".format(dir_CelebA, j, '.jpg')
    fileExists1 = path.exists(dirImage1)

    info_samples = list()
    cfq_list = list()
    while j < 500:
        if fileExists1:
            im = cv2.cvtColor(cv2.imread(dirImage1), cv2.COLOR_BGR2RGB)
            [im2, res] = faceDetector(im, detector, drawResults)
            totaldetectedfaces += len(res)
            res_unit = list()
            res_unit.append(verify_face(res))
            if res_unit[0] is not None:
                [im3, ROI_FH, ROI_BE, ROI_NM] = getFaceRegionsofInterest(im, res_unit, drawResults)
                bndbx_fd = res_unit[0]['box']
                face_info = {
                    'gt_index': 0,
                    'name': 'No_Mask',
                    'bndbx_gt': None,
                    'fd_index': 0,
                    'bndbx_fd': [(bndbx_fd[0], bndbx_fd[1]),
                                 (bndbx_fd[0] + bndbx_fd[2], bndbx_fd[1] + bndbx_fd[3])],
                    'cqf': (cfq(ROI_FH[0], ROI_BE[0], ROI_NM[0])),
                    'status': 'OK'
                }
                cfq_list.append(cfq(ROI_FH[0], ROI_BE[0], ROI_NM[0]))
            else:
                face_info = {
                    'gt_index': None,
                    'name': 'No_Mask',
                    'bndbx_gt': None,
                    'fd_index': None,
                    'bndbx_fd': None,
                    'cqf': None,
                    'status': 'Error: no face detected'
                }
            image_face_info = list()
            image_face_info.append(face_info)
            image_data = {'filename': "{0:06d}".format(j),
                          'data': image_face_info
                          }
            info_samples.append(image_data)
            dirImageSave = "{0}{1:06d}{2}".format(dir_SAVE, j, '.jpg')
            if save_image_results:
                cv2.imwrite(dirImageSave, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
                print('Image no mask {} saved'.format(j))

        # Verifies if there's another image available
        j = j + 1
        dirImage1 = "{0}{1:06d}{2}".format(dir_CelebA, j, '.jpg')
        fileExists1 = path.exists(dirImage1)
        print('CelebA iteration {}'.format(j))

    if save_json:
        json.dump(info_samples, open("./info_samples_CelebA.json", "w"))


def CelebA_get_features_array(info_samples_file='./info_samples_FMD.json'):
    # Opens json file of CelebA data
    f = open(info_samples_file, 'r')
    jsonCelebAstr = f.read()
    jsonCelebAlist = json.loads(jsonCelebAstr)
    fd_errors_CelebA = 0  # Face detection stage error counter
    fd_ok_CelebA = 0  # Face detection stage OK counter
    nomaskclassCelebA = list()

    # Analyze every element of each image
    for i in range(len(jsonCelebAlist)):
        sampleElements = jsonCelebAlist[i]['data']
        for j in range(len(sampleElements)):
            currentElement = sampleElements[j]
            if currentElement is None:
                # Error: no face detected in this image
                fd_errors_CelebA = fd_errors_CelebA + 1
            elif 'OK' in currentElement['status']:
                fd_ok_CelebA = fd_ok_CelebA + 1
                if currentElement['name'] == 'No_Mask':
                    nomaskclassCelebA.append(currentElement['cqf'])
            else:
                fd_errors_CelebA = fd_errors_CelebA + 1

    nomaskCelebA = np.array(nomaskclassCelebA)

    featuresCelebA = nomaskCelebA
    targetsCelebA = np.zeros(len(nomaskclassCelebA)) - 1

    return featuresCelebA, targetsCelebA
