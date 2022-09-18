import cv2
from os import path
import json
import numpy as np
from feature_extraction import cfq, faceDetector, getFaceRegionsofInterest

dir_MFN_SAVE = 'C:/wsPDI/PDI_A6/data/'
save_image_results = False
drawResults = False
save_json = True


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
            if A_j > A_max and A_j > 200000:
                A_max = A_j
                GT_index = j
        if A_max == 0:
            return None
        else:
            return res[GT_index]


def MFN_feature_extraction(detector, dir_MFN):
    j = 0
    totaldetectedfaces = 0
    dirImage1 = "{0}{1:05d}{2}{3}".format(dir_MFN, j, '_Mask', '.jpg')
    dirImage2 = "{0}{1:05d}{2}{3}".format(dir_MFN, j, '_Mask_Chin', '.jpg')
    fileExists1 = path.exists(dirImage1)
    fileExists2 = path.exists(dirImage2)

    info_samples = list()
    while j < 980:
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
                    'name': 'Mask',
                    'bndbx_gt': None,
                    'fd_index': 0,
                    'bndbx_fd': [(bndbx_fd[0], bndbx_fd[1]),
                                 (bndbx_fd[0] + bndbx_fd[2], bndbx_fd[1] + bndbx_fd[3])],
                    'cqf': (cfq(ROI_FH[0], ROI_BE[0], ROI_NM[0])),
                    'status': 'OK'
                }
            else:
                face_info = {
                    'gt_index': None,
                    'name': 'Mask',
                    'bndbx_gt': None,
                    'fd_index': None,
                    'bndbx_fd': None,
                    'cqf': None,
                    'status': 'Error: no face detected'
                }
            image_face_info = list()
            image_face_info.append(face_info)
            image_data = {'filename': "{0:05d}_Mask".format(j),
                          'data': image_face_info
                          }
            info_samples.append(image_data)
            dirImageSave = "{0}{1:05d}{2}{3}".format(dir_MFN_SAVE, j, '_Mask', '.jpg')
            if save_image_results:
                cv2.imwrite(dirImageSave, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
                print('Image mask {} saved'.format(j))

        if fileExists2:
            im = cv2.cvtColor(cv2.imread(dirImage2), cv2.COLOR_BGR2RGB)
            [im2, res] = faceDetector(im, detector, drawResults)
            totaldetectedfaces += len(res)
            res_unit = list()
            res_unit.append(verify_face(res))
            if res_unit[0] is not None:
                [im3, ROI_FH, ROI_BE, ROI_NM] = getFaceRegionsofInterest(im, res_unit, drawResults)
                bndbx_fd = res_unit[0]['box']
                face_info = {
                    'gt_index': 0,
                    'name': 'Mask_Chin',
                    'bndbx_gt': None,
                    'fd_index': 0,
                    'bndbx_fd': [(bndbx_fd[0], bndbx_fd[1]),
                                 (bndbx_fd[0] + bndbx_fd[2], bndbx_fd[1] + bndbx_fd[3])],
                    'cqf': (cfq(ROI_FH[0], ROI_BE[0], ROI_NM[0])),
                    'status': 'OK'
                }
            else:
                face_info = {
                    'gt_index': None,
                    'name': 'Mask_Chin',
                    'bndbx_gt': None,
                    'fd_index': None,
                    'bndbx_fd': None,
                    'cqf': None,
                    'status': 'Error: no face detected'
                }
            image_face_info = list()
            image_face_info.append(face_info)
            image_data = {'filename': "{0:05d}_Mask_Chin".format(j),
                          'data': image_face_info
                          }
            info_samples.append(image_data)
            dirImageSave = "{0}{1:05d}{2}{3}".format(dir_MFN_SAVE, j, '_Mask_Chin', '.jpg')
            if save_image_results:
                cv2.imwrite(dirImageSave, cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
                print('Image mask chin {} saved'.format(j))

        if fileExists1 or fileExists2:
            print('MFN iteration {}'.format(j))
        # Verifies if there's another image available
        j = j + 1
        dirImage1 = "{0}{1:05d}{2}{3}".format(dir_MFN, j, '_Mask', '.jpg')
        dirImage2 = "{0}{1:05d}{2}{3}".format(dir_MFN, j, '_Mask_Chin', '.jpg')
        fileExists1 = path.exists(dirImage1)
        fileExists2 = path.exists(dirImage2)

    if save_json:
        json.dump(info_samples, open("./info_samples_MFN.json", "w"))


def MFN_get_features_array(info_samples_file='./info_samples_MFN.json'):
    # Opens json file of MFD data
    f = open(info_samples_file, 'r')
    jsonMFDstr = f.read()
    jsonMFDlist = json.loads(jsonMFDstr)
    fd_errors_MFD = 0  # Face detection stage error counter
    fd_ok_MFD = 0  # Face detection stage OK counter
    maskclassMFD = list()
    nomaskclassMFD = list()

    # Analyze every element of each image
    for i in range(len(jsonMFDlist)):
        sampleElements = jsonMFDlist[i]['data']
        for j in range(len(sampleElements)):
            currentElement = sampleElements[j]
            if currentElement is None:
                # Error: no face detected in this image
                fd_errors_MFD = fd_errors_MFD + 1
            elif 'OK' in currentElement['status']:
                fd_ok_MFD = fd_ok_MFD + 1
                if currentElement['name'] == 'Mask':
                    maskclassMFD.append(currentElement['cqf'])
                elif currentElement['name'] == 'Mask_Chin':
                    nomaskclassMFD.append(currentElement['cqf'])
            else:
                fd_errors_MFD = fd_errors_MFD + 1

    maskMFD = np.array(maskclassMFD)
    nomaskMFD = np.array(nomaskclassMFD)

    featuresMFN = np.concatenate((maskMFD, nomaskMFD), axis=0)
    targetsMFN = np.concatenate((np.zeros(len(maskclassMFD))+1, np.zeros(len(nomaskclassMFD))-1))

    return featuresMFN, targetsMFN

