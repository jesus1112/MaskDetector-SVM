import cv2
from mtcnn import MTCNN
import numpy as np
from os import path
import xmltodict, json
from feature_extraction import cfq, getFaceRegionsofInterest, faceDetector

'''rFMD dataset'''

totalgroundtruth = 0
totalcorrectdetectedfaces = 0


def addGroundTruth(image, object_gt):
    for i in range(len(object_gt)):
        # Result is an array with all the bounding boxes detected
        try:
            bounding_box = object_gt[i]['bndbox']
            cv2.rectangle(image,
                          (int(bounding_box['xmin']), int(bounding_box['ymin'])),
                          (int(bounding_box['xmax']), int(bounding_box['ymax'])),
                          (255, 155, 0),
                          2)
            cv2.putText(image,
                        str(object_gt[i]['name']),
                        (int(bounding_box['xmin']), int(bounding_box['ymin']) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 155, 0), 1)
        except:
            bounding_box = object_gt['bndbox']
            cv2.rectangle(image,
                          (int(bounding_box['xmin']), int(bounding_box['ymin'])),
                          (int(bounding_box['xmax']), int(bounding_box['ymax'])),
                          (255, 155, 0),
                          2)
            cv2.putText(image,
                        str(object_gt['name']),
                        (int(bounding_box['xmin']), int(bounding_box['ymin']) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 155, 0), 1)
            break
    return image


def verify_intersection(a, b):  # returns False if rectangles don't intersect
    dx = min(a[1][0], b[1][0]) - max(a[0][0], b[0][0])  # min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a[1][1], b[1][1]) - max(a[0][1], b[0][1])  # min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx >= 0) and (dy >= 0):
        return True
    else:
        return False


def overlap_rectangles(bndbx1, bndbx2):
    # Determines the bounding box of the overlapped area between two rectangles
    if verify_intersection(bndbx1, bndbx2):
        xmin_ol = max(bndbx1[0][0], bndbx2[0][0])  # max(xmin1,xmin2)
        ymin_ol = max(bndbx1[0][1], bndbx2[0][1])  # max(ymin1,ymin2)
        xmax_ol = min(bndbx1[1][0], bndbx2[1][0])  # max(xmax1,xmax2)
        ymax_ol = min(bndbx1[1][1], bndbx2[1][1])  # max(ymax1,ymax2)
        bndbx_ol = [(xmin_ol, ymin_ol), (xmax_ol, ymax_ol)]
        return bndbx_ol
    else:
        return None


def verify_center(point, bndbx):
    # Verifies if the point is inside the bounding box
    if point[0] > bndbx[0][0] and point[0] < bndbx[1][0] and point[1] > bndbx[0][1] and point[1] < bndbx[1][1]:
        # If  bndbx.xmin < point.x < bndbx.xmax  and   bndbx.ymin < point.y < bndbx.ymax
        return True
    else:
        return False


def validate_face_detection_and_feature_extraction(res, gt_obj, image):
    global totalcorrectdetectedfaces, totalgroundtruth
    sample_info = list()
    face_info = None
    if len(gt_obj) > 5:
        totalgroundtruth += 1
    else:
        totalgroundtruth += len(gt_obj)
    for i in range(len(gt_obj)):
        # Verifies if ground truth contains only one element
        [_, ROI_FH, ROI_BE, ROI_NM] = getFaceRegionsofInterest(image, res)
        try:
            bndbox_gt = gt_obj[i]['bndbox']
            name = gt_obj[i]['name']
            gt_center = [(int(bndbox_gt['xmin']) + int(bndbox_gt['xmax'])) / 2,
                         (int(bndbox_gt['ymin']) + int(bndbox_gt['ymax'])) / 2]
            for j in range(len(res)):
                bndbox_fd = res[j]['box']
                fd_center = [bndbox_fd[0] + bndbox_fd[2] / 2,
                             bndbox_fd[1] + bndbox_fd[3] / 2]
                bb_overlap = overlap_rectangles([(int(bndbox_gt['xmin']), int(bndbox_gt['ymin'])),
                                                 (int(bndbox_gt['xmax']), int(bndbox_gt['ymax']))],
                                                [(bndbox_fd[0], bndbox_fd[1]),
                                                 (bndbox_fd[0] + bndbox_fd[2], bndbox_fd[1] + bndbox_fd[3])])
                if bb_overlap is None:
                    # The j-th detected face doesn't correspond to current i-th ground truth
                    if face_info is None:
                        face_info = {
                            'gt_index': i,
                            'name': name,
                            'bndbx_gt': [(int(bndbox_gt['xmin']), int(bndbox_gt['ymin'])),
                                         (int(bndbox_gt['xmax']), int(bndbox_gt['ymax']))],
                            'fd_index': None,
                            'bndbx_fd': None,
                            'cqf': None,
                            'status': 'Error: no match found'
                        }
                elif verify_center(gt_center, bb_overlap) and verify_center(fd_center, bb_overlap):
                    # The j-th detected face probably corresponds to current i-th ground truth
                    face_info = {
                        'gt_index': i,
                        'name': name,
                        'bndbx_gt': [(int(bndbox_gt['xmin']), int(bndbox_gt['ymin'])),
                                     (int(bndbox_gt['xmax']), int(bndbox_gt['ymax']))],
                        'fd_index': j,
                        'bndbx_fd': [(bndbox_fd[0], bndbox_fd[1]),
                                     (bndbox_fd[0] + bndbox_fd[2], bndbox_fd[1] + bndbox_fd[3])],
                        'cqf': (cfq(ROI_FH[j], ROI_BE[j], ROI_NM[j])),
                        'status': 'OK'
                    }
                    totalcorrectdetectedfaces += 1
                    break
                else:
                    face_info = {
                        'gt_index': i,
                        'name': name,
                        'bndbx_gt': [(int(bndbox_gt['xmin']), int(bndbox_gt['ymin'])),
                                     (int(bndbox_gt['xmax']), int(bndbox_gt['ymax']))],
                        'fd_index': None,
                        'bndbx_fd': None,
                        'cqf': None,
                        'status': 'Warning: at least one centroid is out of the overlapped area'
                    }
            sample_info.append(face_info)
            face_info = None
        except:
            bndbox_gt = gt_obj['bndbox']
            name = gt_obj['name']
            gt_center = [(int(bndbox_gt['xmin']) + int(bndbox_gt['xmax'])) / 2,
                         (int(bndbox_gt['ymin']) + int(bndbox_gt['ymax'])) / 2]
            for j in range(len(res)):
                bndbox_fd = res[j]['box']
                fd_center = [bndbox_fd[0] + bndbox_fd[2] / 2,
                             bndbox_fd[1] + bndbox_fd[3] / 2]
                bb_overlap = overlap_rectangles([(int(bndbox_gt['xmin']), int(bndbox_gt['ymin'])),
                                                 (int(bndbox_gt['xmax']), int(bndbox_gt['ymax']))],
                                                [(bndbox_fd[0], bndbox_fd[1]),
                                                 (bndbox_fd[0] + bndbox_fd[2], bndbox_fd[1] + bndbox_fd[3])])
                if bb_overlap is None:
                    # The j-th detected face doesn't correspond to current i-th ground truth
                    if face_info is None:
                        face_info = {
                            'gt_index': i,
                            'name': name,
                            'bndbx_gt': [(int(bndbox_gt['xmin']), int(bndbox_gt['ymin'])),
                                         (int(bndbox_gt['xmax']), int(bndbox_gt['ymax']))],
                            'fd_index': None,
                            'bndbx_fd': None,
                            'cqf': None,
                            'status': 'Error: no match found'
                        }
                elif verify_center(gt_center, bb_overlap) and verify_center(fd_center, bb_overlap):
                    # The j-th detected face probably corresponds to current i-th ground truth
                    face_info = {
                        'gt_index': i,
                        'name': name,
                        'bndbx_gt': [(int(bndbox_gt['xmin']), int(bndbox_gt['ymin'])),
                                     (int(bndbox_gt['xmax']), int(bndbox_gt['ymax']))],
                        'fd_index': j,
                        'bndbx_fd': [(bndbox_fd[0], bndbox_fd[1]),
                                     (bndbox_fd[0] + bndbox_fd[2], bndbox_fd[1] + bndbox_fd[3])],
                        'cqf': (cfq(ROI_FH[j], ROI_BE[j], ROI_NM[j])),
                        'status': 'OK'
                    }
                    totalcorrectdetectedfaces += 1
                    break
                else:
                    face_info = {
                        'gt_index': i,
                        'name': name,
                        'bndbx_gt': [(int(bndbox_gt['xmin']), int(bndbox_gt['ymin'])),
                                     (int(bndbox_gt['xmax']), int(bndbox_gt['ymax']))],
                        'fd_index': None,
                        'bndbx_fd': None,
                        'cqf': None,
                        'status': 'Warning: at least one centroid is out of the overlapped area'
                    }
            sample_info.append(face_info)

            break
        # Writes an OK for each element in res that coincides with one of gt_obj
        pass
    return sample_info


def FMD_feature_extraction(detector, dir_rFMD):
    """
    """
    j = 0
    totaldetectedfaces = 0
    j_rFMD = 0
    dirImage = f'{dir_rFMD}/rFMD_{j}.png'
    dirGroundTruth = f'{dir_rFMD}/rFMD_{j}.xml'
    fileExists = path.exists(dirImage)
    gtExists = path.exists(dirGroundTruth)

    info_samples = list()
    while fileExists and gtExists and j < 500:
        # Checks that there are no more than 5 faces (from ground truth) in the image
        f = open(dirGroundTruth, 'r')  # Opens ground truth file
        obj = xmltodict.parse(f.read())  # Reads content of ground truth file
        obj_gt = obj['annotation']['object']  # Object with ground truth of sample

        im = cv2.cvtColor(cv2.imread(dirImage), cv2.COLOR_BGR2RGB)  # Opens image file
        [im2, res] = faceDetector(im, detector)  # Face detection stage
        totaldetectedfaces += len(res)
        image_data = {'filename': f'rFMD_{j}',
                      'data': validate_face_detection_and_feature_extraction(res, obj_gt, im2)
                      }
        info_samples.append(image_data)

        print('Sample {} analyzed and registered'.format(j))

        # Verifies if there's another image available
        j = j + 1

        dirImage = f'{dir_rFMD}/rFMD_{j}.png'
        dirGroundTruth = f'{dir_rFMD}/rFMD_{j}.xml'
        fileExists = path.exists(dirImage)
        gtExists = path.exists(dirGroundTruth)

    print(f"Total detected faces: {totaldetectedfaces}")
    json.dump(info_samples, open("./info_samples_FMD.json", "w"))


def FMD_get_features_array(info_samples_file='./info_samples_FMD.json'):
    f = open(info_samples_file, 'r')
    jsonFMDstr = f.read()
    jsonFMDlist = json.loads(jsonFMDstr)
    fd_errors_FMD = 0  # Face detection stage error counter
    fd_warning_FMD = 0  # Face detection stage warning counter
    fd_ok_FMD = 0  # Face detection stage OK counter

    maskclassFMD = list()
    nomaskclassFMD = list()
    maskclass = list()
    nomaskclass = list()

    # Analyze every element of each image
    for i in range(len(jsonFMDlist)):
        sampleElements = jsonFMDlist[i]['data']
        for j in range(len(sampleElements)):
            currentElement = sampleElements[j]
            if currentElement is None:
                # Error: no face detected in this image
                fd_errors_FMD = fd_errors_FMD + 1
            elif 'Warning' in currentElement['status']:
                fd_warning_FMD = fd_warning_FMD + 1
            elif 'OK' in currentElement['status']:
                fd_ok_FMD = fd_ok_FMD + 1
                if 'with_mask' in currentElement['name']:
                    maskclassFMD.append(currentElement['cqf'])
                    maskclass.append(currentElement['cqf'])
                elif 'without_mask' in currentElement['name']:
                    nomaskclassFMD.append(currentElement['cqf'])
                    nomaskclass.append(currentElement['cqf'])
            else:
                fd_errors_FMD = fd_errors_FMD + 1

    maskFMD = np.array(maskclassFMD)
    nomaskFMD = np.array(nomaskclassFMD)

    featuresFMD = np.concatenate((maskFMD, nomaskFMD), axis=0)
    targetsFMD = np.concatenate((np.zeros(len(maskclassFMD)) + 1, np.zeros(len(nomaskclassFMD)) - 1))

    return featuresFMD, targetsFMD
