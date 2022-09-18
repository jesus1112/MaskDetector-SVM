import cv2
import numpy as np
from mtcnn import MTCNN
from os import path
import xmltodict, json


def faceDetector(img, det, draw_results=False):
    image = img[:, :, :]
    result = det.detect_faces(image)
    for i in range(len(result)):
        # Result is an array with all the bounding boxes detected
        bounding_box = result[i]['box']
        keypoints = result[i]['keypoints']
        if draw_results:
            cv2.rectangle(image,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0, 155, 255),
                          2)
            cv2.putText(image,
                        'FD {}'.format(i),
                        (bounding_box[0], bounding_box[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 155, 255), 1)
            cv2.circle(image, (keypoints['left_eye']), 2, (0, 155, 255), 2)
            cv2.circle(image, (keypoints['right_eye']), 2, (0, 155, 255), 2)
            cv2.circle(image, (keypoints['nose']), 2, (0, 155, 255), 2)
            cv2.circle(image, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
            cv2.circle(image, (keypoints['mouth_right']), 2, (0, 155, 255), 2)
    return image, result


def getFaceRegionsofInterest(image, res, draw_results=False):
    """adas"""
    im_aux = image
    froi_fh = list()
    froi_be = list()
    froi_nm = list()
    for i in range(len(res)):
        bounding_box = res[i]['box']
        xmin = bounding_box[0]
        ymin = bounding_box[1]
        xmax = bounding_box[0] + bounding_box[2]
        ymax = bounding_box[1] + bounding_box[3]
        keypoints = res[i]['keypoints']
        '''Forehead region'''
        fh_l = keypoints['left_eye'][0]  # horizontal coordinate for left eye as left limit of forehead region
        fh_r = keypoints['right_eye'][0]  # horizontal coordinate for right eye as right limit of forehead region
        fh_d = min(keypoints['left_eye'][1], keypoints['right_eye'][1])  # vertical coordinate of the eyes
        fh_u = int(0.15 * (fh_d - ymin)) + ymin  # top limit of forehead region
        fh_d = int(0.85 * (fh_d - ymin)) + ymin  # bottom limit of forehead region
        froi_fh.append(im_aux[fh_u:fh_d, fh_l:fh_r, :])
        if draw_results:
            cv2.rectangle(image,
                          (fh_l, fh_u),
                          (fh_r, fh_d),
                          (255, 0, 155),
                          2)
        '''Between eyes region'''
        eye_center = ((keypoints['left_eye'][0] + keypoints['right_eye'][0]) / 2,
                      (keypoints['left_eye'][1] + keypoints['right_eye'][1]) / 2);
        be_l = int(eye_center[0] - 0.25 * (
                keypoints['right_eye'][0] - keypoints['left_eye'][0]))  # left limit of between eyes region
        be_r = int(eye_center[0] + 0.25 * (
                keypoints['right_eye'][0] - keypoints['left_eye'][0]))  # right limit of between eyes region
        be_u = int(eye_center[1] - 0.1 * (ymax - ymin))  # top limit of between eyes region
        be_d = int(eye_center[1] + 0.1 * (ymax - ymin))  # bottom limit of between eyes region
        froi_be.append(im_aux[be_u:be_d, be_l:be_r, :])
        if draw_results:
            cv2.rectangle(image,
                          (be_l, be_u),
                          (be_r, be_d),
                          (255, 155, 0),
                          2)
        '''Nose-mouth region'''
        nm_l = keypoints['mouth_left'][0]  # left limit of between eyes region
        nm_r = keypoints['mouth_right'][0]  # right limit of between eyes region
        nm_u = keypoints['nose'][1]  # top limit of between eyes region
        nm_d = max(keypoints['mouth_left'][1], keypoints['mouth_right'][1])  # bottom limit of between eyes region
        froi_nm.append(im_aux[nm_u:nm_d, nm_l:nm_r, :])
        if draw_results:
            cv2.rectangle(image,
                          (nm_l, nm_u),
                          (nm_r, nm_d),
                          (0, 255, 155),
                          2)
    return image, froi_fh, froi_be, froi_nm


def cfq(roi_fh, roi_be, roi_nm):
    # Mean of the lower portion of the face
    m_roi_l_R = np.mean(roi_nm[:, :, 0])
    m_roi_l_G = np.mean(roi_nm[:, :, 1])
    m_roi_l_B = np.mean(roi_nm[:, :, 2])

    # Mean of an upper portion of the face
    var_fh = np.var(roi_fh[:, :, 0])
    var_be = np.var(roi_be[:, :, 0])
    if (var_fh < var_be):  # Selects the region with less variance on the R channel
        m_roi_u_R = np.mean(roi_fh[:, :, 0])
        m_roi_u_G = np.mean(roi_fh[:, :, 1])
        m_roi_u_B = np.mean(roi_fh[:, :, 2])
    else:
        m_roi_u_R = np.mean(roi_be[:, :, 0])
        m_roi_u_G = np.mean(roi_be[:, :, 1])
        m_roi_u_B = np.mean(roi_be[:, :, 2])

    # Quotient for RG channels
    rgu = m_roi_u_R / m_roi_u_G  # Upper region RG ratio
    rgl = m_roi_l_R / m_roi_l_G  # Lower region RG ratio
    rgQ = rgl / rgu  # RG channel quotient

    # Quotient for RB channels
    rbu = m_roi_u_R / m_roi_u_B  # Upper region RB ratio
    rbl = m_roi_l_R / m_roi_l_B  # Lower region RB ratio
    rbQ = rbl / rbu  # RB channel quotient

    # Quotient for RB channels
    gbu = m_roi_u_G / m_roi_u_B  # Upper region GB ratio
    gbl = m_roi_l_G / m_roi_l_B  # Lower region GB ratio
    gbQ = gbl / gbu  # RB channel quotient

    # return rgQ, rbQ, gbQ
    return rgQ, gbQ

