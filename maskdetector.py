import cv2
from mtcnn import MTCNN
from feature_extraction import faceDetector, getFaceRegionsofInterest, cfq
from joblib import load
import sys

img_name = '11.png'

if __name__ == '__main__':
    # Gets image filename from console if given
    try:
        if sys.argv[1] is not None:
            img_name = sys.argv[1]
    except:
        # Parameters were not given, using default
        pass
    imageFilename = f'./data/{img_name}'
    # Loads face detection model
    detector = MTCNN()

    # Loads classificator model for face mask detection
    clf = load('./svm_facemaskdetection.joblib')

    # Opens input image file
    im = cv2.cvtColor(cv2.imread(imageFilename), cv2.COLOR_BGR2RGB)  # Opens image file

    # Feature extraction
    [im, res] = faceDetector(im, detector)  # Face detection stage

    CFQ = list()
    for i in range(len(res)):  # Analysis of each detected face
        [im, ROI_FH, ROI_BE, ROI_NM] = getFaceRegionsofInterest(im, res)  # Extracts face regions of interest
        CFQ.append((cfq(ROI_FH[i], ROI_BE[i], ROI_NM[i])))

    print('Detected faces: {}'.format(len(res)))
    print('Feature vector(s): {}'.format(CFQ))
    # Classification
    y_pred = clf.predict(CFQ)
    y = list()

    # Generates an image with the results
    for i in range(len(res)):
        bounding_box = res[i]['box']
        if y_pred[i] == 1:
            y.append('Mask')
            color = (0, 255, 0)
        else:
            y.append('No mask')
            color = (255, 0, 0)
        cv2.rectangle(im,
                      (bounding_box[0], bounding_box[1]),
                      (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                      color,
                      2)
        cv2.putText(im,
                    '{}'.format(y[i]),
                    (bounding_box[0], bounding_box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Show results
    print('Results: {}'.format(y))
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)  # Returns to BGR to be shown using cv2 library
    cv2.imwrite(f'./data/result_{img_name}', im)
    cv2.imshow('Results', im)
    cv2.waitKey()
