"""
Script to extract features and train a linear classifier using SVM
"""
import os.path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm, metrics
from joblib import dump
import numpy as np

from FMD import FMD_feature_extraction, FMD_get_features_array
from MFN import MFN_feature_extraction, MFN_get_features_array
from CelebA import CelebA_feature_extraction, CelebA_get_features_array
from mtcnn import MTCNN

DIR_rFMD = 'C:/wsPDI/DSPDI/rFMD/'
DIR_rMFN = 'C:/wsPDI/DSPDI/rMFN/'
DIR_rCelebA = 'C:/wsPDI/DSPDI/rCelebA/'

if __name__ == '__main__':
    """
    Process:
    1) Feature extraction
        1.1) Reduced FMD dataset
        1.2) Reduced MaskedFaceNet dataset
        1.3) Reduced CelebA dataset
    2) SVM training
    3) Evaluation
    """
    # 1) Feature extraction
    facedetector = MTCNN()

    # 1.1) rFMD dataset
    if not os.path.exists('./info_samples_FMD.json'):
        FMD_feature_extraction(facedetector, DIR_rFMD)
    featuresFMD, targetsFMD = FMD_get_features_array('./info_samples_FMD.json')
    # 1.2) rMFN dataset
    if not os.path.exists('./info_samples_MFN.json'):
        MFN_feature_extraction(facedetector, DIR_rMFN)
    featuresMFN, targetsMFN = MFN_get_features_array('./info_samples_MFN.json')
    # 1.3) rCelebA dataset
    if not os.path.exists('./info_samples_CelebA.json'):
        CelebA_feature_extraction(facedetector, DIR_rCelebA)
    featuresCelebA, targetsCelebA = CelebA_get_features_array('./info_samples_CelebA.json')

    features = np.concatenate((featuresFMD, featuresMFN, featuresCelebA), axis=0)
    targets = np.concatenate((targetsFMD, targetsMFN, targetsCelebA))

    # 2) SVM training
    # Split data into train and test sets with an 80%/20% ratio
    xtrain, xtest, ytrain, ytest = train_test_split(features, targets, test_size=0.2)
    # Create a svm Classifier
    clf = svm.SVC(kernel='linear')  # Linear Kernel

    # Train the model using the training sets
    clf.fit(xtrain, ytrain)
    # Saves linear classifier parameters
    dump(clf, 'svm_facemaskdetection.joblib')

    # 3) Evaluation
    print('Test data scores:')
    y_pred = clf.predict(xtest)
    print("Accuracy:", metrics.accuracy_score(ytest, y_pred))
    print("Precision:", metrics.precision_score(ytest, y_pred))
    print("Recall:", metrics.recall_score(ytest, y_pred))
    print('Train data scores:')
    y_pred = clf.predict(xtrain)
    print("Accuracy:", metrics.accuracy_score(ytrain, y_pred))
    print("Precision:", metrics.precision_score(ytrain, y_pred))
    print("Recall:", metrics.recall_score(ytrain, y_pred))

