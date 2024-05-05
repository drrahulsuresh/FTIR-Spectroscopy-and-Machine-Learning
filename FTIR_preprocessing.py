
import scipy.io as sio
import numpy as np
import os
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt


def read_mat(filename):
    """
    Function to read matlab file for FTIR data.

    Args:
        filename(str): filename and directory location
    Return:
        wavenumber(array): Wavenumber array (ex. 800-1000).
        spcImage(array): Image data spcImage(h, w, wavenumber).
        labelKM_Bkg(array): Clustering result labelKM_Bkg(h, w).
    """
    data = sio.loadmat(filename)
    wavenumber = data['wavenumbers'].squeeze()
    spcImage = data['spcImage']
    labelKM_Bkg = data['labelKM_Bkg']

    return wavenumber, spcImage, labelKM_Bkg

    h, w, _ = spcImage.shape
    img_std = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            img_std[i, j] = np.std(spcImage[i, j, :])
    return img_std


def projection_area(spcImage, wavenumbers):
    """
    FTIR Image Reconstruction where Pixel Intensity = Area Under Curve of SPECTRUM.

    Args:
        spcImage(array): Spectral image data (h, w, wavenumber).
        wavenumbers(array): wavenumbers.
    Return:
        projection(array): h x w image projection.
    """
    h, w, _ = spcImage.shape
    wavenumbers = np.sort(wavenumbers)
    projection = np.zeros((h, w))
    for ii in range(h):
        for jj in range(w):
            projection[ii, jj] = np.trapz(spcImage[ii, jj, :], wavenumbers)
    return projection

def projection_std(spcImage):
    """
    Apply projection based on standard deviation of spcImage.

    Args:
        spcImage(np.ndarray): The datacube of FTIR image.

    Returns:
        img_std(np.ndarray): Image projection.
    """
    h, w, _ = spcImage.shape
    img_std = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            img_std[i, j] = np.std(spcImage[i, j, :])
    return img_std

def img_inverse(img, point=(0,0), background=0):
    """
    Inverse image using cv.bitwise_not by checking the point given whether it fit
    the background. If it's not, inverse will be run.

    Args:
        img: the K-means image projection (2D image).
        point: background coordinate.
        background: background value (i.e; 1 or 0).

    Returns:
        img: Inversed or not inversed image.
    """
    if img[point[0], point[1]] != background:
        img = cv.bitwise_not(img)
    return img

def img_thres_otsu(img, blur_kernel=(3,3), tval=0, maxval=255):
    """
    Applies Otsu thresholding to an image.

    Args:
        img (numpy.ndarray): Input image.
        blur_kernel (tuple): Gaussian blur kernel size.
        tval (int): Threshold value (not used in Otsu's method).
        maxval (int): Maximum value to use with the THRESH_BINARY_INV flag.

    Returns:
        thresh (numpy.ndarray): Thresholded image array.
    """
    if len(img.shape) > 2:  # Convert to grayscale if not already
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blurred = cv.GaussianBlur(img, blur_kernel, 0)
    _, thresh = cv.threshold(img_blurred, tval, maxval, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    return thresh


def img_rm_debris(img, X1=0.01):
    """
    Remove small particles from a 2D image.

    Args:
        img (numpy.ndarray): Input 2D image.
        X1 (float): Multiplier for the average area to determine the size threshold for removal.

    Returns:
        img (numpy.ndarray): Image with small particles removed.
    """
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < (X1 * np.mean([cv.contourArea(c) for c in contours])):
            cv.drawContours(img, [cnt], -1, (0, 0, 0), -1)
    return img

    # determine average area
    average_area = []
    cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x,y,w,h = cv.boundingRect(c)
        area = w * h
        average_area.append(area)
    average = sum(average_area) / len(average_area)

    # remove 'debris'
    cnts = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv.contourArea(c)
        if area < average * X1:
             cv.drawContours(thresh, [c], -1, (0,0,0), -1)
    return thresh


def img_rm_holes(img, X1=0.1, holes_kernel=(5,5), iterations=2):
    """
    Remove holes from an 2D image array.

    Args:
        img(np.ndarray): an array of 2D image.
        X1(float): multiplier of average area size.
        holes_kernel(tup): size of holes to be remove.
        interations(int): number of iterations .

    Returns:
        close(np.ndarray): image array.
    """
    thresh, X1, iterations = img, X1, iterations

    # checking img input
    if len(img.shape) != 2:
        print('Error: Image input invalid.')
        return None

    # determine average area
    average_area = []
    cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x,y,w,h = cv.boundingRect(c)
        area = w * h
        average_area.append(area)
    average = sum(average_area) / len(average_area)

    # remove 'holes'
    cnts = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv.contourArea(c)
        if area < average * X1:
            cv.drawContours(thresh, [c], -1, (0,0,0), -1)

    # Morph close and invert image
    kernel = cv.getStructuringElement(cv.MORPH_RECT, holes_kernel)
    close = cv.morphologyEx(
        thresh,cv.MORPH_CLOSE,
        kernel, iterations=iterations
         )

    return close

def calc_snr(dfX, signal_range, noise_range):
    """
    Calculate signal to noise (SNR) ratio based on mean signal / std noise.

    Args:
        dfX(pd.DataFrame): matrix of a df.
        signal_range(tuple): wavenumber range from df columns.
        noise_range(tuple): wavenumer ranger from df columns.
    Return:
        SNR ratio(np.array).
    """

    numeric_cols = pd.to_numeric(dfX.columns) # convert columns to numbers
    signal = dfX.loc[:, (numeric_cols >= signal_range[1]) & (numeric_cols <= signal_range[0])]
    signal = signal.mean(axis=1) # calculate signal
    noise = dfX.loc[:, (numeric_cols >= noise_range[1]) & (numeric_cols <= noise_range[0])]
    noise = noise.std(axis=1) # calculate noise
    snr = np.where(noise==0, 0, signal/noise) # calculate SNR ratio
    return snr

def calc_outliers_threshold(snr_column, n):
    """
    Calculate the outliers threshold of all spectra dataframe based on the formula
    = mean + n * SD, use this value to filter spectra outliers from the dataset.

    Args:
        snr_column(pd.core.series.Series): SNR column.
        n(int): SD multiplier.
    Return:
        outliers_threshold(float): the threshold value.
    """

    outliers_threshold = snr_column.mean() + (n * snr_column.std()) # mean+3xSD
    return outliers_threshold

