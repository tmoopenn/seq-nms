# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Sergey Karayev
# --------------------------------------------------------

cimport cython
import numpy as np
cimport numpy as np


def compute_area(np.ndarray[double, ndim=2] boxes):
    '''
    Args
        boxes: (N,4) ndarray of float
    '''
    cdef unsigned int N = boxes.shape[0]
    cdef np.ndarray[double, ndim=1] areas = np.zeros((N), dtype=np.float64)
    
    for i in range(N):
        areas[i] = (boxes[i, 2] - boxes[i,0]) * (boxes[i, 3] - boxes[i, 1])
    return areas 

def compute_overlap(
    np.ndarray[double, ndim=2] boxes,
    np.ndarray[double, ndim=2] query_boxes
):
    """
    Args
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float

    Returns
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[double, ndim=2] overlaps = np.zeros((N, K), dtype=np.float64)
    cdef double iw, ih, box_area
    cdef double ua
    cdef unsigned int k, n
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0]) *
            (query_boxes[k, 3] - query_boxes[k, 1])
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) 
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) 
                )
                if ih > 0:
                    ua = np.float64(
                        (boxes[n, 2] - boxes[n, 0]) *
                        (boxes[n, 3] - boxes[n, 1]) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps

def compute_overlap_areas_given(
    np.ndarray[double, ndim=2] boxes,
    np.ndarray[double, ndim=2] query_boxes,
    np.ndarray[double, ndim=1] query_areas
):
    """
    Args
        a: (N, 4) ndarray of float
        b: (K, 4) ndarray of float
        c: (K) ndarray of float

    Returns
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[double, ndim=2] overlaps = np.zeros((N, K), dtype=np.float64)
    cdef double iw, ih, box_area
    cdef double ua
    cdef unsigned int k, n
    for k in range(K):
        box_area = query_areas[k]
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) 
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) 
                )
                if ih > 0:
                    ua = np.float64(
                        (boxes[n, 2] - boxes[n, 0]) *
                        (boxes[n, 3] - boxes[n, 1]) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps
