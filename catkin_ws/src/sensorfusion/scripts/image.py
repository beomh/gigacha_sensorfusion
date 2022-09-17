#!/usr/bin/env python
#-*- coding: utf-8 -*-

# Python 2/3 compatibility
from __future__ import print_function
import cv2

img = cv2.imread('image.jpg', cv2.IMREAD_COLOR)
print(img.shape)
cv2.imshow('train', img)
cv2.waitKey(0)