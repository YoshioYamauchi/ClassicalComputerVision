import os
import numpy as np
from mss import mss
from PIL import Image
import time
import cv2
import utils_cy
import matplotlib.pyplot as plt
[plt.close() for x in range(4)]


def estimate_shift(img1, img2):
    img_size = img1.shape[0]
    ham2d = np.hamming(img_size)[:,None]
    f1 = cv2.dft(img1 * ham2d, flags=cv2.DFT_COMPLEX_OUTPUT)
    f2 = cv2.dft(img2 * ham2d, flags=cv2.DFT_COMPLEX_OUTPUT)
    f1 = f1[:,:,0] + 1j*f1[:,:,1]
    f2 = f2[:,:,0] + 1j*f2[:,:,1]
    P = f1 * np.conj(f2) / np.abs(f1 * np.conj(f2))
    Pinv = np.real(np.fft.ifft2(P))
    maxidx = np.argmax(Pinv.flatten())
    # dx_pred = int(Pinv.shape[0] - maxidx / Pinv.shape[0])
    # dy_pred = int(Pinv.shape[0] - maxidx % Pinv.shape[0])
    dx_pred = int(maxidx % img_size)
    dy_pred = int(maxidx / img_size)
    if dx_pred < 0.5 * img_size:
        dx_pred = dx_pred * (-1)
    else:
        dx_pred = img_size - dx_pred
    if dy_pred < 0.5 * img_size:
        dy_pred = dy_pred * (-1)
    else:
        dy_pred = img_size - dy_pred
    dx_r = float(dx_pred) / img_size
    dy_r = float(dy_pred) / img_size
    return dx_r, dy_r




def simple_edge_detector(img):
    img = img * 1.
    # d = int(img.shape[0]*0.02)
    d = 3
    th = 100
    crop1 = img[d:,d:]
    crop2 = img[:-d, d:]
    crop3 = img[:-d, :-d]
    crop4 = img[d:, :-d]
    # f = np.abs(crop1 - crop3) > th
    f1 = np.abs(crop1 - crop2) > th
    f1 = f1 + (np.abs(crop1 - crop3) > th)
    f1 = f1 + (np.abs(crop1 - crop4) > th)
    # f1 = f1 + (np.abs(crop2 - crop3) > th)
    # f1 = f1 + (np.abs(crop2 - crop4) > th)
    # f1 = f1 + (np.abs(crop3 - crop4) > th)
    return f1 * np.ones((img.shape[0]-d, img.shape[1]-d))
    # return np.abs(crop1 - crop2).astype(np.uint8)

def stretch_contrast(img):
    lb = 4.
    ub = 100.
    img = (255.-1)/(ub-lb)*(img*1. - lb)
    img[img < 0] = 255
    img[img > 255] = 255
    img = img.astype(np.uint8)
    return img

def canny(img):
    edges = cv2.Canny(img, 200, 220)
    return edges

l = 600
mon = {'top':400, 'left':2900,'width':l,'height':l}
sct = mss()
edegs = None
d = 1
frame_before = np.zeros((l,l), dtype=np.uint8)[::d,::d]
frame_current = np.zeros((l,l), dtype=np.uint8)[::d,::d]
frequency = 100 * 100
dxr_avg = 0
dyr_avg = 0
fps_avg = 100
while True:
    t00 = time.time()
    sct.get_pixels(mon)
    frame_3c = Image.frombytes(mode='RGB', size=(sct.width, sct.height), data=sct.image)
    # frame = Image.frombytes(mode='L', size=(sct.width, sct.height), data=sct.image)
    frame_3c = np.array(frame_3c)
    frame = frame_3c[:,:,2]
    frame = np.array(frame, order='c', dtype=np.uint8) # no overhead added
    frame_before = frame_current
    frame_current = frame
    dxr, dyr = estimate_shift(frame_before[::4,::4], frame_current[::4,::4])
    dxr_avg =  dxr * 0.1 + 0.9 * dxr_avg
    dyr_avg =  dyr * 0.1 + 0.9 * dyr_avg
    bdx = max(min(int(dxr_avg*600), 19), -19)
    bdy = max(min(int(dyr_avg*600), 19), -19)
    # print(bdx, bdy)
    frame_before_crop = frame_before[20:-20, 20:-20]
    frame_current_crop = frame_current[20-bdy:-(20+bdy), 20-bdx:-(20+bdx)]
    frame_diff = frame_current_crop - frame_before_crop

    t10 = time.time()
    t11 = time.time()
    t20 = time.time()
    frame_3c = cv2.putText(frame_3c, 'dx = {}'.format(int(dxr_avg*100)), (10, 20),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
    frame_3c = cv2.putText(frame_3c, 'dy = {}'.format(int(dyr_avg*100)), (150, 20),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
    dx_display = int(dxr_avg * 4000 * fps_avg / 100.)
    dy_display = int(dyr_avg * 4000 * fps_avg / 100.)
    frame_3c = cv2.line(frame_3c, (300, 300), (300+dx_display, 300+dy_display), (0,255,0), thickness=7)

    cv2.imshow('frame_3c', frame_3c)
    # cv2.imshow('frame_diff', frame_diff)
    if cv2.waitKey(1) & 0xff == ord('q') :
        cv2.destroyAllWindows()
    t21 = time.time()
    t01 = time.time()
    time.sleep(max(1./frequency - (t01 - t00), 0))
    t02 = time.time()
    fps = int(1./(t02 - t00))
    fps_avg = 0.01 * fps + 0.99 * fps_avg
    t1 = 1000*(t11 - t10)
    t2 = 1000*(t21 - t20)
    txt = '{:>3} fps, t1={:5.2f} ms, t2={:5.2f} ms {}, '.format(fps, t1, t2, utils_cy.stretch_contrast_cy.__doc__)
    txt += 'dxr= {:2}, dyr= {:2}'.format(int(dxr*100), int(dyr*100))
    print(txt)
