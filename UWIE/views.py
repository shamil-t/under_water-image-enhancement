from .MIP.sceneRadiance import sceneRadianceRGBMIP
from .MIP.TM import getTransmission
from .MIP.getRefinedTramsmission import Refinedtransmission
from .MIP.EstimateDepth import DepthMap
from .MIP.BL import getAtomsphericLight

from .DCP.GuidedFilter import GuidedFilter
from .DCP.main import getMinChannel
from .DCP.main import getDarkChannel
from .DCP.main import getAtomsphericLight

from .RAY.sceneRadiance import sceneRadianceRGB
from .RAY.rayleighDistribution import rayleighStretching
from .RAY.hsvStretching import HSVStretching
from .RAY.global_stretching_RGB import stretching
from .RAY.color_equalisation import RGB_equalisation

from .CLAHE.sceneRadianceCLAHE import RecoverCLAHE

from matplotlib import pyplot as plt
from django.shortcuts import render
from .models import InputCLAHE, InputRAY, InputMIP, InputDCP, InputClassify

import cv2
import shutil
import os
import numpy as np
import os

from keras.models import load_model
from keras.preprocessing import image as image_utils

from PIL import Image

import matplotlib
matplotlib.use('Agg')
plt.ioff()


def index(request):
    return render(request, 'index.html')


def clahe(request):
    return render(request, 'clahe.html', {'img1': "static/ip_img.jpg", 'v': "hidden", 'in': "visible"})


def rayleigh(request):
    return render(request, 'rayleigh.html', {'img1': "static/ip_img.jpg", 'v': "hidden", 'in': "visible"})


def mip(request):
    return render(request, 'mip.html', {'img1': "static/ip_img.jpg", 'v': "hidden", 'in': "visible"})


def dcp(request):
    return render(request, 'dcp.html', {'img1': "static/ip_img.jpg", 'v': "hidden", 'in': "visible"})


def classify(request):
    return render(request, 'classify.html', {'img1': "static/ip_img.jpg"})

def paper(request):
    return render(request, 'paper.html')

def algorithm(request):
    return render(request, 'algorithms.html')

def about(request):
    return render(request, 'about.html')


def get_image(request):
    if not os.path.exists("UWIE/static/Input/CLAHE/"):
        os.makedirs("UWIE/static/Input/CLAHE/")
    shutil.rmtree("UWIE/static/Input/CLAHE/")
    if request.method == "POST":
        in_img = request.FILES['image']
        in_img.name = "input.jpg"
        input = InputCLAHE(img=in_img)
        input.save()
        enhanceImageCLAHE("UWIE/static")
        img1 = "static/Input/CLAHE/input.jpg"
        img2 = "CLAHE.jpg"
        hist_in = "hist_in.jpg"
        hist_out = "hist_op.jpg"

    return render(request, 'clahe.html', {'img1': img1, 'img2': img2, 'hist_in': hist_in,
                                          'hist_out': hist_out, 'v': 'block', 'in': "none"})

def enhanceImageCLAHE(folder):
    np.seterr(over='ignore')
    img = cv2.imread(folder + '/Input/CLAHE/input.jpg')
    sceneRadiance = RecoverCLAHE(img)

    if not os.path.exists(folder + '/Output/CLAHE/'):
        os.makedirs(folder + '/Output/CLAHE/')
    else:
        cv2.imwrite(folder + '/Output/CLAHE/' + 'CLAHE.jpg', sceneRadiance)

        # HISTOGRAM
        input_img = cv2.imread(folder + '/Input/CLAHE/input.jpg', 0)

        output_img = cv2.imread(folder + '/Output/CLAHE/CLAHE.jpg', 0)

        hist_ip = cv2.calcHist([input_img], [0], None, [256], [0, 256])
        inp = plt.figure()
        plt.plot(hist_ip)
        inp.savefig(folder+'/Output/CLAHE/hist_in.jpg')
        plt.close(inp)
        hist_op = cv2.calcHist([output_img], [0], None, [256], [0, 256])
        op = plt.figure()
        plt.plot(hist_op)
        op.savefig(folder+'/Output/CLAHE/hist_op.jpg')
        plt.close(op)

        # plt.hist(input_img.ravel(),256,[0,256]);
        # plt.hist(output_img.ravel(),256,[0,256]);

def get_image_ray(request):
    print('get_imageray')

    if not os.path.exists("UWIE/static/Input/RAY/"):
        os.makedirs("UWIE/static/Input/RAY/")
    shutil.rmtree("UWIE/static/Input/RAY/")

    if request.method == "POST":
        in_img = request.FILES['image']
        in_img.name = "input.jpg"
        input = InputRAY(img=in_img)
        input.save()
        enhanceImageRAY("UWIE/static")
        img1 = "static/Input/RAY/input.jpg"
        img2 = "RAY.jpg"
        RGB_equ = "RAY_RGB.jpg"
        stretch = "RAY_stretch.jpg"
        R_stretch = "RAY_Rstretch.jpg"
        HSV_str = "RAY_HSV.jpg"
    return render(request, 'rayleigh.html', {'img1': img1, 'img2': img2, 'RGB': RGB_equ,
                                             'STR': stretch, 'Rstr': R_stretch, 'HSV': HSV_str, 'in': "none"})

def enhanceImageRAY(folder):
    img = cv2.imread(folder + '/Input/RAY/input.jpg')
    height = len(img)
    width = len(img[0])
    sceneRadiance = RGB_equalisation(img, height, width)
    if not os.path.exists(folder+"/Output/RAY/"):
        os.makedirs(folder+"/Output/RAY/")
    else:
        cv2.imwrite(folder + '/Output/RAY/' + 'RAY_RGB.jpg', sceneRadiance)
    sceneRadiance = stretching(sceneRadiance)
    cv2.imwrite(folder + '/Output/RAY/' +
                'RAY_stretch.jpg', sceneRadiance)

    sceneRadiance_Lower, sceneRadiance_Upper = rayleighStretching(
        sceneRadiance, height, width)

    sceneRadiance = (np.float64(sceneRadiance_Lower) +
                     np.float64(sceneRadiance_Upper)) / 2

    cv2.imwrite(folder + '/Output/RAY/' +
                'RAY_Rstretch.jpg', sceneRadiance)

    sceneRadiance = HSVStretching(sceneRadiance)
    cv2.imwrite(folder + '/Output/RAY/' + 'RAY_HSV.jpg', sceneRadiance)

    sceneRadiance = sceneRadianceRGB(sceneRadiance)
    cv2.imwrite(folder + '/Output/RAY/' + 'RAY.jpg', sceneRadiance)

def get_image_dcp(request):
    print('get_image_dcp')

    if not os.path.exists("UWIE/static/Input/DCP/"):
        os.makedirs("UWIE/static/Input/DCP/")

    shutil.rmtree("UWIE/static/Input/DCP/")

    if request.method == "POST":
        in_img = request.FILES['image']
        in_img.name = "input.jpg"
        input = InputDCP(img=in_img)
        input.save()
        restoreDCP("UWIE/static")
        img1 = "static/Input/DCP/input.jpg"
        img2 = "DCP.jpg"
    return render(request, 'dcp.html', {'img1': img1, 'img2': img2, 'G': "Gray.jpg",
                                        'D': "Dark.jpg", 'T': "DCP_TM.jpg", 'TRA': "TRA.jpg", 'in': 'none'})

def restoreDCP(folder):

    def getRecoverScene(img, omega=0.95, t0=0.1, blockSize=15, meanMode=False, percent=0.001):
        gimfiltR = 50
        eps = 10 ** -3

        imgGray = getMinChannel(img)

        cv2.imwrite(folder + '/Output/DCP/' + 'Gray.jpg', imgGray)

        imgDark = getDarkChannel(imgGray, blockSize=blockSize)
        cv2.imwrite(folder + '/Output/DCP/' + 'Dark.jpg', imgDark)
        atomsphericLight = getAtomsphericLight(
            imgDark, img, meanMode=meanMode, percent=percent)
        print("Atmospheric light", atomsphericLight)
        # cv2.imwrite(folder + '/Output/DCP/' + 'Light.jpg', atomsphericLight)

        imgDark = np.float64(imgDark)
        transmission = 1 - omega * imgDark / atomsphericLight

        guided_filter = GuidedFilter(img, gimfiltR, eps)

        print("Guided filter", guided_filter)
        # cv2.imwrite(folder + '/Output/DCP/' + 'Filter.jpg', guided_filter)

        transmission = guided_filter.filter(transmission)
        cv2.imwrite(folder+"/Output/DCP/"+"TRA.jpg",
                    np.uint8(transmission * 255))

        transmission = np.clip(transmission, t0, 0.9)

        sceneRadiance = np.zeros(img.shape)

        for i in range(0, 3):
            img = np.float64(img)
            sceneRadiance[:, :, i] = (
                img[:, :, i] - atomsphericLight) / transmission + atomsphericLight

        sceneRadiance = np.clip(sceneRadiance, 0, 255)
        sceneRadiance = np.uint8(sceneRadiance)

        return transmission, sceneRadiance

    if not os.path.exists(folder+"/Output/DCP/"):
        os.makedirs(folder+"/Output/DCP/")

    img = cv2.imread(folder + '/Input/DCP/input.jpg')

    transmission, sceneRadiance = getRecoverScene(img)
    cv2.imwrite(folder + '/Output/DCP/' + 'DCP_TM.jpg',
                np.uint8(transmission * 255))
    cv2.imwrite(folder + '/Output/DCP/' + 'DCP.jpg', sceneRadiance)

def get_image_mip(request):
    if not os.path.exists("UWIE/static/Input/MIP/"):
        os.makedirs("UWIE/static/Input/MIP/")

    shutil.rmtree("UWIE/static/Input/MIP/")

    if request.method == "POST":
        in_img = request.FILES['image']
        in_img.name = "input.jpg"
        input = InputMIP(img=in_img)
        input.save()
        restoreMIP("UWIE/static")
        img1 = "static/Input/MIP/input.jpg"
        img2 = "MIP.jpg"
    return render(request, 'mip.html', {'img1': img1, 'img2': img2, 'D': "MIP_diff.jpg",
                                        'TR': "MIP_tr.jpg", 'RT': "MIP_rtra.jpg", 'TM': "MIP_TM.jpg", 'in': 'none'})

def restoreMIP(folder):
    img = cv2.imread(folder + '/Input/MIP/input.jpg')

    if not os.path.exists(folder+"/Output/MIP/"):
        os.makedirs(folder+"/Output/MIP/")

    blockSize = 9

    Diff = None
    Tr = None
    Rtr = None

    largestDiff = DepthMap(img, blockSize)
    Diff = largestDiff

    transmission = getTransmission(largestDiff)
    Tr = transmission

    transmission = Refinedtransmission(transmission, img)
    Rtr = transmission

    AtomsphericLight = getAtomsphericLight(transmission, img)

    sceneRadiance = sceneRadianceRGBMIP(
        img, transmission, AtomsphericLight)

    cv2.imwrite(folder + '/Output/MIP/' + 'MIP_TM.jpg',
                np.uint8(transmission * 255))
    cv2.imwrite(folder + '/Output/MIP/' + 'MIP_diff.jpg', np.uint8(Diff * 255))
    cv2.imwrite(folder + '/Output/MIP/' + 'MIP_tr.jpg', np.uint8(Tr * 255))
    cv2.imwrite(folder + '/Output/MIP/' + 'MIP_rtra.jpg', np.uint8(Rtr * 255))
    cv2.imwrite(folder + '/Output/MIP/' + 'MIP.jpg', sceneRadiance)

def classifyimage(request):
    folder = "UWIE/static/Input/CLASSIFY/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    shutil.rmtree(folder)

    ans = None

    if request.method == "POST":
        in_img = request.FILES['image']
        in_img.name = "input.jpg"
        input = InputClassify(img=in_img)
        input.save()

        img_width, img_height = 128, 128
        model = 'UWIE/CLASSIFY/models'
        model_path = model+'/model_25.h5'
        model_weights_path = model+'/weights_25.h5'
        model = load_model(model_path, compile=False)
        model.load_weights(model_weights_path)
        # BytesIO(response.content))
        test_image = Image.open(folder+"input.jpg")
        put_image = test_image.resize((400, 400))
        test_image = test_image.resize((128, 128))
        test_image = image_utils.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        result = model.predict(test_image)
        print(result)

        rr = result[0]
        print(np.argmax(rr))
        if np.argmax(rr) == 0:
            ans = 'A'
        elif np.argmax(rr) == 1:
            ans = 'P'
        elif np.argmax(rr) == 2:
            ans = "T"
        print("prediction", ans)
    img1 = "static/Input/CLASSIFY/input.jpg"
    return render(request, 'classify.html', {'img1':img1,'r': ans})
