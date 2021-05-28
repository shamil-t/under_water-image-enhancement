# 🆄🅽🅳🅴🆁 🆆🅰🆃🅴🆁 🅸🅼🅰🅶🅴 🅴🅽🅰🅽🅲🅴🅼🅴🅽🆃 🅰🅽🅳🅳 🆁🅴🆂🆃🅾🆁🅰🆃🅸🅾🅽 : 🆁🅴🅴🅵 🅲🅻🅰🆂🆂🅸🅵🅸🅲🅰🆃🅸🅾🅽

The main objective of the Project is to reduce the noises in the Underwater Images.
We propose some methods for efficient removal of Noises using Image Processing
Techniques.
  The Underwater images have low quality which makes it a difficult process to analyze
the images. Here we propose Image Enhancement and Image Restoration process for
increasing the quality of Underwater Images. Clahe, Reyleigh distribution, DCP and
MIP,RGHS,ULAP methods are used in this project.

### IMAGE ENHANCEMENT

- CLAHE - CONTRAST LIMITED ADAPTIVE HISTOGRAM EQUALIZATION
- RAYLEIGH DISTRIBUTION
- RGHS - Relative Global Histogram Stretching

### IMAGE RESTORATION

- DCP - DARK CHANNEL PRIOR
- MIP - MAXIMUM INTENSITY PROJECTION
- ULAP - Underwater Light Attenuation Prior

## 🅿🆁🅴 🆁🅴🆀🆄🅴🆂🆃🅸🅴🆂

### Environment Setup

- python 3.8.6 64bit
- install dependencies `$ pip install -r requirements.txt`
- download models from [here]( "link title"), place it in models folder`/UWIE/CLASSIFY/models/`

### Dataset

- Pocillopora
- Acropora
- Turf

[Download DataSet from here](http://vision.ucsd.edu/~beijbom/moorea_labeled_corals/patches/)

## 🅷🅾🆆🆆 🆃🅾 🆁🆄🅽

`$ py manage.py runserver`

Demo project hosted on heroku [link](https://under-water-image-enhancement.herokuapp.com/)
