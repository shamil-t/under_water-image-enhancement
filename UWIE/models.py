from django.db import models
# Create your models here.


class InputCLAHE(models.Model):
	img = models.ImageField(upload_to = "UWIE/static/Input/CLAHE/")
	def __str__(self):
		return self.img

class InputRAY(models.Model):
	img = models.ImageField(upload_to = "UWIE/static/Input/RAY/")
	def __str__(self):
		return self.img

class InputDCP(models.Model):
	img = models.ImageField(upload_to = "UWIE/static/Input/DCP/")
	def __str__(self):
		return self.img
	
class InputMIP(models.Model):
	img = models.ImageField(upload_to = "UWIE/static/Input/MIP/")
	def __str__(self):
		return self.img

class InputClassify(models.Model):
	img = models.ImageField(upload_to = "UWIE/static/Input/CLASSIFY/")
	def __str__(self):
		return self.img