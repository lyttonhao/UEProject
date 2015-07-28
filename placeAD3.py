import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_eye.xml')
threshd_value = 200
threshd_expo = 0.35
threshd_skin = 130

def detect_faces( img ):
	'''
	detect faces in img by opencv face detect module
	return detected faces
	'''
	
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.1, 3, minSize=(img.shape[0]/3, img.shape[1]/3))

	return faces

def detectEye( img, rect ):
	'''
	detect eyes in faces
	'''
	mask = np.zeros( img.shape[:2], np.uint8)

	(x,y,w,h)= rect
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	eyes = eye_cascade.detectMultiScale( gray[y:y+h, x:x+w] )

	for (ex, ey, ew, eh) in eyes:
		mask[y+ey:y+ey+eh, x+ex:x+ex+ew] = 1

	return mask

def calMeanStd(x, l, r):
	'''
	calculate mean and std of x in the percentage of [l, r]
	'''
	y = np.sort(x)
	n = x.shape[0]
	return np.mean(y[l*n:r*n]), np.std(y[l*n:r*n])

def detect_skin( _img, _mask ):
	'''
	Calculate the skin region of _img according to the mean and std of YCrCb channels.
	_mask: initial mask

	Pixel which are in [u-2*sigma, u+2*sigma] are considered in the skin region
	'''
	scale = 2.0
	img = cv2.resize(_img, dsize = (0,0), fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_NEAREST)
	mask = cv2.resize(_mask, dsize = (0,0), fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_NEAREST)

	skinMask = np.zeros( img.shape[:2], np.uint8 )

	YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
	Y, Cr, Cb = YCrCb[:,:,0], YCrCb[:,:,1], YCrCb[:,:,2]
	Uy, Sy = calMeanStd(Y[mask == 1], 0.05, 0.95)
	Ucb, Scb = calMeanStd(Cb[mask == 1], 0.05, 0.95)
	Ucr, Scr = calMeanStd(Cr[mask == 1], 0.05, 0.95)

	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			skinMask[i,j] = 1 if Y[i,j] > Uy -  2*Sy and Y[i,j] < Uy + 2* Sy \
							and Cb[i,j] > Ucb - 2* Scb and Cb[i,j] < Ucb + 2* Scb \
							and Cr[i,j] > Ucr - 2* Scr and Cr[i,j] < Ucr + 2* Scr else 0

	skinMask = cv2.resize(skinMask, dsize = (_img.shape[1], _img.shape[0]), interpolation=cv2.INTER_NEAREST)

	return skinMask



def cal_exposure( img ):
	'''
	calculate exposure ratio of img according to the proportion of large value pixels
	'''
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	hist = cv2.calcHist( [gray], [0], None, [256], [0,256])
	chist = np.cumsum(hist, axis = 0)
	value =  (chist[255,0] - chist[threshd_value,0]) / chist[255,0]

	print value

	return value

def calAvgWhiten( img, mask ):
	'''
	Calculate the whiten level of mask region of img 
	by the Euclidean distance to (255,255,255)
	'''
	d = 255-img[mask == 1, :].astype(np.int32) 
	d = d*d
	value = np.mean( np.sqrt(d.sum(axis=1)) )
	
	print value

	return value

def grabcut( img, rect ):
	'''
	using grabcut of opencv to extract foreground face with inital rect region
	'''
	mask = np.zeros( img.shape[:2], np.uint8 )

	bgdModel = np.zeros((1,65), np.float64)
	fgdModel = np.zeros((1,65), np.float64)

	mask, bgdModel, fgdModel = cv2.grabCut( img, mask, rect, bgdModel, fgdModel, 4, cv2.GC_INIT_WITH_RECT )


	mask2 = np.where( (mask==2)|(mask==0), 0, 1 ).astype('uint8')

	return mask2

def gamma_correction(img, gamma = 1.5):
	'''
	return gamma correction result of img with parameter gamma
	'''
	vmap = [math.pow(x/255.0, 1.0/gamma)*255.0 for x in range(256)]
	hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
	l = hls[:, :, 1]

	newl = np.zeros( l.shape, dtype=np.uint8 )
	for i in range(l.shape[0]):
		for j in range(l.shape[1]):
			newl[i, j] = vmap[l[i,j]]

	hls[:,:,1] = newl

	result = cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)

	return result 



def whiten( img ):
	'''
	Case 3: whiten face in the img
	'''
	faces = detect_faces( img )

	if len(faces) > 1 or len(faces) == 0:
		print len(faces)
		print "more than one or no faces detected"
		return img

	x,y,h,w = faces[0]
	if (h < img.shape[0]/2.5 and w < img.shape[1]/2.5):
		print "face is too small"
		return img

	if cal_exposure( img ) > threshd_expo:
		print "don't meet exposure condition"
		return img

	rect = (x,y,h,w)
	mask = grabcut( img, rect )

	eyemask = detectEye( img, rect )
	mask = mask & (1-eyemask)

	skinMask = detect_skin( img, mask )

	if calAvgWhiten( img, skinMask ) < threshd_skin:
		print "skin is too white"
		return img

#	im = img*mask[:,:,np.newaxis]
#	cv2.imshow("mask", im)
#	im = img*skinMask[:,:,np.newaxis]
#	cv2.imshow("skinmask", im)

	ga_im = gamma_correction( img, 1.5 )
	bl_im = cv2.bilateralFilter(ga_im, 10, 30, 30)
	
	result = ga_im*(1-skinMask[:,:,np.newaxis]) + bl_im*skinMask[:,:,np.newaxis]

#	mask3 = cv2.cvtColor((1-skinMask)*255, cv2.COLOR_GRAY2BGR)
#	output = cv2.seamlessClone(ga_im, bl_im, mask3, (ga_im.shape[1]/2, ga_im.shape[0]/2), cv2.MIXED_CLONE)

	return result


if __name__ == '__main__':
	for i in range(1,9):
		print i
		im = cv2.imread('images3/image' +str(i) +'.jpg')

		result = whiten( im )

	#	cv2.imshow("result",result)

		cv2.imwrite("images3/result" +str(i) + ".png", result)

	#	cv2.waitKey(0)
	#	cv2.destroyAllWindows()











