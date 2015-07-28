import cv2
import math
import numpy as np
from skimage import segmentation, color
from skimage.future import graph
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from putAD import putAD

threshd_warm = 0 #warm color threshold
threshd_dark = 0.2 #dark color threshold
dark_value = 50  #below dark_value is considered dark

def measure_warm(im):
	'''
	measure warm degree by difference of R channel and B channel
	'''
	im = im.astype(float)
	value = sum( sum( im[:,:,2] - im[:,:,0] ) )

	return 1.0 * value / (255.0*im.shape[0]*im.shape[1])

def measure_dark(im):
	'''
	measure dark region proportion
	'''
	im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	value = sum( sum( im < dark_value ) )

	return 1.0 * value / (im.shape[0]*im.shape[1])

def check_bg(bg):
	'''
	check if background image meet requirements
	'''
	#print measure_warm(bg), measure_dark(bg)
	
	return (measure_warm(bg) >= threshd_warm) and (measure_dark(bg) >= threshd_dark)

def normalized_cut( img, ncut = 10 ):
	'''
	segment image by normalized_cut
	'''
	labels1 = segmentation.slic(img, convert2lab = True, compactness=40, n_segments=400)
#	out1 = color.label2rgb(labels1, img, kind='avg')

	g = graph.rag_mean_color(img, labels1, mode='similarity')
	labels2 = graph.cut_normalized(labels1, g, num_cuts=ncut)
#	out2 = color.label2rgb(labels2, img, kind='avg')

	return labels2

def cal_segmentation_score( bg, (h1, w1), rect ):
	'''
	calculate score accorrding to segmentation region
	'''
	label = normalized_cut(bg, 10)
	label1 = label[rect[0]:rect[1], rect[2]:rect[3]]

	lab = cv2.cvtColor( bg, cv2.COLOR_BGR2LAB  )	
	a = lab[rect[0]:rect[1], rect[2]:rect[3], 1]
	b = lab[rect[0]:rect[1], rect[2]:rect[3], 2]

	#computer stardard vaiation for each segmentation region
	x, c = np.unique(label1, return_counts = True)
	std = [np.std(a[label1 == each]) + np.std(b[label1 == each]) for each in x]
	stat = dict( zip(x, zip(c, std)) )


	score = np.zeros( bg.shape[:2] )
	for i in range(rect[0], rect[1]):
		for j in range(rect[2], rect[3]):
			#consider local std and the size of segmentation region
			local_std = np.std( lab[i-h1/3:i, j-w1/2:j+w1/2, 1]  ) + np.std( lab[i-h1/3:i, j-w1/2:j+w1/2, 2]  )
			score[i, j] = math.exp( -local_std/10 ) * math.exp( -stat[label[i][j]][1]/20 ) * math.sqrt( stat[label[i, j]][0] ) \
							

	return score

def cal_center_score( (h, w), h1, rect ):
	'''
	calculate score accorrding to the distance to center of image
	'''
	score = np.zeros( (h, w) )
	cx, cy = (rect[2]+rect[3])/2, (rect[0]+rect[1])/2

	for i in range(rect[0], rect[1]):
		for j in range(rect[2], rect[3]):
			score[i, j] =  math.exp( - ((i-cy)*(i-cy) + (j-cx)*(j-cx)) / (0.5*h*w)  )

	return score

def cal_heatmap( bg, fg ):
	'''
	calculate heatmap 
	'''

	h, w = bg.shape[:2]
	rect = [max(h*0.3, fg.shape[0]), h*0.9, 0.1*w, 0.9*w]
	rect = map(int, rect)

	score_seg = cal_segmentation_score( bg, fg.shape[:2], rect)
	score_center = cal_center_score( bg.shape[:2], fg.shape[0], rect )

#	visualize_heatmap( score_seg )
#	visualize_heatmap( score_center )

	heatmap = score_seg * score_center
#	visualize_heatmap( heatmap )

	return heatmap

def placeAD( bg, fg ):
	'''
	handle Case2, place fg image on bg image 
	'''

	if not check_bg(bg):
		print "background image don't meet the conditions"
		return bg

	heatmap = cal_heatmap( bg, fg )

#	visualize_heatmap( heatmap )

	offsetx = heatmap.argmax() % bg.shape[1]
	offsety = heatmap.argmax() / bg.shape[1]

	print offsetx, offsety
	#put fg on bg by (offsetx, offsety)
	result = putAD(bg, fg, offsetx - fg.shape[1]/2, offsety - fg.shape[0])

	return result



def visualize_heatmap( heatmap ):
#	heatmap = (heatmap - heatmap.min()) / (heatmap.max()-heatmap.min()) 
	plt.imshow( heatmap, cmap = cm.gist_heat )
	plt.colorbar()
	plt.show()


if __name__ == '__main__':
	bg = cv2.imread('images2/christ5.jpg')  # No Alpha channel
	fg = cv2.imread('images2/cola.jpg', cv2.IMREAD_UNCHANGED)  # Maybe including Alpha channel


	result = placeAD(bg, fg)

	cv2.imwrite("images2/result5.png", result)

	#visual_heatmap( heatmap )
	#visual_heatmap

	#cv2.imshow('img1', fg)

	cv2.imshow('result', result)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
