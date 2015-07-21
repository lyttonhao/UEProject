import cv2

def putAD(bg, fg, offsetx, offsety, scale=1.0):
	'''
	put fg on bg with corresponding offset and scale
	bg: background image
	fg: foreground image
	offsetx, offsety: offset of top-left corner in background image
	scale: scaling factor of foreground image
	'''
	#resize foreground image with scale
	fg = cv2.resize(fg, None, fx = scale, fy = scale, interpolation = cv2.INTER_CUBIC)

	#convert gray image to RGB image
	if bg.shape[2] == 1:
		bg = cv2.cvtColor(bg, cv2.CV_GRAY2BGR)
	if bg.shape[2] == 1:
		fg = cv2.cvtColor(fg, cv2.CV_GRAY2BGR)

	(bg_h, bg_w) = bg.shape[:2]
	(fg_h, fg_w, fg_c) = fg.shape
	if (offsetx + fg_w > bg_w) or (offsety + fg_h > bg_h):
		print "warning: foreground image exceeds background image!"

	result = bg.copy()
	h = min(bg_h - offsety, fg_h)
	w = min(bg_w - offsetx, fg_w)
	if fg_c == 3:
		result[offsety : offsety+h, offsetx : offsetx+w] = fg[0 : h, 0 : w].copy()
	else: 
		# deal with Alpha channel
		for i in range(h):
			for j in range(w):
				result[offsety+i, offsetx+j] = result[offsety+i, offsetx+j] * (1 - fg[i, j, 3] / 255.0) \
					+ fg[i, j, :3] * (fg[i, j, 3] / 255.0)

	return result 

if __name__ == "__main__":
	
	bg = cv2.imread('images/test1.jpg')  # No Alpha channel
	fg = cv2.imread('images/fg1.png', cv2.IMREAD_UNCHANGED)  # Maybe including Alpha channel

	result = putAD(bg, fg, 200, 400, 1.0)

	cv2.imshow('img1', fg)

	cv2.imshow('img', result)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
