import cv2
import numpy as np
import matplotlib.pyplot as plt


i = 0
backSub = cv2.createBackgroundSubtractorKNN()
video_file = "sMoky.avi"
video = cv2.VideoCapture(video_file)
success, firstframe = video.read()
firstframe = cv2.resize(firstframe,(480,360))
thresh = 25


def back_sub(frame):
    frame_backsub = backSub.apply(frame)
    # frame_backsub[frame_backsub < 255] = 0
    # output = cv2.bitwise_and(frame, frame, mask=frame_backsub)
    return frame_backsub 


def smoke_detect(frame):
	

    # Motion detection using Frame Subtraction and Thresholding to determine difference between grayscale frames 
    kernel = np.ones((2,2),np.uint8)
    bgsub = back_sub(frame)
    bgsub_erosion = cv2.erode(bgsub,kernel,iterations = 2)
    bgsub_dilation = cv2.dilate(bgsub_erosion,kernel,iterations = 1)
    bgsub_dilation = np.expand_dims(bgsub_dilation,axis=2)
    bgsub_dilation = np.concatenate((bgsub_dilation,bgsub_dilation,bgsub_dilation),axis=2)
    bgsub_frame = cv2.bitwise_and(frame,bgsub_dilation)


    diff = cv2.absdiff(cv2.cvtColor(firstframe,cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
    diff[diff <= thresh] = 0
    diff[diff > thresh] = 255
    

    # print(seq_frame.shape)
    diff = np.expand_dims(diff,axis=2)
    diff = np.concatenate((diff,diff,diff),axis=2)
    # print(diff.shape)


    # Subtract all unnecesssary static background
    frame_and = cv2.bitwise_and(frame,diff)


    #Smoke color model
    m = np.max(frame_and,axis=2)
    n = np.min(frame_and,axis=2)
    delta = m-n
    delta = cv2.inRange(delta, 5, 20)
    I = cv2.cvtColor(frame_and,cv2.COLOR_BGR2GRAY)
    mask0 = cv2.inRange(I, 80, 150)
    mask1 = cv2.inRange(I, 190, 255)
    mask2 = cv2.bitwise_or(mask0, mask1)
    mask2 = cv2.bitwise_or(mask2, delta)
    mask = cv2.bitwise_and(mask2,I)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    result = cv2.bitwise_and(frame,mask)
    cv2.waitKey(100)


    # #Fast Fourier Transform
    # I = np.max(result,axis = 2)
    # fft = np.fft.fft2(I)
    # fft_shift = np.fft.fftshift(fft)
    # magnitude_spectrum = 20*np.log(np.abs(fft_shift))
    # magnitude_spectrum = np.asarray(magnitude_spectrum, dtype = np.uint8)


    erosion = cv2.erode(mask,kernel,iterations = 2)
    dilation = cv2.dilate(erosion,kernel,iterations = 1)
    erosion = cv2.resize(erosion,(480,360))
    dilation = cv2.resize(dilation,(480,360))
    dilation[dilation > 1] = 255
    
    result = cv2.bitwise_and(bgsub_frame,dilation)
    result = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
    result[result > 1] = 255

    screen = np.hstack((erosion, dilation))
    mask = cv2.resize(mask,(480,360))
    n_white = np.sum(result == 255)
    # print('Number of white pixels: ', n_white)



    return n_white, dilation, bgsub_frame, result


	# ret, seq_frame = video.read()
	# if cv2.waitKey(40) == 27:
	#     break


delta_s = 0
s_list = []

# cv2.waitKey(1000)
while video.isOpened():

	ret, seq_frame = video.read()
	seq_frame = cv2.resize(seq_frame,(480,360))
	s_smoke, dilation, bg, result = smoke_detect(seq_frame)


	# cv2.imshow('bg',bg)
	cv2.imshow('result', result)
	# print('s_smoke: ', s_smoke)
	if len(s_list) == 5:
		s_list = s_list[1:] + [s_smoke]
		if s_list[0] != 0:

			# s_list = [s_list[1:], s_smoke]
			s_list_sub = s_list[1:]
			# print('s_list_sub: ',s_list_sub)
			# print('s_list[0]: ',s_list[0])

			T = (np.array(s_list_sub) - np.array(s_list[0])) / np.array(s_list[0])
			std = np.std(T)
			if std > 0.1:
				print('Warning! ')
			# print("T: ", T)
			print("STD: ", std)
		else:
			print("no STD")
	else:
		s_list = s_list + [s_smoke]


	screen = np.hstack((seq_frame, dilation))
	cv2.imshow('s',screen)



    # ret, seq_frame = video.read()
    # ret, seq_frame1 = video.read()



    # # Motion detection using Frame Subtraction and Thresholding to determine difference between grayscale frames 
    # diff = cv2.absdiff(cv2.cvtColor(firstframe,cv2.COLOR_BGR2GRAY), cv2.cvtColor(seq_frame,cv2.COLOR_BGR2GRAY))
    # diff[diff <= thresh] = 0
    # diff[diff > thresh] = 255
    

    # # print(seq_frame.shape)
    # diff = np.expand_dims(diff,axis=2)
    # diff = np.concatenate((diff,diff,diff),axis=2)
    # # print(diff.shape)


    # # Subtract all unnecesssary static background
    # frame_and = cv2.bitwise_and(seq_frame,diff)


    # #Smoke color model
    # m = np.max(frame_and,axis=2)
    # n = np.min(frame_and,axis=2)
    # delta = m-n
    # delta = cv2.inRange(delta, 5, 20)
    # I = cv2.cvtColor(frame_and,cv2.COLOR_BGR2GRAY)
    # mask0 = cv2.inRange(I, 80, 150)
    # mask1 = cv2.inRange(I, 190, 255)
    # mask2 = cv2.bitwise_or(mask0, mask1)
    # mask2 = cv2.bitwise_or(mask2, delta)
    # mask = cv2.bitwise_and(mask2,I)
    # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # result = cv2.bitwise_and(seq_frame,mask)
    # cv2.waitKey(100)


    # #Fast Fourier Transform
    # I = np.max(result,axis = 2)
    # fft = np.fft.fft2(I)
    # fft_shift = np.fft.fftshift(fft)
    # magnitude_spectrum = 20*np.log(np.abs(fft_shift))
    # magnitude_spectrum = np.asarray(magnitude_spectrum, dtype = np.uint8)

    
    # # print(magnitude_spectrum)
    # # print(np.max(magnitude_spectrum))
    # # cv2.imshow('m',magnitude_spectrum)
    # # cv2.waitKey(0)
    # # dft = cv2.dft(I, flags = cv2.DFT_COMPLEX_OUTPUT)
    # # dft_shift = np.fft.fftshift(dft)
    # # cv2.imshow('d',dft_shift)
    # # blur = cv2.GaussianBlur(gray, (5,5), 0)
    # # _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    # # dilated = cv2.dilate(thresh, None, iterations=3)


    # kernel = np.ones((2,2),np.uint8)
    # erosion = cv2.erode(mask,kernel,iterations = 2)
    # dilation = cv2.dilate(erosion,kernel,iterations = 1)
    # erosion = cv2.resize(erosion,(480,360))
    # dilation = cv2.resize(dilation,(480,360))
    # dilation[dilation > 1] = 255
    # screen = np.hstack((erosion, dilation))
    # mask = cv2.resize(mask,(480,360))
    # n_white = np.sum(dilation == 255)
    # print('Number of white pixels 0: ', n_white)
    # cv2.imshow('',screen)
    # cv2.imshow('mask',mask)


    # ret, seq_frame = video.read()
    # if cv2.waitKey(40) == 27:
    #     break

# while True:

#     (grabbed, frame) = video.read()
#     if not grabbed:
#         break

#     firstframe = getFirstFrame(video)

#     gray = cv2.cvtColor(firstframe, cv2.COLOR_BGR2GRAY)
#     background = cv2.GaussianBlur(gray, (21, 21), 0)

#     cv2.imshow('background',background)

#     keyboard = cv2.waitKey(0)
#     if keyboard == 'q' or keyboard == 27:
#         break

#     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # frame_backsub = back_sub(frame)
#     # frame_filter = filter_image(frame)

#     # output = cv2.bitwise_and(frame_backsub, frame_filter)

#     # # frame_backsub1 = back_sub1(frame)

#     # # cv2.imshow('Frame_Sub', frame_backsub)
#     # # cv2.imshow('Frame_Sub1', frame_backsub1)

#     # cv2.imshow('Frame', frame)
#     # cv2.imshow('Output', output)

#     # keyboard = cv2.waitKey(30)
#     # if keyboard == 'q' or keyboard == 27:
#     #     break

 
cv2.destroyAllWindows()
video.release()