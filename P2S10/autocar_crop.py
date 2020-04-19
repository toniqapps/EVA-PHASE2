from PIL import Image
import numpy as np 

'''
Since rotating the image with expand=1 which adds padding to the image 
with pixel value of 255 which we convert to 0 by masking white image and 
creating a composite image using alpha layer and convert the image to grayscale
'''
def rotate_grayscale_img(img, angle):
    img_1 = img.convert('RGBA')
    # rotated image
    img_rt = img_1.rotate(angle, expand=1)
    # a white image same size as rotated image
    white_mask = Image.new('RGBA', img_rt.size, (255,)*4)
    # create a composite image using the alpha layer of rot as a mask
    out = Image.composite(img_rt, white_mask, img_rt)
    return out.convert('L')

#pad the image with given x and y pixel
def pad(img, x, y):
    return np.pad(img, (x, y), 'constant',constant_values=255)

'''
Step1 : Convert the image to numpy array
Step2 : Get startx and starty cordinates from the x,y as image center subtracted by half crop size
Step3 : If startx and starty and less than or greater than image width and height pad the image with missing pixels
Step4 : Crop the image based on startx, stary, cropx, cropy values
'''
def crop_center(img, x, y, cropx=300, cropy=300):
    #convert image to numpy
    img_np = np.asarray(img)
    max_x, max_y = img_np.shape
    startx = int(x - (cropx//2))
    starty = int(y - (cropy//2))
    pad_x_0 = 0
    pad_x_1 = 0
    pad_y_0 = 0
    pad_y_1 = 0
    
    #Pad image if x, y coordinates are close to max, min coordinates 
    if startx < 0:
        pad_x_0 = -startx
        startx=0
    if (startx + cropx) > max_x:
        pad_x_1 = startx + cropx - max_x
    if starty < 0:
        pad_y_0 = -starty
        starty=0
    if (starty + cropy) > max_y:
        pad_y_1 = starty + cropy - max_y
        
    img_pad = pad(img_np, (pad_x_0, pad_x_1), (pad_y_0, pad_y_1))
    return Image.fromarray(img_pad[startx:startx+cropx, starty:starty+cropy])

'''
Step1: Crop Image taking x,y as center and cropping the image 5 time from the actual cropx and cropy value
Step2: Rotate the cropped image using defined angle and return gray scale image
Step3: Recrop the cropped image after rotation 3 times the actual crop values
Step4: Resize the image based on the size defined
''' 
def croppedImage(img, x, y, angle, size, cropx=40, cropy=40):
    img_crop = crop_center(img, x, y, np.multiply(cropx,5), np.multiply(cropy,5))
    img_crop_rt = rotate_grayscale_img(img_crop, angle)
    width, height = img_crop_rt.size 
    img_crop_1 =  crop_center(img_crop_rt, width/2,height/2,np.multiply(cropx,3),np.multiply(cropx,3))
    return img_crop_1.resize((size,size), Image.ANTIALIAS)