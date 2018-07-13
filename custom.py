import cv2
import numpy as np
from imutils import resize

def normalize_image_size(image, a, b):
    width = a
    height = b
    (h,w) = image.shape[:2]
    if w > h:
        image = resize(image, width=width)
    else:
        image = resize(image, height=height)
    padW = int((width-image.shape[1])/2.0)
    padH = int((height-image.shape[0])/2.0)
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
        cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))
    return image

def get_letters_list2(im_name):
    image = cv2.imread(im_name,0)
    image = cv2.bitwise_not(image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    min_size = 100
    for i in range(0, nb_components):
        if sizes[i] <= min_size:
            image[output == i + 1] = 0
    edge = 16
    inv_cropped = image[edge:image.shape[0]-edge,edge:image.shape[1]-edge]
    new_image, letters_only_contour, hierarchy = cv2.findContours(inv_cropped,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    letters_list = []
    cropped = cv2.bitwise_not(inv_cropped)
    # sorts contours by x position
    starting_points = []
    for cnt in letters_only_contour:
        x,y,w,h = cv2.boundingRect(cnt)
        starting_points.append(x)
    sorted_letters_only_contour = [x for _,x in sorted(zip(starting_points,letters_only_contour))]
    # append letters to list.  following code detects unually long letters, and assumes the contour found multiple letters
    # splits such contours in half until it's a reasonable length
    for cnt in sorted_letters_only_contour:
        x,y,w,h = cv2.boundingRect(cnt)
        temp_width = w
        while temp_width > 34:
            if temp_width < 60:
                temp_width = np.floor_divide(temp_width,2)
            elif temp_width < 91:
                temp_width = np.floor_divide(temp_width,3)
            else:
                temp_width = np.floor_divide(temp_width,2)
        for i in range(np.floor_divide(w,temp_width)):
            letters_list.append(cropped[y:y+h,x+i*temp_width:x+(i+1)*temp_width])
    while len(letters_list)<6:
        letter_widths = []
        for letter in letters_list:
            letter_widths.append(letter.shape[1])
        widest_letter_index = letter_widths.index(max(letter_widths))
        widest_letter = letters_list[widest_letter_index]
        letters_list.remove(widest_letter)
        letters_list.insert(widest_letter_index,widest_letter[:,np.floor_divide(widest_letter.shape[1],2):])
        letters_list.insert(widest_letter_index,widest_letter[:,0:np.floor_divide(widest_letter.shape[1],2)])
    if len(letters_list)>6:
        return ('An error has occurred.  Too many letters detected',letters_list)
    else:
        return letters_list

