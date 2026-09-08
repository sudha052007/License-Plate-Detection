# License-Plate-Detection
Russian License Plate Blurring
Welcome to your object detection project! Your goal will be to use Haar Cascades to blur license plates detected in an image!

Ways to Approach this project:
Use the given image (car_plate.jpg) and create a function that will blur the image of its license plate. Check out the correct pre-trained .xml file (given) to use.
Use this notebook! Here we offer a guide of what main steps you should take to complete the project.
TASK: Import the usual libraries you think you'll need.
 # program & output
~~~
PROGRAM DEVELOPED BY
NAME: SUDHARSHINI .G
REGISTER NO : 212225220106


# importing libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

# read the image
img = cv2.imread('car_plate.jpg')

def display(img):
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
display(img)

plate_cascade = cv2.CascadeClassifier(
    'haarcascade_russian_plate_number.xml'
)

## TASK: Create a function that takes in an image and draws a rectangle around what it detects to be a license plate. Keep in mind we're just drawing a rectangle around it for now, later on we'll adjust this function to blur. You may want to play with the scaleFactor and minNeighbor numbers to get good results.
def detect_plate(img):

    plate_img = img.copy()

    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    for (x, y, w, h) in plates:

        cv2.rectangle(
            plate_img,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            3
        )

    return plate_img
result = detect_plate(img)
display(result)

## FINAL TASK: Edit the function so that is effectively blurs the detected plate, instead of just drawing a rectangle around it. Here are the steps you might want to take:

The hardest part is converting the (x,y,w,h) information into the dimension values you need to grab an ROI. you just need to convert the information about the top left corner of the rectangle and width and height, into indexing position values.
Once you've grabbed the ROI using the (x,y,w,h) values returned, you'll want to blur that ROI. You can use cv2.medianBlur for this.
Now that you have a blurred version of the ROI (the license plate) you will want to paste this blurred image back on to the original image at the same original location. Simply using Numpy indexing and slicing to reassign that area of the original image to the blurred roi.

def detect_and_blur_plate(img):

    plate_img = img.copy()

    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    for (x, y, w, h) in plates:

        plate = plate_img[y:y+h, x:x+w]

        blurred = cv2.GaussianBlur(
            plate,
            (51, 51),
            0
        )

        plate_img[y:y+h, x:x+w] = blurred

    return plate_img

result = detect_and_blur_plate(img)
display(result)
~~~

# output
<img width="876" height="512" alt="image" src="https://github.com/user-attachments/assets/8e6bcb1c-2ef0-463a-a416-1ba353fc91c3" />

<img width="877" height="511" alt="image" src="https://github.com/user-attachments/assets/1198ddec-a16f-4e9d-9f65-11bdcec1177f" />

<img width="881" height="540" alt="image" src="https://github.com/user-attachments/assets/f11a3868-5e96-4c16-a425-4778fb19a165" />
