import cmake
import face_recognition as fr
import cv2

#loading both images
img1 = fr.load_image_file("images/download.jpg")
img2 = fr.load_image_file("images/images.jpg")

#convert the images to its original colors
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

#here it return the array of values extracting from the images
#and by comparing these values we check whether they are similar or not
enc1 = fr.face_encodings(img1)[0]
enc2 = fr.face_encodings(img2)[0]

#here we compare the values of both images, here we check whether the second match to first or not
#by default tolerance is 0.6

rst = fr.compare_faces([enc1],enc2)[0]
print(rst)

if(rst):
    print("they are same...")
else:
    print("They are not same...")

loc = fr.face_locations(img2)[0]
print(loc)

top, right,bottom,left = loc
faceimage = img2[top:bottom,left:right]

cv2.imshow("face image",faceimage)
cv2.waitKey(0)
