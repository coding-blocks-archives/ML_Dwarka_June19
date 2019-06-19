import numpy as np
import cv2

cap = cv2.VideoCapture(0)

label = input("Enter Name: ")
num = input("Enter Number of Photos to be taken: ") or 10
num = int(num)

mugshots = []

while num:

	ret, frame = cap.read()

	if not ret:
		continue

	cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Load haar cascade
	faces = cascade_classifier.detectMultiScale(frame, 1.3, 5) # Detect faces

	faces = sorted(faces, key=lambda e: e[2]*e[3], reverse=True)
	if not faces:
		continue
	
	faces = [faces[0]]

#	print(len(faces))

	for face in faces:
		x,y,w,h = face # Tuple Unpacking

		cropped_face = frame[y:y+h, x:x+w]
#		print(cropped_face.shape)
		cropped_face = cv2.resize(cropped_face, (100,100))
#		print(cropped_face.shape)
		mugshots.append(cropped_face)
		num -= 1

#	print(frame.shape)
	cv2.imshow("Feed", frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

mugshots = np.array(mugshots)

print(mugshots.shape)
print("Mugshots Taken: ", len(mugshots))

np.save('face_dataset/' + label, mugshots)
cap.release()
cv2.destroyAllWindows()










