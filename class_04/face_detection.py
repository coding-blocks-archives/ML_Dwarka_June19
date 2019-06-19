import cv2

cap = cv2.VideoCapture(0)

while True:

	ret, frame = cap.read()

	if not ret:
		continue

	cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Load haar cascade
	faces = cascade_classifier.detectMultiScale(frame, 1.3, 5) # Detect faces

	print(len(faces))

	for face in faces:
		x,y,w,h = face # Tuple Unpacking

# 					Fram,e Start, ENd, Color, Thickness
		cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2) # Drawing a box around the face

#	print(frame.shape)
	cv2.imshow("Feed", frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
