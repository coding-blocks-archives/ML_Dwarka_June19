import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


filenames = os.listdir('face_dataset')
labels = [f[:-4] for f in filenames if f.endswith('.npy')]

print(labels)

mugshots = []

for fname in filenames:
	a = np.load('face_dataset/' + fname)
	mugshots.append(a)

print(len(mugshots))
mugshots = np.concatenate(mugshots, axis=0)
mugshots = mugshots.reshape((mugshots.shape[0], -1))
print(mugshots.shape)

labels = np.repeat(labels, 10)
labels = labels.reshape(labels.shape[0], -1)
print(labels.shape)

dataset = np.hstack((mugshots, labels))
print(dataset.shape)


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(dataset[:,:-1], dataset[:,-1])

cap = cv2.VideoCapture(0)

while True:

	ret, frame = cap.read()

	if not ret:
		continue

	cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Load haar cascade
	faces = cascade_classifier.detectMultiScale(frame, 1.3, 5) # Detect faces

#	print(len(faces))

	for face in faces:
		x,y,w,h = face # Tuple Unpacking

		cropped_face = frame[y:y+h, x:x+w]
		cropped_face = cv2.resize(cropped_face, (100,100))
		cropped_face = cropped_face.reshape((1,-1))	

		preds = knn.predict(cropped_face)

		cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2) # Drawing a box around the face
		cv2.putText(frame, preds[0], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)

	cv2.imshow("Feed", frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
