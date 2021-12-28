import tensorflow as tf
import numpy as np
import cv2

labels = ['<name 1>', '<name 2>', '<name 3>', '<name 4>', '<name 5>']
model = tf.keras.models.load_model('E:/models/faces.h5')
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    try:
        ret, frame = capture.read()
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 8)
        for (x, y, w, h) in faces:
            face = frame[(y - 10):(y + h + 10), (x - 10):(x + w + 10)]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            if face is not None:
                resized = cv2.resize(face, (32, 32), interpolation=cv2.INTER_AREA)
                resized_tensor = np.expand_dims(resized, axis=0)
                resized_tensor = resized_tensor.astype(float)
                resized_tensor /= 255.0
                prediction = model.predict(resized_tensor)
                orig = (x - 10, y - 30)
                cv2.putText(frame, labels[np.argmax(prediction[0], axis=0)], orig,
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 10, cv2.LINE_AA)
            cv2.imshow('frame', frame)
    except Exception as e:
        print(e)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
