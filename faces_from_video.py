import cv2


def main():
    capture = cv2.VideoCapture('E:/videos/Sophia.mp4')
    counter = 1
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 8)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 255), 8)
            face = frame[(y-10):(y+h+10), (x-10):(x+w+10)]
            cv2.imshow('frame', cv2.resize(frame, (300, 300)))
            try:
                resized = cv2.resize(face, (32, 32), interpolation=cv2.INTER_AREA)
                cv2.imwrite('E:/faces32/Laetitia/' + str(counter).zfill(6) + '.jpg', resized)
                counter += 1
            except Exception as e:
                print(str(e))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
