import cv2

def detect_and_crop_face(image_path):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for i, (x, y, w, h) in enumerate(faces):
        cropped_face = image[y:y+h, x:x+w]
        cv2.imshow(f'Detected Face {i+1}', cropped_face)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = 'Images\Leonardo DiCaprio.jpg'
xml_path = 'haarcascade_frontalface_default.xml'

detect_and_crop_face(image_path)
