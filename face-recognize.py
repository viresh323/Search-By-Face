import face_recognition
import os
import shutil
import pickle

#function to get all the images with extension .jpg 
def get_images(rootdir):
    imageList = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            filepath = subdir + os.sep + file
    
            if filepath.endswith(".jpg"):
                imageList.append(filepath)

    return imageList

#function to store the encodings of the images to save time in case of larger dataset for repetative testing
def store_encodings(images):
    encodings = load_encodings()
    for img in images:
        if img not in encodings:
            unknown_image = face_recognition.load_image_file(img)
            x = face_recognition.face_encodings(unknown_image)
            encodings[img] = x
    
    with open('unknown_encodings_dataset.dat', 'wb') as f:
        pickle.dump(encodings, f)

#function to get the stores encodings from the pickle dump file
def load_encodings():
    if not os.path.exists('unknown_encodings_dataset.dat'):
        return {}

    with open('unknown_encodings_dataset.dat', 'rb') as f:
	    all_face_encodings = pickle.load(f)

    return all_face_encodings

#function to compare the images for the encodings
def compare_images(known_encoding, unknown_encoding):

    if len(unknown_encoding) > 0:
        result = face_recognition.compare_faces(unknown_encoding,known_encoding,tolerance=0.5)

    return any(result)

#main functio
def main():
    known_image = face_recognition.load_image_file("TestFace/test.jpg") # search face
    known_encoding = face_recognition.face_encodings(known_image)[0]
    images = get_images("Dataset") #dataset of pictures
    store_encodings(images)
    unknown_encodings = load_encodings()

    print ("Matched Images : ")
    for image in images:
        result = compare_images(known_encoding, unknown_encodings[image])
        if result : 
            #shutil.copy(image, 'Output')
            print("%s"%(image))
   

#entry point for the project
main()