import os
os.environ['TL_BACKEND'] = 'tensorflow'

import numpy as np
import tensorlayerx as tlx
from srgan import SRGAN_g, SRGAN_d
from config import config
import vgg
import cv2

import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

import dlib
from PIL import Image
from skimage import io

import csv
from knn_train import KNN
emotions = ["", "anger", "contempt", "discust", "fear", "happy", "sad", "supprise"]

# Read filenames from a given directory
img_base_path = 'datasets/CK+/'
img_path = img_base_path + 'cohn-kanade-images/*/*/*.png'
emotion_path = img_base_path + 'Emotion/*/*/*.txt'
import glob
fileNames = glob.glob(img_path)
emotionNames = glob.glob(emotion_path)

###====================== HYPER-PARAMETERS ===========================###
batch_size = 8
n_epoch_init = config.TRAIN.n_epoch_init
n_epoch = config.TRAIN.n_epoch
# create folders to save result images and trained models
save_dir = "upscaled"
checkpoint_dir = "models"
landmarkFilePath = "dataset_results/landmarks.txt"
landmarkFilePathWithoutSR = "dataset_results/landmarks_without_sr.txt"
predictedEmotionFilePath = "dataset_results/predictedEmotions.txt"
predictedEmotionFilePathWithoutSR = "dataset_results/predictedEmotions_without_sr.txt"
predictedEmotionFilePathCustom = "dataset_results/predictedEmotions_custom.txt"
tlx.files.exists_or_mkdir(save_dir)
tlx.files.exists_or_mkdir(checkpoint_dir)
tlx.files.exists_or_mkdir("dataset_results")

G = SRGAN_g()
D = SRGAN_d()
VGG = vgg.VGG19(pretrained=False, end_with='pool4', mode='dynamic')
# automatic init layers weights shape with input tensor.
# Calculating and filling 'in_channels' of each layer is a very troublesome thing.
# So, just use 'init_build' with input shape. 'in_channels' of each layer will be automaticlly set.
G.init_build(tlx.nn.Input(shape=(8, 96, 96, 3)))
D.init_build(tlx.nn.Input(shape=(8, 384, 384, 3)))

landmark_ids = [
    0, 4, 17, 46, 48, 50, 61, 105, 107, 122,
    130, 133, 145, 159, 206, 276, 280, 289, 292, 334,
    336, 351, 359, 362, 374, 386, 426,
]

def sr_upscale(image, saveImage=False):
    ###========================LOAD WEIGHTS ============================###
    G.load_weights(os.path.join(checkpoint_dir, 'g.npz'), format='npz_dict')
    G.set_eval()
    
    #valid_lr_img = tlx.vision.load_images(path=config.VALID.lr_img_path)[0]
    valid_lr_img = image
    
    valid_lr_img_tensor = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]

    valid_lr_img_tensor = np.asarray(valid_lr_img_tensor, dtype=np.float32)
    valid_lr_img_tensor = valid_lr_img_tensor[np.newaxis, :, :, :]
    valid_lr_img_tensor= tlx.ops.convert_to_tensor(valid_lr_img_tensor)
    size = [valid_lr_img.shape[0], valid_lr_img.shape[1]]

    out = tlx.ops.convert_to_numpy(G(valid_lr_img_tensor))
    out = np.asarray((out + 1) * 127.5, dtype=np.uint8)
    #print("LR size: %s /  generated HR size: %s" % (size, out.shape))
    #print("[*] save images")
    if (saveImage):
        #tlx.vision.save_image(out[0], file_name='upscaled.png', path=save_dir)
        cv2.imwrite(save_dir + '/upscaled.png', out[0])
    return out[0]


def face_detection(image, idx, saveImage=False):
    # For static images:
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
        
        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Draw face detections of each face.
        if not results.detections:
            return image
        for detection in results.detections:
            #mp_drawing.draw_detection(image, detection)
            
            h, w, c = image.shape
            ymin = detection.location_data.relative_bounding_box.ymin
            xmin = detection.location_data.relative_bounding_box.xmin
            height = detection.location_data.relative_bounding_box.height
            width = detection.location_data.relative_bounding_box.width
            
            cropped_image = image[
                int(ymin * h):int(ymin * h) + int(height * h),
                int(xmin * w):int(xmin * w) + int(width * w)
            ]

            """print('Nose tip:')
            print(mp_face_detection.get_key_point(
            detection, mp_face_detection.FaceKeyPoint.NOSE_TIP))"""
        if(saveImage):
            cv2.imwrite('faces/cropped_image' + str(idx) + '.png', cropped_image)
        return cropped_image

def face_mash(image, fileName, idx, saveImage=False, saveLandmarks=True, detectEmotion=False, predictedEmitionFile=False):
    # For static images:
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:

        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            return image
        
        #Get the important 27 landmark coordinates
        important_27_landmarks = []
        for face_landmarks in results.multi_face_landmarks:
            for i in range(27):
                important_27_landmarks.append(face_landmarks.landmark[landmark_ids[i]])
        
        landmark = []
        if(detectEmotion):
            #Get 27 landmark coordinates
            for face_landmarks in important_27_landmarks:
                landmark.append(face_landmarks.x)
                landmark.append(face_landmarks.y)
                landmark.append(face_landmarks.z)
            #Predict emotion based on 27 landmarks
            predictedEmotionWriter = csv.writer(predictedEmitionFile)
            predictedEmotionWriter.writerow([ KNN.predict([landmark])[0] ])
            #print(emotions[KNN.predict([landmark])[0]])
        
        if(saveLandmarks):
            writer = csv.writer(landmarksFile)
            
            #print('face_landmarks:', face_landmarks)
            writer.writerow(["["])
            for face_landmarks in important_27_landmarks:
                writer.writerow([
                    str(face_landmarks.x) + ", " +
                    str(face_landmarks.y) + ", " +
                    str(face_landmarks.z) + ","
                ])
            writer.writerow(["],"])
          
        if(saveImage):
            #save images with 27 landmarks
            for face_landmarks in important_27_landmarks:
                shape = image.shape 
                relative_x = int(face_landmarks.x * shape[1])
                relative_y = int(face_landmarks.y * shape[0])
                cv2.circle(image, (relative_x, relative_y), radius=1, color=(225, 0, 100), thickness=1)
            cv2.imwrite('face_mashes/img_with_27_landmarks' + str(idx) + '.png', image)
        
            #save images with face mesh
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                  image=image,
                  landmark_list=face_landmarks,
                  connections=mp_face_mesh.FACEMESH_TESSELATION,
                  landmark_drawing_spec=None,
                  connection_drawing_spec=mp_drawing_styles
                  .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                  image=image,
                  landmark_list=face_landmarks,
                  connections=mp_face_mesh.FACEMESH_CONTOURS,
                  landmark_drawing_spec=None,
                  connection_drawing_spec=mp_drawing_styles
                  .get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                  image=image,
                  landmark_list=face_landmarks,
                  connections=mp_face_mesh.FACEMESH_IRISES,
                  landmark_drawing_spec=None,
                  connection_drawing_spec=mp_drawing_styles
                  .get_default_face_mesh_iris_connections_style())

            cv2.imwrite('face_mashes/img_with_face_mash' + str(idx) + '.png', image)
        
        if(saveLandmarks):
            writer.writerow([])


if __name__ == '__main__':
    saveImage = True
    upscale = True
    saveLandmarks = False
    trainKNN = False
    detectEmotion = False
    customImage = True
    customImageFilePath = "input/S053_003_00000038.png"
    
    if(trainKNN):
        KNN.train()
        exit()
    
    predictedEmitionFile = False
    if(detectEmotion):
        if(upscale == False):
            predictedEmotionFilePath = predictedEmotionFilePathWithoutSR
        if(customImage):
            predictedEmotionFilePath = predictedEmotionFilePathCustom
        open(predictedEmotionFilePath, 'w').close()
        predictedEmitionFile = open(predictedEmotionFilePath, 'a')
    
    if(saveLandmarks):
        if(upscale == False):
            landmarkFilePath = landmarkFilePathWithoutSR
        open(landmarkFilePath, 'w').close()
        landmarksFile = open(landmarkFilePath, 'a')
        writer = csv.writer(landmarksFile)
    
    #Process a custom image
    if(customImage):
        print(customImageFilePath)
        
        image = cv2.imread(customImageFilePath)
        image = face_detection(image, 0, saveImage)
        if(upscale):
            image = sr_upscale(image, saveImage)
        face_mash(image, customImageFilePath, 0, saveImage, saveLandmarks, detectEmotion, predictedEmitionFile)
    
    #Process a dataset
    if(customImage == False):
        emotion_idx = 0
        sum = 0
        for idx, file in enumerate(fileNames):
            #Only deal with those folders that has emotion labels
            imageFolderName = fileNames[idx].split(img_base_path + 'cohn-kanade-images/')[1].split("/")
            emotionFolderName = emotionNames[emotion_idx].split(img_base_path + 'Emotion/')[1].split("/")
            
            if(imageFolderName[0] != emotionFolderName[0] or imageFolderName[1] != emotionFolderName[1]):
                continue
            sum = sum + 1
            
            print(file)
            
            image = cv2.imread(file)
            image = face_detection(image, idx, saveImage)
            if(upscale):
                image = sr_upscale(image, saveImage)
            face_mash(image, file, idx, saveImage, saveLandmarks, detectEmotion, predictedEmitionFile)
            
            #Increase emotion index if we are at the last image inside the folder
            imageFileNumber = int(fileNames[idx].split(img_base_path + 'cohn-kanade-images/')[1].split("/")[2].split("_")[2].split(".")[0])
            emotionFileNumber = int(emotionNames[emotion_idx].split(img_base_path + 'Emotion/')[1].split("/")[2].split("_")[2])
            if(imageFileNumber == emotionFileNumber):
                emotion_idx += 1
    
    if(saveLandmarks):
        landmarksFile.close()
    if(detectEmotion):
        predictedEmitionFile.close()

    print(sum)
    print("Done")
