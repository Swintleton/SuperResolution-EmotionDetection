import dlib
from PIL import Image
from skimage import io

import csv

# Read filenames from a given directory
base_path = 'datasets/CK+'
emotion_path = 'datasets/CK+/Emotion/*/*/*'
import glob
fileNames = glob.glob(emotion_path)


if __name__ == '__main__':
    open('dataset_results/ground_truth_emotions.txt', 'w').close()
    
    f = open('ground_truth_emotions.csv', 'a')
    writer = csv.writer(f)
    #writer.writerow(["emotion"])
    
    sum = 0
    
    for idx, file in enumerate(fileNames):
        with open(file) as t_file:
            emotion = t_file.readlines()[0].strip()[0]
        
        emotionFileNumber = int(fileNames[idx].split(base_path)[1].split("/")[4].split("_")[2])
        row = ""
        sum = sum + emotionFileNumber
        for i in range (emotionFileNumber):
            row += emotion + ", "
        writer.writerow([row])
    
    f.close()
    print(sum)
    
