import numpy as np
import pandas as pd
import cv2
import os
from math import ceil
import csv
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def textPreprocessing(data):
    #Removing Punctuations
    data = [s.translate(str.maketrans('', '', string.punctuation)) for s in data]

    #Converting text to lower case and calculating max length
    size_1 = []
    for i in range(len(data)):
        data[i] = data[i].lower()
        size_1.append(len(data[i].split()))

    #Max Sentence Length
    max_sentence_length = max(size_1)

    #Converting Text into Tokens
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)
    seq = tokenizer.texts_to_sequences(data)
    seq = pad_sequences(seq, maxlen=max_sentence_length, padding='post')
    vocab_size = len(tokenizer.word_index) + 1
    print(max_sentence_length,vocab_size)

    return seq,max_sentence_length,vocab_size

#To Find the selected Images
def takespread(sequence, num):
    length = float(len(sequence))
    a = []
    for i in range(num):
        a.append(sequence[int(ceil(i * length / num))])
    return a


def framing(folderImages,window_size=8,stride=4):
    num_images = len(folderImages)
    # print(num_images)
    # print("AA")
    meta_frames = int(np.floor((num_images-window_size)/stride)) + 1
    # print(meta_frames)
    output = []
    for i in range(meta_frames):
        imgs=[]
        for j in range(i*stride,window_size+i*stride):
            imgs.append(folderImages[j])
        output.append(imgs)
    # print(len(output))
    return output


annonationsPath = 'G:\\phoenix-2014.v3\\phoenix2014-release\\phoenix-2014-multisigner\\annotations\\manual\\train.xlsx'
annonationData = pd.read_excel(annonationsPath)
seq,max_length,vocab_size = textPreprocessing(annonationData.iloc[:,3].values)
id = annonationData.iloc[:,0].values
finalOutput = {}
for i in range(len(id)):
    finalOutput[id[i]] = seq[i]

X_train = []
y_train = []

parentFolder = 'G:\\phoenix-2014.v3\\phoenix2014-release\\phoenix-2014-multisigner\\features\\fullFrame-210x260px\\train\\'
for folder in os.listdir(parentFolder)[:40]:
    print(folder)
    for subfolder in os.listdir(parentFolder+folder):
        childPath = parentFolder+folder+'\\'+subfolder
        numFiles = len(os.listdir(childPath))
        images = os.listdir(childPath)
        selectedImages = []
        if(numFiles>100):
            encoding = finalOutput[folder]
            # print(encoding)
            # print(numFiles)
            indexes = takespread(np.arange(1,numFiles,1),100)
            # print(indexes)
            for i in indexes:
                selectedImages.append(images[i])

            folderImages = []
            for image in selectedImages:
                # print(image)
                img = cv2.imread(childPath+'\\'+image)
                img = cv2.resize(img,(224,224), interpolation = cv2.INTER_AREA)
                folderImages.append(img)
            framed_images = framing(folderImages)
            X_train.append(framed_images)
            y_train.append(encoding)

X_train = np.array(X_train)
print(X_train.shape)
y_train = np.array(y_train)
print(y_train.shape)

np.save('X_train.npy',X_train)
np.save('y_train.npy',y_train)
        
        
            
            

            

            
                


