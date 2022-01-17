import os
import json

import numpy as np
import cv2

class PyArtGenerator:
    IMAGE_EXTENSION = ["jpg", "png", "jpeg", "jfif"]
    OUTPUT_DIR = "output"
    def __init__(self, config):
        self.config = config
    def createImage(self, width, height):
        self.img = blank_image = np.zeros(shape=[width, height, 4], dtype=np.uint8)
        return self.img

    def saveImage(self, file_name):
        cv2.imwrite(f"{PyArtGenerator.OUTPUT_DIR}/{file_name}.png", self.img)

    def loadImages(self):
        self.imgObjs = []
        for layer in self.config["layers"]:
            layerImgObj = []
            for f_path in os.listdir(layer["directory"]):
                if any([ f_path.endswith(ex) for ex in PyArtGenerator.IMAGE_EXTENSION]):
                    data = cv2.imread(os.path.join(layer["directory"], f_path))
                    #create alpha channel
                    if data.shape[2] == 3:
                        b_channel, g_channel, r_channel = cv2.split(data)
                        alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
                        data = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

                    layerImgObj.append(data)
            self.imgObjs.append(layerImgObj)
        self.imgIndex = [0] * len(self.imgObjs)
    
    def getNext(self):
        carry = self.imgIndex[-1] == len(self.imgObjs[-1])
        i = len(self.imgIndex) - 1
        while carry:
            self.imgIndex[i] = 0
            i = i - 1
            if i == -1:
                return None
            self.imgIndex[i] += 1
            carry = self.imgIndex[i] == len(self.imgObjs[i])

        res = []
        for i in range(len(self.imgIndex)):
            res.append(self.imgObjs[i][self.imgIndex[i]])
        self.imgIndex[-1] += 1
        return res

    def create_output_dir(self):
        if not os.path.isdir(PyArtGenerator.OUTPUT_DIR):
            os.mkdir(PyArtGenerator.OUTPUT_DIR)

    def start(self):
        imageWidth = self.config["imageSetup"]["width"]
        imageHeight = self.config["imageSetup"]["height"]
        self.create_output_dir()
        all_images = self.loadImages()
        
        i = 0
        
        curr_images = self.getNext()
        while curr_images != None:

            img = self.createImage(imageWidth, imageHeight)
            for j, layer in enumerate(self.config["layers"]):
                layer_img = curr_images[j]
                x, y = 0, 0
                if "scaleMode" in layer and layer["scaleMode"] == "full":
                    layer_img = cv2.resize(layer_img, (imageHeight, imageWidth,))
                else:
                    layer_img = cv2.resize(layer_img, (layer["height"], layer["width"],))
                    x, y = layer["left"], layer["top"]
                
                w, h = layer_img.shape[1], layer_img.shape[0]
                print(f"printing layer: {i} image: {j}, ({w}, {h}, {x}, {y})")
                print(layer_img.shape, self.img.shape)
                self.img[y:y+h, x:x+w,0:] = layer_img
            #cv2.imshow("final", self.img)
            self.saveImage(f"{i}_{j}")
            curr_images = self.getNext()
            i += 1

        print("all image generations are completed.")


def run():
    with open("config.json") as f:
        config = json.load(f)
        generator = PyArtGenerator(config)
        generator.start()

if __name__ == "__main__":
    run()