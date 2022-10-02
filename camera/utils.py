import cv2
import urllib.request
import numpy as np
from PIL import Image
import base64


def cvt2RGB(image):
    img = image.copy()
    img = Image.fromarray(image)
    img = img.convert('RGB')
    return np.array(img)

def grab_image(path=None, stream=None, url=None):
    # if the path is not None, then load the image from disk
    if path is not None:
        image = cv2.imread(path)
    # otherwise, the image does not reside on disk
    else:	
        # if the URL is not None, then download the image
        if url is not None:
            resp = urllib.request.urlopen(url)
            data = resp.read()
        # if the stream is not None, then the image has been uploaded
        elif stream is not None:
            data = stream.read()
        # convert the image to a NumPy array and then read it into OpenCV format
        image = np.asarray(bytearray(data), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cvt2RGB(image)
    # return the image
    return image

def readb64(uri):
   encoded_data = uri.split(',')[-1]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img
