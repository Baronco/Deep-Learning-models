import tensorflow as tf
from keras.models import model_from_json
import requests
import gradio as gr
import numpy as np
import PIL.Image as Image
from pathlib import Path
from glob import glob
import cv2


dataset = str(
    Path(
        r'D:\ML Datasets\kagglecatsanddogs_5340\Color'
    )
)
model_architecture = str(
    Path(
        r'C:\Users\jean-\OneDrive\Documentos\DSProject\Cats And Dogs Classifier\Models\Model_Edited_128_128.json'
    )
)

model_weights = str(
    Path(
        r'C:\Users\jean-\OneDrive\Documentos\DSProject\Cats And Dogs Classifier\Models\Model_Edited_128_128.h5'
    )
)

RESCALE_IMG_WIDTH, RESCALE_IMG_HEIGHT = 128, 128
SIZE = 128


def resize_image_pil(img, new_width, new_height):

    # Convert to PIL image
    img = Image.fromarray(img)
    
    # Get original size
    width, height = img.size

    # Calculate scale
    width_scale = new_width / width
    height_scale = new_height / height 
    scale = min(width_scale, height_scale)

    # Resize
    resized = img.resize((int(width*scale), int(height*scale)), Image.NEAREST)
    
    # Crop to exact size
    resized = resized.crop((0, 0, new_width, new_height))

    return resized


def transform_img(img):
    img_blur = cv2.GaussianBlur(img, (3 ,3), 0, 0)
    kernel = np.array([[-3,-1,-1], [-1,9,-1], [-1,-1,1]])
    im = cv2.filter2D(img_blur, -1, kernel)
    img_inv = cv2.bitwise_not(im)

    return img_inv

def classify_image(inp):
    label = ['Dog','Cat']
    # resize img
    #img_resized =  img_crop_resize(inp, nominal_size = SIZE)
    inp2 = transform_img(inp)
    img_resized = resize_image_pil(inp2,RESCALE_IMG_WIDTH, RESCALE_IMG_HEIGHT) 
    img = np.array(img_resized)
    img = img * (1/255)
    reshaped_array = img.reshape((-1, RESCALE_IMG_WIDTH, RESCALE_IMG_HEIGHT , 3))

    predictions = model.predict(x=reshaped_array).flatten()

    confidences = {label[i]: float(predictions[i]) for i in range(2)}

    img_resized = transform_img(inp)
    img_resized = Image.fromarray(img_resized)
    img_resized = img_resized.resize(size = (512,512), resample = Image.NEAREST)

    return [confidences, img_resized]


# load json and create model
json_file = open(model_architecture, 'r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(model_weights)

np.random.seed(123)
random_imgs = np.random.choice(
    glob(
        str(
            Path(
                dataset,
                '**',
                '*.jpg'
            )
        )
    ),
    size=128,
    replace=False,
)

demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(width=256, height=256),
    outputs=[gr.Label(num_top_classes=2), gr.Image(width=256, height=256)],
    examples=[img_path for img_path in random_imgs],
    examples_per_page=64,
    title= 'Cat & Dog classifier'
)
demo.launch(inbrowser=True)
