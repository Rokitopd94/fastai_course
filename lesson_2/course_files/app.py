# this is the app to deploy on a webserver or cloud platformm (I will try HuggingFace Spaces)

import gradio as gr
from fastai.vision.all import *

learn = load_learner('export.pkl')

labels = learn.dls.vocab # these are the labels used when training the model (the categories defined when creating the DataLoaders)

# function that uses the model to infere the type of bear in an image
def predict(img):
    img = PILImage.create(img)
    img.resize((128,128))
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))} # returns the probability for each type of bear

title = "Bears Classifier"
description = "This classifies a bear from a picture of it."
enable_queue = True

gr.Interface(fn=predict,
             inputs=gr.components.Image(),
             outputs=gr.components.Label(num_top_classes=3),
             title=title,
             description=description).launch(share=True)