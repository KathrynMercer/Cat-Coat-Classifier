import gradio as gr
from fastai.vision.all import *
from huggingface_hub import from_pretrained_fastai

learn = from_pretrained_fastai(repo_id = "KathrynMercer/CatPatternClassifier")

labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

gr.Interface(fn=predict, 
             inputs='image', 
             outputs='label',
             title = 'Cat Coat Pattern Classifier',
             description = 'A computer vision classifier that can tell you which of the major coat patterns your cat has! Extra fancy coat patterns to follow.',
             examples = [r'Domestic_cat_sleeping.JPG', r'calico cat.jpg', r'1200px-British_shorthair_cat-3113513.jpg'],
             interpretation='default').launch(share=True)
