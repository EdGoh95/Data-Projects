#!/usr/bin/env python3
"""
Machine Learning Infrastructure And Best Practices For Software Engineers (Packt Publishing)
Chapter 12: Designing Machine Learning Pipelines (MLOps) and Their Testing
"""
from transformers import pipeline
from termcolor import colored

#%% Raw Data-Based ML Pipeline
####  NLP-Related Tasks
print(colored('Sentiment Analysis', 'magenta', attrs = ['bold']))
text_classifier = pipeline('text-classification')
result = text_classifier("This movie is amazing and highly recommended!")
print(result)

print(colored('Text Generation', 'red', attrs = ['bold']))
text_generator = pipeline("text-generation")
generated_result = text_generator("In a galaxy far, far away...", max_length = 50, num_return_sequences = 3)
for generated_output in generated_result:
    print(generated_output['generated_text'])

print(colored('Text Summarization', 'yellow', attrs = ['bold']))
summarizer = pipeline('summarization')
article = """
In a groundbreaking discovery, scientists have found a new species of dinosaur in South America. The
dinosaur, named "Titanus maximus", is estimated to have been the largest terrestial creature to ever
walk the Earth. It belongs to the sauropod group of dinosaurs, known for their long necks and tails.
The discovery sheds new light on the diversity of dinosaurs that once inhabited our planet.
"""
summarized_result = summarizer(article, max_length = 100, min_length = 30, do_sample = False)
print(summarized_result[0]['summary_text'])

#### Image Processing Tasks
image_url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"
print(colored('\nImage Classification', 'blue', attrs = ['bold']))
image_classifier = pipeline(model = "google/vit-large-patch16-224")
print(image_classifier(image_url))

print(colored('Image Segmentation', 'green', attrs = ['bold']))
segmenter = pipeline(model = "facebook/detr-resnet-50-panoptic")
segments = segmenter(image_url)
print('Segment Label:', segments[0]['label'])

print(colored('Object Detection', 'grey', attrs = ['bold']))
object_detector = pipeline(model = 'facebook/detr-resnet-50')
print(object_detector(image_url))