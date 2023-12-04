#!/usr/bin/env python3
"""
Deep Learning With TensorFlow and Keras Third Edition (Packt Publishing)
Chapter 10: Self-Supervised Learning
"""
import torch
import requests
from transformers import BertTokenizer, BertForMaskedLM, CLIPProcessor, CLIPModel
from PIL import Image

#%% Bidirectional Encoder Representation from Transformers (BERT)
BERT_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
BERT_model = BertForMaskedLM.from_pretrained('bert-base-cased')

inputs = BERT_tokenizer('The capital of France is [MASK].', return_tensors = 'pt')
logits = BERT_model(**inputs).logits
masked_token_index = torch.where(inputs.input_ids == BERT_tokenizer.mask_token_id)[1]
predicted_token_id = torch.argmax(logits[:, masked_token_index], axis = -1)
print('Missing Word (Predicted):', BERT_tokenizer.convert_ids_to_tokens(predicted_token_id)[0])

#%% Contrastive Language Image Pre-training (CLIP)
CLIP_model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
CLIP_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

sample_image = Image.open(requests.get('http://images.cocodataset.org/val2017/000000039769.jpg', stream = True).raw)
CLIP_inputs = CLIP_processor(text = ['a photo of a cat', 'a photo of a dog'], images = sample_image, return_tensors = 'pt', padding = True)
CLIP_logits_per_image = CLIP_model(**CLIP_inputs).logits_per_image
CLIP_image_probabilities = CLIP_logits_per_image.softmax(dim = 1)
print('Probabilities:', CLIP_image_probabilities.detach().numpy())