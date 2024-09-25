from transformers import AutoImageProcessor, TFResNetModel
from datasets import load_dataset
import numpy as np
import tensorflow

num_classes = 10
dataset = load_dataset("mnist")
image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = TFResNetModel.from_pretrained("microsoft/resnet-50")
samples = dataset["test"]
classifier = tensorflow.keras.layers.Dense(num_classes)
total = 0
true = 0
for sample in samples:
    total += 1
    image = sample['image']
    label = sample['label']
    inputs = image_processor(image.resize((224, 224)).convert('RGB'), return_tensors="tf")
    outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state
    logits = classifier(hidden_states)
    predictions = np.argmax(logits[:,0]).item()
    if predictions == label:
        true += 1
accuracy = true/total
print(accuracy)
