BLIP is Vision-Language Pretraining model mainly for image captioning . BLIP core Strength is it works well for image-to-text tasks, it’s lightweight compared to flamingo (easier to fine-tune on smaller hardware). This program is made to teach and use an image captioning model based on the BLIP model (it uses frozen LLM + lightweight Q-Former for vision-text alignment). BLIP model that takes an image and makes a meaningful description of it. We are using the Flickr30k dataset, which includes images and their corresponding descriptions. 
Simplified Architecture ; How BLIP works - 
a practical (few lines) solution using Hugging Face’s pretrained BLIP, and
a from-scratch, educational PyTorch “mini-BLIP” that mirrors the real design (image encoder + text encoder + cross-attention + 3 losses).
Components:
Image Encoder: Vision Transformer (ViT) - extracts visual features.
Text Encoder: BERT-like transformer –extracts text feature
Multimodal Encoder: Cross-attention between image and text features.
Training Objectives:
ITC (Image-Text Contrastive) – Aligns images with correct captions.
ITM (Image-Text Matching) – Binary classification for matching pairs.
LM (Language Modeling) – Caption generation using images.

Walk through by the whole process step by step:
Step 1: Installing and Importing Libraries
We first install and bring in the necessary tools.
Transformers help us get the BLIP model and processor.
Torch is used for training and improving the model.
Pandas helps manage the text data.
PIL is for opening and changing images.
 
Step 2: Preparing the Device
Next, we check if a GPU (CUDA) is available. If it is, we use it for faster training. If not, we use the CPU. We also use cuDNN features to make training on GPUs faster.
 
Step 3: Setting Up Paths
We give the path where our dataset is:
The folder that has all the images. 
The text file that has the names of the images and their captions.
 
Step 4: Loading the Dataset
We use Pandas to load the captions file into a DataFrame.
This has two parts:
image — the name of the image file.
caption — the description that goes with the image.
This makes it easy to match images with their captions.
 
 

Step 5: Creating a Custom Dataset Class
To train the model in PyTorch, we make a Dataset class.
This class:
Takes the DataFrame, image folder path, and BLIP processor as input. 
Has a method to:
Load the image from the computer.
Make it in RGB format.
Get the caption.
Use the BLIP processor to make both the image and text into forms the model can use.
 
The processor handles turning text into tokens and normalizing images, returning:
pixel_values — the image as a tensor.
input_ids — the tokens of the caption.
attention_mask — which tokens need attention (non-padding ones).
 
Step 6: Loading the BLIP Model and Processor
We then load:
BlipProcessor — which turns raw images and text into the right format for BLIP.
BlipForConditionalGeneration — the BLIP model that makes captions.
We move the model to our device (GPU or CPU).
 
Step 7: Creating the DataLoader
We wrap our custom dataset into a DataLoader, which:
Groups the data into batches (size 8 here).
Mixes the data for better training.
Uses multiple workers to speed up loading.
 
 

Step 8: Training Setup
To train the model, we:
Use the AdamW optimizer (good for transformer models) with a learning rate of 5e-5.
Enable mixed precision training with GradScaler and autocast to speed things up and save memory on GPUs.
Set epochs = 1 (this can be changed for better results).
In the training loop:
For each batch, we move the data to the GPU or CPU.
We get the model’s output and calculate the loss.
We adjust the model's weights by doing backpropagation, scaling the gradients, and updating the weights.
A progress bar (tqdn) shows the current epoch and loss.
 
Step 9: Saving the Model
After training, we save the model and processor for use later. This way, we don't have to train from scratch again.
 
Step 10: Creating Captions (Inference)
Finally, we test the model on a sample image:
We load the image and process it with the BLIP processor.
We use the model’s generate() method to create a caption.
We take the generated tokens and turn them into human-readable text.
We print the caption.
 
In short, this code trains the BLIP image captioning model on the Flickr30k dataset and then gives captions for new images.
