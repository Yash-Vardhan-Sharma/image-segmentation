import cv2
import torch
from PIL import Image
from torchvision import transforms
import numpy as np

def load_model():
  model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', weights=True)
  model.eval()
  return model


def mask_background(pic, mask):
  # Convert the image and mask to numpy arrays if they are not already
  pic = np.array(pic).astype('uint8')
  mask = np.array(mask).astype('uint8')
  b,g,r = cv2.split(pic)
  # Merge the channels with the alpha channel as mask
  rgba_image = cv2.merge([b,g,r,mask])
  
  return rgba_image

def remove_background(model, input_file):
  input_image = Image.open(input_file)
  preprocess = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  input_tensor = preprocess(input_image)
  input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

  # move the input and model to GPU for speed if available
  if torch.cuda.is_available():
      input_batch = input_batch.to('cuda')
      model.to('cuda')

  with torch.no_grad():
      output = model(input_batch)['out'][0]
  output_predictions = output.argmax(0)

  # create a binary (black and white) mask of the profile foreground
  mask = output_predictions.byte().cpu().numpy()
  background = np.zeros(mask.shape)
  bin_mask = np.where(mask, 255, background).astype(np.uint8)

  foreground = mask_background(input_image ,bin_mask)

  return foreground, bin_mask



def merge_image(background_file, foreground):
  # Load the foreground image from the numpy array
  final_foreground = Image.fromarray(foreground)
  
  # Load the background image
  background = Image.open(background_file)
  
  # Resize the background to match the size of the foreground
  background = background.resize(final_foreground.size, Image.LANCZOS)
  
  # Create a new image with the same size as the foreground
  final_image = Image.new("RGBA", final_foreground.size)
  
  # Paste the background into the new image
  final_image.paste(background, (0, 0))
  
  # Create a mask from the alpha channel of the foreground image
  mask = final_foreground.split()[-1]
  
  # Paste the foreground on top of the background, using the mask to handle transparency
  final_image.paste(final_foreground, (0, 0), mask)
  
  return final_image
