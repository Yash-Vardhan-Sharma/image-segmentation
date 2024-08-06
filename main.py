from model_utils import load_model, remove_background, merge_image
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--image_path',
                    '-i',
                    type = str,
                    help ='path of the image whose background is to be changed')
 
parser.add_argument('--background_path', 
                    '-b',
                    type = str, 
                    help ='path of the background to be applied')

parser.add_argument('--image_save_dir',
                    '-d',
                    type = str, 
                    default = './saved_images')


if __name__ == "__main__":


    args = parser.parse_args()
    image_path = args.image_path
    background_path = args.background_path
    image_save_dir = args.image_save_dir

    img_name = image_path.split('/')[-1].split('.')[0]
    bkg_name = background_path.split('/')[-1].split('.')[0]

    deeplab_model = load_model()
    foreground, bin_mask = remove_background(deeplab_model, image_path)

    final_image = merge_image(background_path, foreground)

    os.makedirs(image_save_dir, exist_ok=True)
    final_image.save(f'{image_save_dir}/updated_{img_name}_{bkg_name}.png')