from PIL import Image
import os

src_dir = os.getcwd() + '/ChestXRay_images'
for (root, dirs, files) in os.walk(src_dir):
    for filename in files:
        print(os.path.join(root,filename))
        image_path = os.path.join(root, filename)
        with Image.open(image_path) as im:
            reduction_factor = 8
            im = im.resize((im.width // reduction_factor, im.height // reduction_factor))
            im.save('Resized_Images/' + filename.split('.')[0] + '.png')



