# Filmgrainer - by Lars Ole Pontoppidan - MIT License

from PIL import Image, ImageFilter
import os
import tempfile
import numpy as np

import filmgrainer.graingamma as graingamma
import filmgrainer.graingen as graingen


def _grainTypes(typ):
    # After rescaling to make different grain sizes, the standard deviation
    # of the pixel values change. The following values of grain size and power
    # have been imperically chosen to end up with approx the same standard 
    # deviation in the result:
    if typ == 1:
        return (0.8, 63) # more interesting fine grain
    elif typ == 2:
        return (1, 45) # basic fine grain
    elif typ == 3:
        return (1.5, 50) # coarse grain
    elif typ == 4:
        return (1.6666, 50) # coarser grain
    else:
        raise ValueError("Unknown grain type: " + str(typ))

# Grain mask cache
MASK_CACHE_PATH = os.path.join(tempfile.gettempdir(), "mask-cache")

def _getGrainMask(img_width:int, img_height:int, saturation:float, grayscale:bool, grain_size:float, grain_gauss:float, seed):
    sat = -1.0 if grayscale else saturation
    filename = os.path.join(MASK_CACHE_PATH, f"grain-{img_width}-{img_height}-{saturation}-{grain_size}-{grain_gauss}-{seed}.png")
    
    if not os.path.isfile(filename):
        os.makedirs(MASK_CACHE_PATH, exist_ok=True)
        mask = graingen.grainGen(img_width, img_height, grain_size, grain_gauss, sat, seed)
        print(f"Saving: {filename}")
        mask.save(filename, format="png", compress_level=1)
    else:
        print(f"Reusing: {filename}")
        mask = Image.open(filename)
    
    return mask

def process(image, scale:float, src_gamma:float, grain_power:float, shadows:float,
            highs:float, grain_type:int, grain_sat:float, gray_scale:bool, sharpen:int, seed:int):
    
    image = np.clip(image, 0, 1) # Ensure the values are within [0, 1]
    image = (image * 255).astype(np.uint8)
    img = Image.fromarray(image).convert("RGB")
    org_width, org_height = img.size
    
    if scale != 1.0:
        print("Scaling source image ...")
        img = img.resize((int(org_width / scale), int(org_height / scale)), resample=Image.LANCZOS)
    
    img_width, img_height = img.size
    print(f"Size: {img_width} x {img_height}")
    map = graingamma.Map.calculate(src_gamma, grain_power, shadows, highs)
    grain_size, grain_gauss = _grainTypes(grain_type)
    mask = _getGrainMask(img_width, img_height, grain_sat, gray_scale, grain_size, grain_gauss, seed)

    lookup = map.map
    img_pixels = np.array(img)
    mask_pixels = np.array(mask)

    if gray_scale:
        print("Film graining image ... (grayscale)")
        gray_img = np.dot(img_pixels[...,:3], [0.21, 0.72, 0.07])
        img_pixels = lookup[gray_img.astype(int), mask_pixels.astype(int)][..., np.newaxis]
    else:
        print("Film graining image ...")
        img_pixels = lookup[img_pixels.astype(int), mask_pixels.astype(int)]

    img = Image.fromarray(img_pixels.astype('uint8'), 'RGB')
    
    if scale != 1.0:
        print("Scaling image back to original size ...")
        img = img.resize((org_width, org_height), resample=Image.LANCZOS)
    
    if sharpen > 0:
        print(f"Sharpening image: {sharpen} pass ...")
        for _ in range(sharpen):
            img = img.filter(ImageFilter.SHARPEN)

    return np.array(img).astype('float32') / 255.0
