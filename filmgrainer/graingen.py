from PIL import Image
import numpy as np

def _makeGrayNoise(width, height, power):
    # Generate noise directly in the uint8 data type to avoid conversion
    noise = np.random.normal(128, power, (height, width)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(buffer)

def _makeRgbNoise(width, height, power, saturation):
    intens_power = power * (1.0 - saturation)
    intens = np.random.normal(128, intens_power, (height, width))

    noise = np.random.normal(0, power, (height, width, 3)) * saturation
    noise += intens[:, :, np.newaxis]

    buffer = np.clip(noise, 0, 255).astype(np.uint8)
    return Image.fromarray(buffer)

def grainGen(width, height, grain_size, power, saturation, seed=1):
    # A grain_size of 1 means the noise buffer will be made 1:1
    # A grain_size of 2 means the noise buffer will be resampled 1:2
    noise_width = int(width / grain_size)
    noise_height = int(height / grain_size)
    np.random.seed(seed) # Set the seed for reproducibility

    if saturation < 0.0:
        print(f"Making B/W grain, width: {noise_width}, height: {noise_height}, grain-size: {grain_size}, power: {power}, seed: {seed}")
        img = _makeGrayNoise(noise_width, noise_height, power)
    else:
        print(f"Making RGB grain, width: {noise_width}, height: {noise_height}, saturation: {saturation}, grain-size: {grain_size}, power: {power}, seed: {seed}")
        img = _makeRgbNoise(noise_width, noise_height, power, saturation)

    # Resample
    if grain_size != 1.0:
        img = img.resize((width, height), resample=Image.LANCZOS)

    return img


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 8:
        width = int(sys.argv[2])
        height = int(sys.argv[3])
        grain_size = float(sys.argv[4])
        power = float(sys.argv[5])
        sat = float(sys.argv[6])
        seed = int(sys.argv[7])
        out = grainGen(width, height, grain_size, power, sat, seed)
        out.save(sys.argv[1])
