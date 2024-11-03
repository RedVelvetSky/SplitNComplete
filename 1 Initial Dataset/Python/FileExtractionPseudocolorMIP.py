import os
import numpy as np
import tifffile
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed

# Constants
TIFF_FILE_PATH = r"D:\Cilia test\RPE1_FHmNG_MmKatnal2i1_aCEP164_A555_aARL13B_CY5_DAPI\RPE1_FHmNG_MmKatnal2i1_aCEP164_A555_aARL13B_CY5_DAPI.ome.tiff"
OUTPUT_BASE_DIR = os.path.dirname(TIFF_FILE_PATH)
MAX_INTENSITY_PROJECTION_ENABLED = True
PSEUDOCOLOR_ENABLED = True

def max_intensity_projection(images):
    """
    Create a maximum intensity projection from a stack of images.
    :param images: 3D numpy array (num_images, height, width).
    :return: 2D numpy array representing the MIP.
    """
    return np.max(images, axis=0)

def apply_pseudocolor(image_path):
    """
    Apply pseudocolor to a grayscale image, keeping the background black.
    :param image_path: Path to the grayscale image.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Unable to load image at {image_path}.")
        return

    # Check the data type
    print(f"Pseudocolor - Image dtype before normalization: {img.dtype}")
    print(f"Pseudocolor - Image min: {img.min()}, max: {img.max()}")

    # If image is not 8-bit, normalize it
    if img.dtype != np.uint8:
        img = normalize_image(img)
        print(f"Pseudocolor - Image dtype after normalization: {img.dtype}")
        print(f"Pseudocolor - Image min after normalization: {img.min()}, max after normalization: {img.max()}")

    # Apply threshold to create mask
    _, mask = cv2.threshold(img, 2, 255, cv2.THRESH_BINARY)

    # Apply pseudocolor map
    pseudocolor_img = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)

    # Set background to black
    pseudocolor_img[mask == 0] = 0

    # Save pseudocolored image
    directory, filename = os.path.split(image_path)
    output_path = os.path.join(directory, f'pseudocolor_{filename}')
    cv2.imwrite(output_path, pseudocolor_img)
    print(f"Pseudocolor image saved at: {output_path}")

def normalize_image(img):
    """
    Normalize image to 0-255 and convert to uint8.
    """
    img_min = img.min()
    img_max = img.max()
    if img_max == img_min:
        return np.zeros_like(img, dtype=np.uint8)
    normalized = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    return normalized

def process_channel(series_idx, channel, num_images, image_arrays, output_dir):
    """
    Process a single channel: save individual images, create MIP, and apply pseudocolor.
    :param series_idx: Index of the series.
    :param channel: Channel number.
    :param num_images: Number of Z-slices.
    :param image_arrays: 3D numpy array (num_images, height, width) for the channel.
    :param output_dir: Directory to save the output images.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Debug: Check image array properties
    print(f"Processing Series {series_idx + 1}, Channel {channel + 1}")
    print(f"Image array dtype: {image_arrays.dtype}, min: {image_arrays.min()}, max: {image_arrays.max()}")

    # Determine if normalization is needed
    need_normalization = False
    if image_arrays.dtype != np.uint8:
        need_normalization = True
        image_min = image_arrays.min()
        image_max = image_arrays.max()
        print(f"Channel {channel + 1} - Original dtype: {image_arrays.dtype}, min: {image_min}, max: {image_max}")

    # Save all images
    for z in range(num_images):
        img = image_arrays[z]

        if need_normalization:
            img_to_save = normalize_image(img)
        else:
            img_to_save = img

        image_path = os.path.join(output_dir, f'image_{z + 1}.png')

        # Save image using cv2.imwrite
        success = cv2.imwrite(image_path, img_to_save)
        if not success:
            print(f"Failed to save image at {image_path}")

    print(f"All images for channel {channel + 1} in series {series_idx + 1} saved to {output_dir}.")

    if MAX_INTENSITY_PROJECTION_ENABLED:
        mip_image = max_intensity_projection(image_arrays)
        print(f"MIP - dtype before normalization: {mip_image.dtype}, min: {mip_image.min()}, max: {mip_image.max()}")

        # Determine if normalization is needed
        if mip_image.dtype != np.uint8:
            mip_image_normalized = normalize_image(mip_image)
            print(f"MIP - dtype after normalization: {mip_image_normalized.dtype}, min: {mip_image_normalized.min()}, max: {mip_image_normalized.max()}")
        else:
            mip_image_normalized = mip_image

        mip_image_path = os.path.join(output_dir, 'max_intensity_projection.png')
        success = cv2.imwrite(mip_image_path, mip_image_normalized)
        if not success:
            print(f"Failed to save MIP image at {mip_image_path}")
        else:
            print(f"Max intensity projection saved for channel {channel + 1} in series {series_idx + 1} at {mip_image_path}.")

            if PSEUDOCOLOR_ENABLED:
                apply_pseudocolor(mip_image_path)

def main():
    with tifffile.TiffFile(TIFF_FILE_PATH) as tif:
        series_count = len(tif.series)
        print(f"Total series to process: {series_count}")

        # Prepare all tasks for parallel processing
        tasks = []
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            for series_idx, series in enumerate(tif.series):
                print(f"Processing series {series_idx + 1} of {series_count}")

                # Read all image data for the current series at once
                image_data = series.asarray()  # Shape: (Z, C, Y, X)
                print(f"Series {series_idx + 1} shape: {image_data.shape}, dtype: {image_data.dtype}")

                num_z, num_channels, height, width = image_data.shape  # Assuming shape order is (Z, C, Y, X)

                for channel in range(num_channels):
                    channel_dir = os.path.join(OUTPUT_BASE_DIR, f'series_{series_idx + 1}', f'channel_{channel + 1}')

                    # Extract all images for this channel
                    channel_images = image_data[:, channel, :, :]  # Shape: (Z, Y, X)

                    # Submit processing task
                    tasks.append(executor.submit(
                        process_channel,
                        series_idx,
                        channel,
                        num_z,
                        channel_images,
                        channel_dir
                    ))

            # Optionally, monitor task completion
            for future in as_completed(tasks):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing a channel: {e}")

    print("Processing complete.")

if __name__ == "__main__":
    main()