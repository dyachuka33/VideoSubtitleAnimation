import numpy as np
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from PIL import Image

def apply_motion_blur(img, max_horiz_blur_strength=15.0, max_vert_blur_strength=4.0):

    print("MOTION BLUR\n")
    print("hb",max_horiz_blur_strength,"vb",max_vert_blur_strength)
    
    if max_horiz_blur_strength < 1.0 and max_vert_blur_strength < 1.0 : return img
    
    # Ensure the image has an alpha channel
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    img_np = np.array(img)

    h, w, _ = img_np.shape
    blurred_img_np = np.zeros_like(img_np)

    # Separate RGB and alpha channels
    rgb_img_np = img_np[:, :, :3]
    alpha_channel = img_np[:, :, 3]

    # Apply blur to RGB channels
    if max_horiz_blur_strength != 0 :
        for y in range(h):
            row = rgb_img_np[y, :, :]
            blurred_row = gaussian_filter1d(row, sigma=max_horiz_blur_strength, axis=0)

            for x in range(w):
                weight = max(x / w, (w - x) / w)
                blurred_img_np[y, x, :3] = (1 - weight) * row[x, :] + weight * blurred_row[x, :]

    if max_vert_blur_strength != 0 :
        for x in range(w):
            blur_strength = max(1, int(max_vert_blur_strength * max(x / w, (w - x) / w)))
            column_blur = gaussian_filter(rgb_img_np[:, x, :], sigma=(blur_strength, 0))
            blurred_img_np[:, x, :3] = column_blur

    # Re-attach the alpha channel
    blurred_img_np[:, :, 3] = alpha_channel

    # Convert back to PIL image and return
    blurred_image = Image.fromarray(blurred_img_np, 'RGBA')
    return blurred_image

    # blurred_image_path = image_path.rsplit('.', 1)[0] + '_horizontal_motion_blur.png'
    # blurred_image.save(blurred_image_path)

    # return blurred_image_path


import numpy as np
from scipy.ndimage import gaussian_filter

def apply_motion_blur_optimized(img, max_horiz_blur_strength=15, max_vert_blur_strength=4):
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    
    img_np = np.array(img)

    # Separate RGB and alpha channels
    rgb_img_np = img_np[:, :, :3]
    alpha_channel = img_np[:, :, 3]

    # Apply horizontal blur vectorized
    weights_horizontal = np.linspace(0, 1, img_np.shape[1])[None, :, None]
    horizontal_blur = gaussian_filter(rgb_img_np, sigma=(0, max_horiz_blur_strength, 0))
    rgb_img_np = rgb_img_np * (1 - weights_horizontal) + horizontal_blur * weights_horizontal


    # Apply vertical blur vectorized
    weights_vertical = np.linspace(0, 1, img_np.shape[0])[:, None, None]
    vertical_blur = gaussian_filter(rgb_img_np, sigma=(max_vert_blur_strength, 0, 0))
    rgb_img_np = rgb_img_np * (1 - weights_vertical) + vertical_blur * weights_vertical


    # Re-attach the alpha channel
    img_np[:, :, :3] = rgb_img_np
    img_np[:, :, 3] = alpha_channel

    # Convert back to PIL image and return
    return Image.fromarray(img_np, 'RGBA')