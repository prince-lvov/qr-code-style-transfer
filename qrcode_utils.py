import numpy as np
from skimage import color
from skimage.util import compare_images
from PIL import Image, ImageOps
from pyzbar.pyzbar import decode
import qrcode
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.colormasks import HorizontalGradiantColorMask
import torch


def get_qrcode(version, data):
    qr = qrcode.QRCode(
        version=version,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=1,
        border=0,
    )
    qr.add_data(data)
    qr.make(fit=False)
    return qr


def get_color_mask(style_name):
    color_mask = HorizontalGradiantColorMask()
    if style_name == "green_orange":
        color_mask.back_color = (255, 255, 255)
        color_mask.left_color = (21, 137, 6)
        color_mask.right_color = (247, 105, 9)
    elif style_name == "purple_violet":
        color_mask.back_color = (255, 255, 255)
        color_mask.left_color = (128, 0, 128)
        color_mask.right_color = (138, 43, 226)
    
    return color_mask


def generate_qrcode_image(version, data, color_mask=None):
    qr = get_qrcode(version, data)
    if color_mask:
        return qr.make_image(
        	image_factory=StyledPilImage, color_mask=color_mask
    	)
    return qr.make_image(fill_color="black", back_color="white")


def qrcode_path_to_np_array(qrcode_path, dtype=np.float32, grayscale=False):
	image = Image.open(qrcode_path)
	if grayscale:
		image = ImageOps.grayscale(image)
	image_array = np.asarray(image, dtype=dtype)
	return image_array


def np_to_ts_qrcode_array(qrcode_np_array):
    array = np.expand_dims(qrcode_np_array, axis=0)
    return np.expand_dims(array, axis=0)


def ts_to_np_qrcode_array(image_array):
    image_array = image_array.squeeze(0) * 255
    image_array = np.transpose(image_array, (1, 2, 0))
    return image_array.astype('uint8')


def np_qrcode_array_to_tensor(qrcode_np_array, device="cpu"):
    return torch.from_numpy(qrcode_np_array)\
        .unsqueeze(0).unsqueeze(0).to(device=device)
    

def tensor_to_np_qrcode_array(image_tensor):
    image_ts_qrcode_array = image_tensor.detach().numpy()
    return ts_to_np_qrcode_array(image_ts_qrcode_array)


def qrcode_array_to_image(image_array):
    return Image.fromarray(image_array, "RGB")


def is_valid_qrcode(image_np_array, k=10):
    image = qrcode_array_to_image(image_np_array)
    big_image = image.resize((image.size[0] * k, image.size[0] * k), resample=Image.Resampling.NEAREST)
    decode_result = decode(big_image)
    if not decode_result:
        return False
    return True


def get_grayscale_binary_loss(img_gcolor, grcode_gcolor):
	"""
	Calculate loss between input image and target QR Code image
	
	:param int img_gcolor: Input image pixel grayscale color
	:param int qrcode_gcolor: Target QR Code pixel grayscale color
	"""
	is_qrcode_white = 1 * (grcode_gcolor >= 123)
	is_qrcode_black = 1 * (grcode_gcolor < 123)

	threshold_white = 190 
	threshold_black = 100 

	is_img_white = 1 * (img_gcolor >= threshold_white)
	is_img_black = 1 * (img_gcolor < threshold_black)

	loss = is_qrcode_white * (0 if is_img_white else 1) * (threshold_white - img_gcolor) + \
				 is_qrcode_black * (0 if is_img_black else 1) * (img_gcolor - threshold_black)

	return loss


def get_corrected_rgbcolor_lab(img_rgbcolor, qrcode_gcolor):
	"""
	Shifts the input image RGB color inside LAB color space to minimize QR Code reader loss between input image and target QR Code image

    :param np.ndarray img_rgbcolor: Input image pixel color
	:param int qrcode_gcolor: Target QR Code pixel grayscale color
	"""
	img_gcolor = color.rgb2gray(img_rgbcolor / 255) * 255
	img_labcolor = color.rgb2lab(img_rgbcolor / 255)

	is_qrcode_white = 1 * (qrcode_gcolor >= 123)
	is_qrcode_black = 1 * (qrcode_gcolor < 123)
	l_correction_step = 2
	l_correction = l_correction_step * is_qrcode_white - l_correction_step * is_qrcode_black

	while (get_grayscale_binary_loss(img_gcolor, qrcode_gcolor) > 0):
		img_labcolor[0] += l_correction # correct Lightness only
		img_rgbcolor = color.lab2rgb(img_labcolor) * 255
		img_gcolor = color.rgb2gray(img_rgbcolor / 255) * 255

	return np.array(img_rgbcolor, dtype=np.uint8)


def get_corrected_qrcode_image(img_array, qrcode_array):
	"""
	Shifts the RGB pixel colors of input image to minimize QR Code reader loss between input image and target QR Code grayscale image

	:param np.ndarray img: Input RGB image in Numpy format [h,w,c]
	:param np.ndarray qrcode_img_gray: Target QR Code grayscale image
	"""
  
	corrected_image_array = np.array(img_array)

	for i in range(img_array.shape[0]):
		for j in range(img_array.shape[1]):		
			corrected_image_array[i,j] = get_corrected_rgbcolor_lab(img_array[i,j], qrcode_array[i,j])
  
	return corrected_image_array


def get_diff_between_image_arrays(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Return a mean value showing the differences between two images.

    :param np.ndarray image1: Image to compare
    :param np.ndarray image2: Image to compare
    """
    image_diff = compare_images(image1, image2, method='diff')

    return np.mean(image_diff)