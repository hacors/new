from scipy import io as sio


def read_annotations():
    data = sio.loadmat('Datasets/mall_dataset/mall_gt')
    count = data['count']
    position = data['frame'][0]
    return count, position


a, b = read_annotations()
'''

def map_pixels(img, image_key, annotations, size):
    """map annotations to density map.

    Arguments:
        img: ndarray, img.
        image_key: int, image_key.
        annotations: ndarray, annotations.
        size: resize size.

    Returns:
        pixels: ndarray, density map.
    """
    gaussian_kernel = 15
    h, w = img.shape[:-1]
    sh, sw = size / h, size / w
    pixels = np.zeros((size, size))

    for a in annotations[image_key][0][0][0]:
        x, y = int(a[0] * sw), int(a[1] * sh)
        if y >= size or x >= size:
            print("{},{} is out of range, skipping annotation for {}".format(x, y, image_key))
        else:
            pixels[y, x] += 1

    pixels = cv2.GaussianBlur(pixels, (gaussian_kernel, gaussian_kernel), 0)

    return pixels


def get_data(i, size, annotations):
    """get data accoding to the image_key.

    Arguments:
        i: int, image_key.
        size: int, input shape of network.
        annotations: ndarray, annotations.

    Returns:
        img: ndarray, img.
        density_map: ndarray, density map.
    """
    name = 'data\\mall_dataset\\frames\\seq_{}.jpg'.format(str(i + 1).zfill(6))
    img = cv2.imread(name)

    density_map = map_pixels(img, i, annotations, size // 4)

    img = cv2.resize(img, (size, size))
    img = img / 255.
    
    density_map = np.expand_dims(density_map, axis=-1)

    return img, density_map
    '''
