import tensorflow as tf


def path_from_type(ds_type):
    if ds_type == 'training':
        return './data/training/'
    else:
        return './data/test/'

def images_ds_processor(image, data_path, new_shape):
    image = tf.io.read_file(data_path + image)
    image = tf.image.decode_jpeg(image, channels=new_shape[2])
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    shape = tf.shape(image)
    image = tf.image.resize(image, size=(new_shape[0], new_shape[1]))
    return image, shape

def key_pts_processor(key_pts, shape, new_shape):
    key_pts = key_pts * tf.convert_to_tensor([new_shape[1]/shape[1], new_shape[0]/shape[0]])
    key_pts = tf.reshape(key_pts, shape=(1, -1))
    key_pts = key_pts / new_shape[0]
    return key_pts

def process_ds(ds_type, new_shape):
    def processor(image_name, key_pts):
        data_path = path_from_type(ds_type)
        image, shape = images_ds_processor(image_name, data_path, new_shape)
        key_pts_p = key_pts_processor(key_pts, shape, new_shape)
        return image, key_pts_p

    return processor

def image_list_ds(key_pts_frame):
    image_list = key_pts_frame.iloc[:, 0].to_numpy()
    ds = tf.data.Dataset.from_tensor_slices(image_list)
    return ds

def key_pts_ds(key_pts_frame):
    first_col = list(key_pts_frame.columns)[0]

    only_pts = key_pts_frame.drop(first_col, axis=1)
    only_pts_np = only_pts.to_numpy()
    only_pts_reshaped = only_pts_np.reshape(key_pts_frame.shape[0], -1, 2)

    return tf.data.Dataset.from_tensor_slices(only_pts_reshaped)

def get_ds(key_pts_frame):
    image_ds = image_list_ds(key_pts_frame)
    keys_ds = key_pts_ds(key_pts_frame)
    return tf.data.Dataset.zip((image_ds, keys_ds))
