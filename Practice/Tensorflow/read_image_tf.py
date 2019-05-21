def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_image(image_string)
  image_resized = tf.image.resize_images(image_decoded, [28, 28])
  return image_resized, label

# 图片文件的列表
filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg", ...])
# label[i]就是图片filenames[i]的label
labels = tf.constant([0, 37, ...])

# 此时dataset中的一个元素是(filename, label)
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

# 此时dataset中的一个元素是(image_resized, label)
dataset = dataset.map(_parse_function)

# 此时dataset中的一个元素是(image_resized_batch, label_batch)
dataset = dataset.shuffle(buffersize=1000).batch(32).repeat(10)