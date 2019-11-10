import runway
import numpy as np
import tensorflow as tf
from PIL import Image
from inpaint_model import InpaintCAModel
import neuralgym as ng

g = None
sess = None
input_image = None

FLAGS = ng.Config('inpaint.yml')

@runway.setup(options={'checkpoint_dir': runway.file(is_directory=True)})
def setup(opts):
    global g
    global sess
    global input_image
    model = InpaintCAModel()
    g = tf.get_default_graph()
    sess = tf.Session(graph=g)
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    input_image = tf.placeholder(tf.float32, shape=(1, 256, 256*2, 3))
    output_image = model.build_server_graph(FLAGS, input_image)
    output_image = (output_image + 1.) * 127.5
    output_image = tf.reverse(output_image, [-1])
    output_image = tf.saturate_cast(output_image, tf.uint8)
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    for var in vars_list: 
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable(opts['checkpoint_dir'], from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)
    print('Model loaded.')
    return output_image

command_inputs = {
  'image': runway.image,
  'mask': runway.segmentation(label_to_id={'background': 0, 'mask': 1}, label_to_color={'background': [0,0,0], 'mask': [255, 255, 255]})
}

@runway.command('inpaint', inputs=command_inputs, outputs={'inpainted': runway.image})
def inpaint(output_image, inputs):
  image = inputs['image']
  original_size = image.size
  image = np.array(image.resize((256, 256)))
  image = np.expand_dims(image, 0)
  mask = np.array(inputs['mask'].resize((256, 256)))
  mask = mask * 255
  mask = np.stack((mask,)*3, axis=-1)
  mask = np.expand_dims(mask, 0)
  feed_dict = {input_image: np.concatenate([image, mask], axis=2)}
  with g.as_default():
    result = sess.run(output_image, feed_dict=feed_dict)
  return Image.fromarray(result[0][:, :, ::-1]).resize(original_size)


if __name__ == '__main__':
  runway.run(port=5232, model_options={'checkpoint_dir': './model_logs/release_places2_256'})