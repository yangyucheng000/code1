"""Generate testing output."""

import argparse
import pathlib

# import imageio
import imageio.v2 as imageio
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from util import plugin


def _parse_argument():
    """Return arguments for conversion."""
    parser = argparse.ArgumentParser(description='Testing.')
    parser.add_argument('--model_path', help='Path of model file.', type=str, required=True)
    parser.add_argument('--model_name', help='Name of model class.', type=str, required=True)
    parser.add_argument('--ckpt_path', help='Path of checkpoint.', type=str, required=True)
    parser.add_argument(
        '--data_dir', help='Directory of testing frames in REDS dataset.', type=str, required=True
    )
    parser.add_argument(
        '--output_dir', help='Directory for saving output images.', type=str, required=True
    )

    args = parser.parse_args()

    return args


def main(args):
    """Run main function for converting keras model to tflite.

    Args:
        args: A `dict` contain augments.
    """
    # prepare dataset
    data_dir = pathlib.Path(args.data_dir)

    # prepare model
    model_builder = plugin.plugin_from_file(args.model_path, args.model_name, tf.keras.Model)
    model = model_builder()

    # load checkpoint
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(args.ckpt_path).expect_partial()

    save_path = pathlib.Path(args.output_dir)
    save_path.mkdir(exist_ok=True)

    # testing
    for i in range(30):
        for j in tqdm(range(100)):
            if j == 0:
                input_image_1 = np.expand_dims(
                    imageio.imread(data_dir / str(i).zfill(3) / f'{str(j).zfill(8)}.png'), axis=0
                ).astype(np.float32)
                input_image_2 = np.expand_dims(
                    imageio.imread(data_dir / str(i).zfill(3) / f'{str(j+1).zfill(8)}.png'), axis=0
                ).astype(np.float32)
                b, h, w, _ = input_image_1.shape
                input_tensor = tf.concat([input_image_1, input_image_1, input_image_2], axis=-1)
                
                pred_tensor = model(input_tensor, training=False)
            elif j == 99:
                input_image_1 = np.expand_dims(
                    imageio.imread(data_dir / str(i).zfill(3) / f'{str(j-1).zfill(8)}.png'), axis=0
                ).astype(np.float32)
                input_image_2 = np.expand_dims(
                    imageio.imread(data_dir / str(i).zfill(3) / f'{str(j).zfill(8)}.png'), axis=0
                ).astype(np.float32)
                b, h, w, _ = input_image_1.shape
                input_tensor = tf.concat([input_image_1, input_image_2, input_image_2], axis=-1)
                pred_tensor = model(input_tensor, training=False)
            else:
                input_image_1 = np.expand_dims(
                    imageio.imread(data_dir / str(i).zfill(3) / f'{str(j-1).zfill(8)}.png'), axis=0
                ).astype(np.float32)
                input_image_2 = np.expand_dims(
                    imageio.imread(data_dir / str(i).zfill(3) / f'{str(j).zfill(8)}.png'), axis=0
                ).astype(np.float32)
                input_image_3 = np.expand_dims(
                    imageio.imread(data_dir / str(i).zfill(3) / f'{str(j+1).zfill(8)}.png'), axis=0
                ).astype(np.float32)
                input_tensor = tf.concat([input_image_1, input_image_2, input_image_3], axis=-1)
                pred_tensor = model(input_tensor, training=False)

            # if (j+1) % 10 == 0:
            imageio.imwrite(save_path / f'{str(i).zfill(3)}_{str(j).zfill(8)}.png', pred_tensor[0].numpy().astype(np.uint8))


if __name__ == '__main__':
    arguments = _parse_argument()
    main(arguments)