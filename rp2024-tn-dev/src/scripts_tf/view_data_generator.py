import hydra
import cv2
import numpy as np
from omegaconf import DictConfig
from hydra.utils import instantiate
from data_util import load_image_stats

@hydra.main(version_base=None, config_path="config", config_name="train_tn")
def main(cfg: DictConfig) -> None:
    data_generator = instantiate(cfg.data_generator)

    image_stats = load_image_stats(f'{cfg.dataset_directory}/scene_0000.hdf5')

    for i in range(len(data_generator)):
        inputs, targets = data_generator[i]
        for j in range(data_generator.batch_size):
            position_input, orientation_input = inputs[0][j], inputs[1][j]
            position_target, orientation_target = targets[0][j], targets[1][j]

            heightmap = position_input[..., 0]
            colormap = position_input[..., 1:]

            heightmap = heightmap * image_stats['std_heightmap'] + image_stats['mean_heightmap']
            heightmap = (heightmap - np.min(heightmap)) / (np.max(heightmap) - np.min(heightmap))
            colormap = colormap * image_stats['std_colormap'] + image_stats['mean_colormap']
            colormap = colormap.astype(np.uint8)
            cv2.imshow("heightmap", heightmap)
            cv2.imshow("colormap", colormap)
            cv2.imshow("position target", position_target)
            for o_img, o_tgt in zip(orientation_input, orientation_target):
                if o_tgt == 1:
                    o_img = o_img[..., 1:] * image_stats['std_colormap'] + image_stats['mean_colormap']
                    o_img = o_img.astype(np.uint8)
                    cv2.imshow("correct orientation input", o_img)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
        if key == ord('q'):
            break


if __name__ == "__main__":
    main()