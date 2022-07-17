from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
	'celeba_encode_minimized': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['celeba_train_minimized'],
		'train_target_root': dataset_paths['celeba_train_minimized'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test'],
	},
}
