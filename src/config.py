DATA_DIR = "../data/processed"
SEG_TYPE = 'binary' # binary | multiclass
EPOCHS = 100
TRAIN_IMAGES = '../data/processed/Train/image'
TRAIN_MASKS = '../data/processed/Train/masks'
TEST_IMAGES = '../data/processed/Test/image'
TEST_MASKS = '../data/processed/Test/masks'
LOG_DIR = "../logs"
WEIGHTS_DIR = "../weights"
INFERENCE_DIR = "../inference"
DATASET = "consep" # consep | kumar | cpm

MULTICLASS_COLOR_VALUES = {
                    0: [0, 0, 0],
                    1: [255, 0, 0],
                    2: [255, 215, 0],
                    3: [0, 128, 0],
                    4: [0, 0, 255],
                    5: [139, 69, 19],
                    6: [47, 79, 79],
                    7: [25, 25, 112]
                }
BINARY_COLOR_VALUES = {
                    0: [0, 0, 0],
                    1: [255, 255, 255]
                }

LEARNING_RATE = 0.0004
IMAGE_SIZE = (256, 256)
MULTICLASS_NUM_CLASSES = 8
BINARYCLASS_NUM_CLASSES = 2
BATCH_SIZE = 4