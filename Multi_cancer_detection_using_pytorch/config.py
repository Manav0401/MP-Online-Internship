import torch

SOURCE_PATH = "OriginalDataset"
DESTINATION_PATH = "FlattenedDataset"

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

MODEL_PATH = "saved_models/cancer_cnn.pth"
HISTORY_PATH = "training_history.json"

TEST_IMAGE_DIR = "test_images"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")