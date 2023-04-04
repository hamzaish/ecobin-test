import tensorflow as tf
import numpy as np
import json
from PIL import Image
from model import model_load

path = "model_ex-136_acc-0.923399.h5"
json_path = "model_class.json"
pred = model_load(path, json_path, "slow")

print(pred.classify('snapple.png'))