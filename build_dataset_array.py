import numpy
import numpy as np
from PIL import Image
from glob import glob
import random
from tqdm import tqdm

digit_array = np.ndarray((10, 5000, 784), dtype=np.float16)

for digit in tqdm(range(10)):
	digit_files = glob("mnist/training/"+str(digit)+"/*.jpg")
	random.shuffle(digit_files)
	digit_files = digit_files[:5000]
	for i in tqdm(range(5000)):
		img_arr = np.asarray(Image.open(digit_files[i])).reshape((784))
		digit_array[digit][i] = img_arr / 255

print(digit_array)
numpy.save("mnist-5000-normalized", digit_array)