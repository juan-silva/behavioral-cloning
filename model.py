
from random import shuffle
import csv
import cv2
import numpy as np
import mymodels
import sklearn
from sklearn.model_selection import train_test_split

# Load the data file into an array
lines = []
runs = [1, 2, 4, 5]
for run in runs:
	with open('data_runs/run' + str(run) + '/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			lines.append(line)

# Split train and validation data
train_samples, validation_samples = train_test_split(lines, test_size=0.2)


def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1:  # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			lines = samples[offset:offset + batch_size]
			images = []
			angles = []
			correction = 0.2
			for line in lines:
				# Use central and side cameras
				for i in range(3):
					# Append the image data
					path = line[i]
					filename = path.split('/')[-1]
					run = path.split('/')[-3]
					image = cv2.imread('data_runs/' + run + '/IMG/' + filename)
					images.append(image)

					# Append the angle
					angle = float(line[3])
					if(i == 1):
						angle = angle + correction
					if(i == 2):
						angle = angle - correction
					angles.append(angle - correction)

					# Data Augmentation
					images.append(cv2.flip(image, 1))
					angles.append(angle * -1.0)

			# Return the shuffled numpy arrays 
			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)


# Create the generators
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Create a model
model = mymodels.myNvidia()
# Compile, train ans save it
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, 
					samples_per_epoch=len(train_samples), 
					validation_data=validation_generator, 
					nb_val_samples=len(validation_samples), 
					nb_epoch=3)
model.save('model_run1245.h5')
print("Model Saved.")
