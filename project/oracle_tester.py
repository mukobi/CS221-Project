import random
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
plt.ion()

PROJECT_FOLDER = os.path.dirname(__file__)
DATASET_FOLDER = PROJECT_FOLDER + '/data/columbia-prcg-datasets'
PHOTO_FOLDER = DATASET_FOLDER + '/google_images/'
CG_FOLDER = DATASET_FOLDER + '/prcg_images/'
NUM_IMAGES_PER_CLASS = 10


class classification:
    cg = 1
    photo = 0


def pick_random_images(folder, num_images):
    files = random.sample(os.listdir(folder), num_images)
    return [folder + file for file in files]


def get_user_class_from_image(image_path):
    img = mpimg.imread(image_path)
    # show image
    imgplot = plt.imshow(img)
    plt.show(block=False)
    # get user classification
    user_input = None
    while user_input != 'p' and user_input != 'c':
        user_input = input(
            'Enter a classification: (p)hoto or (c)omputer-generated')
    plt.close('all')
    return classification.photo if user_input == 'p' else classification.cg


def main():
    # load in images
    image_paths_with_labels = []
    for img in pick_random_images(PHOTO_FOLDER, NUM_IMAGES_PER_CLASS):
        image_paths_with_labels.append((img, classification.photo))
    for img in pick_random_images(CG_FOLDER, NUM_IMAGES_PER_CLASS):
        image_paths_with_labels.append((img, classification.cg))
    print("Loaded {} images.".format(len(image_paths_with_labels)))

    # randomize images
    random.shuffle(image_paths_with_labels)
    # print(image_paths_with_labels)

    user_classifications = []
    for (image_path, label) in image_paths_with_labels:
        user_classifications.append(get_user_class_from_image(image_path))

    # process results
    total_correct = 0
    for i in range(len(user_classifications)):
        total_correct += int(
            user_classifications[i] == image_paths_with_labels[i][1])
    accuracy = total_correct / len(user_classifications)
    print("Accuracy: {}".format(accuracy))


if __name__ == "__main__":
    main()
