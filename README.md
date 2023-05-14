# DatasetCreation
# Code Explanation
The given code is for generating a custom dataset by combining object images and background images. The dataset is generated by creating a CustomDataset class which inherits from the torch.utils.data.Dataset class. This CustomDataset class has three arguments which are directories where object images, background images, and the final combined images with targets will be saved. Other than these arguments, it also takes an optional argument transform which is used to apply transformations on the object images. The num_samples argument specifies the number of samples we want to generate in the dataset. If this argument is not passed, the total number of samples is the product of the number of object images and background images. The total number of samples will be set to num_samples if it is smaller than the default.

The __len__ method returns the number of samples in the dataset. The __getitem__ method is used to generate a single sample. It takes an index idx as an argument and uses it to determine which object image and which background image to use to create the final combined image.

In __getitem__, we first calculate the indices of the object and background images using idx and the length of the background and object filenames. Then, we load the object image using the calculated index and the background image randomly from the available backgrounds. The class label for the object image is extracted from the filename using the split method.

Next, we apply the specified transformation to the object image and combine it with the background image. The combine_images function is used to combine the images and create a target dictionary that contains information about the class label and bounding box coordinates. The target dictionary is saved as an XML file using the save_target_as_xml function.

Finally, we get images and target in xml format in target folder which can used to train the object detection model.

# Overall Functionality
The given code generates a custom dataset by combining object images and background images. The object images are randomly placed on the backgrounds, and the resulting images are saved in a target directory along with an XML file that contains the class label and bounding box coordinates. The dataset is generated by creating a CustomDataset class which takes in the directories of the object images, background images, and target images, along with the optional argument of applying transforms on the object images. The num_samples argument specifies the number of samples we want to generate in the dataset. The resulting dataset contains num_samples images, each with a unique object and background combination.

![Example Image](./final_sample/background_(69).jpg)
