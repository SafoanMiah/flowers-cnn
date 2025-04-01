# %% [markdown]
# # Visual Exploration of Neural Networks for Flower Classification
# AE2 – Visualization Presentation
#
# Questions for prefessor:
# - Do handmade visualizations work of things liek preocesses
# - Does it have to flow
# - As long as I submit a HTML, can I have an utilities file for the longer fucntions, usually visualization
#
# ### Introduction
#
# This notebook presents visual journey into the development of a convolutional neural network (CNN) for classifying flowers using the Oxford 102 Flower Dataset. Through these visualizations, we'll explore how a deep learning model can "see" and learn to recognize different flower species.
#
# ### Dataset Information
# [Souce Link](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) | [Labels](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/categories.html) | [Images on Drive](https://drive.google.com/drive/folders/1C_7gjRPE9claxoN5bAMajXYxz-_YqBhz?usp=sharing)
#
# The Oxford 102 Flower Dataset was compiled by Maria-Elena Nilsback and Andrew Zisserman at the Visual Geometry Group, University of Oxford. It contains images of flowers belonging to 102 different categories common in the United Kingdom. The dataset features:
#
# - 102 flower categories
# - Each category contains between 40-258 images
# - Collected from various sources including professional photographs and web searches
#
# ### Project Objectives
#
# This project aims to:
#
# 1. Visualize the data preparation process
# 2. Design and implement a CNN for flower classification
# 3. Visualize the training process and how the model learns over time
# 4. See what the neural network (NN) "sees" by visualizing activations and features
# 5. Analyze the model's performance

# %% [markdown]
# ---
# ## Data Loading and Exploration
#
# Before building the CNN, we need to load and understand the dataset:
#
# 1. Loading the image files and their labels
# 2. Exploring the dataset
# 3. Setting up data preprocessing for neural network training
#
# We'll start by importing the libraries and loading the dataset and setting some constants

# %%
import torch
import plotly.express as px
import pandas as pd
import numpy as np
import scipy.io
import json
import matplotlib.pyplot as plt
import seaborn as sns
import random
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from pathlib import Path
import torch.nn as nn
import torchvision.transforms as transforms
from IPython.display import HTML, display
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import plotly.subplots as sp


# Setting the seeds for reproducability
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# %% [markdown]
# ### Data Loading

# %%
# PyTorch device, (P.S. You're cooked ☠️ if you dont have a NVDIA GPU for this. RIP 🪦)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(f"Using device: {device}")

# Directories
data_dir = Path("data/102flowers/jpg")  # Directory containing the flower images
labels_file = Path("data/imagelabels.mat")  # File with image labels

# Load the category-to-names map
cat_to_name = "data/cat_to_name.json"
with open(cat_to_name, "r") as f:
    categories_dict = json.load(f)

categories = list(categories_dict.values())

# %% [markdown]
# Labels is in order the first label (index 0) corresponds to image_00001
#
# Here lets link each to them and the names category; image name format is image_00001.jpg to image_08189.jpg, then split the data before we start visually looking

# %%
labels = scipy.io.loadmat(labels_file)["labels"][0]
# str(i).zfill(5) pads with 0's for 5 total digits
image_names = [f"image_{str(i).zfill(5)}.jpg" for i in range(1, len(labels) + 1)]

# %%
# Giving flower_name
df = pd.DataFrame(zip(labels, image_names), columns=["label", "image"])
df["flower_name"] = df["label"].map(lambda x: categories_dict.get(str(x), "Unknown"))
df.sample(2)

# %% [markdown]
# ### Dataset Exploration
#
# I picked the dataset due to its diffucult nature with many flower looking similar, with light differences that even humans might not be able to always recognize. The dataset includes variations in lighting, angle, and background, making it an great test for a CNN's ability to learn meaningful features.
#
# This project wont go into too much detail on the flowers dataset itself but rather the Deep Learning CNN for it; however its still important to understand the dataset at a surface level first.

# %% [markdown]
# #### Distribution Analysis (bar chart)

# %%
class_counts = df["label"].value_counts().sort_index()

# Class Distribution with color
fig = px.bar(
    x=categories,
    y=class_counts.values,
    color=class_counts.values,
    labels={"x": "Flower Name", "y": "Number of Images"},
    title="Distribution of Flower Classes in Dataset",
    color_continuous_scale="Viridis",
)
fig.update_layout(
    xaxis=dict(tickangle=90, tickfont=dict(size=8)),
)
display(
    HTML(fig.to_html())
)  # Use IPython Display to show PLotly even in HTML format (off by default)

print(f"Average images / class:   {class_counts.mean():.2f}")
print(f"Minimum images / class:   {class_counts.min()}")
print(f"Maximum images / class:   {class_counts.max()}")

# %% [markdown]
# The distribution of the classes shows an imbalance. The average class contains approximately 80 images, but there's a wide difference between the least represented flowers (40 images) and the most common ones (258 images), of 645%.
#
# This does might actually cause some issues later down the line during the training process. With uneven representation the model will naturally learn from more common types during the iterations, which can lead to classfication bias where a model becomes great at some classes while underperforming in other ones.
#
# Although in some specific use cases this might be an advantage, for exmaple if more common flowers, are ones that will be scanned more using the model post production. Although its a good idea to keep this in mind for now let's also analyze some potential augmentation (without proper implementation) methods to get more data from what we already have.

# %% [markdown]
# #### Data Augmentation Exmaple


# %%
# Returns random image path
def random_img_path(df):
    """Function to return a random image path from the dataset"""
    image = df.iloc[random.randint(0, len(df))]["image"]
    img_path = data_dir / image
    return img_path


random_img_path(df)

# %%
original = Image.open(random_img_path(df))

# List of transformations to apply
transforms_list = [
    ("Original", original),  # No changes
    ("Rotated 15deg", original.rotate(15)),  # Roatate 15deg clockwise
    (
        "Flipped Top to Bottom",
        original.transpose(Image.FLIP_TOP_BOTTOM),
    ),  # Flip top to bottom
    (
        "Brightened",
        ImageEnhance.Brightness(original).enhance(random.uniform(1, 2)),
    ),  # Sets brightness from 100-200%
]

fig, axes = plt.subplots(1, len(transforms_list), figsize=(16, 4))

# Plot a cleaned trasposed images
for i, (title, transformed_img) in enumerate(transforms_list):
    axes[i].imshow(transformed_img)
    axes[i].axis("off")
    axes[i].set_title(title)

fig.suptitle("Augmented Images", fontsize=17)
plt.tight_layout()
plt.show()

# %% [markdown]
# This is a simple way on how we can augement data to get more variations from the limited data that we have for under represented categories. It can:
# * Improve genralization: by making the model more invatiant, getting used to and familiar with multiple angles, rotations, etc
# * Reduce overfitting
#
# Some other techniques include: color jitter, saturation, gaussian noise, etc.

# %% [markdown]
# #### Dimention Analysis (histogram + boxplot)


# %%
def get_dimentions(img_path):
    """Get the width and height of an image
    Ex: get_dimentions(df.iloc[0])"""
    img = Image.open(img_path)  # Open the image
    return img.size  # (width, height)


# Sample for simplicity and to get a general overview
sample = 200
dimentions = [get_dimentions(random_img_path(df)) for _ in range(0, sample)]

dimension_df = pd.DataFrame(dimentions, columns=["height", "width"])
# combine into width height into 1 col with a catergorical column
long_df = dimension_df.melt(var_name="dimention")

# Histogram and boxplot of spreads
fig = px.histogram(long_df, x="value", color="dimention", marginal="box")

display(HTML(fig.to_html()))

# %% [markdown]
# Keeping up with the distributions, here is an assesment of the dimentions spread (height, width), this shows a lot of variability in the dataset. The plot shows that most images maintain relatively similar aspect ratios with most of the widhts and heights being on one column each, but there's still a variation in overall image size with many that cont conform the majority size.
#
# We can see outliers that are one of; unusually small or large.
#
# This disparity in size does inpact the design of the Neural Network:
#
# 1. **Input standardization**: All images need to be of common dimension (ex. 200x200)
# 2. **Information loss**: While It's necessary to not overcomplicate the the NN, cropping or stretching images will cause loss of detail
# 3. **Aspect ratio changes**: Converting images to square dimensions will slightly distort some flowers
# 4. Median width: 500 | Median height : 667
#
# This will have to be considered as one of the steps for the preprocessing pipeline.

# %% [markdown]
# #### Color Channel Arrays (imshow)


# %%
def img_to_array(img_path):
    """Function to convert a specific flower image to an array given a path."""
    img = Image.open(img_path)
    return np.array(img)


def get_channel(array, channel=0):
    """Function to extract a specific channel from the image array.
    channel: 0=R, 1=G, 2=B for RGB images"""
    return array[:, :, channel]


# Get and array image
img_path = random_img_path(df)
img_array = img_to_array(img_path)

# transformations to do
# Function to extract a specific channel from the image array.
# Channel: 0=R, 1=G, 2=B for RGB images
transforms_list = [
    ("Original", img_array, None),  # No color map on original
    ("Red Channel", img_array[:, :, 0], "Reds"),
    ("Green Channel", img_array[:, :, 1], "Greens"),
    ("Blue Channel", img_array[:, :, 2], "Blues"),
]

# Plotting on subplots
fig, axes = plt.subplots(1, len(transforms_list), figsize=(16, 4))

# Apply transform, plot with colormap, title
for i, (title, transformed_img, cmap) in enumerate(transforms_list):
    if cmap:
        sns.heatmap(transformed_img, ax=axes[i], cmap=cmap, cbar=False)
    else:
        axes[i].imshow(transformed_img)
    axes[i].axis("off")
    axes[i].set_title(title)


fig.suptitle("Image Channels", fontsize=16)
plt.tight_layout()
plt.show()

# %% [markdown]
# This breaks down a flower image, into its fundamental RGB color channels, shown by the heatmap overlayes on top; this shows how the neural network actually "sees" the image, turning each of these images pixels into numbers representing intesntity for thier respective color channel.
#
# While we are capable of understanding the left image (Original) computers only work with numerical arrays representing the intensity of pixels, usually using the Red Green and Blue chells. These combines create full colors. Thes brighter areas indicate higher values on that picel are for that specific channel.
#
# This is one of the primary porblems with computer vision as NNs don't inherently see or understand patterns like the colors, petals or shapes. They look at the raw numerical values and need to learn to identify patterns. The classification CNN will need to figure these out by itself using the arrays of data that we extract from the images.

# %%
# This is how we woudl load in all array's for the images, however it takes too long
# To save time we'll go PIL Image -> Tensors directly, and ill precalc the mean and std once

# df['array'] = df.apply(img_to_array, axis=1)

# %% [markdown]
# #### Color Pattern Analysis (k-means)
# The reason I've decided to use K-means clustering here over traditional R G B addition then finding the top.

# %%
from sklearn.utils import shuffle


def extract_colors(img_path, n_colors=5):
    """Extract the dominant colors from an image using K-means"""

    # Load image and reshape to be a list of pixels
    img = Image.open(img_path)  # Open image
    img_array = np.array(img)  # Turn into array
    h, w, c = img_array.shape  # Translate muti-dimentions into individual variables
    reshaped_img = img_array.reshape(h * w, c)  # Reshpe

    # Sample of pixels to speed it up
    sample = shuffle(reshaped_img)[:10000]

    # Fit K-means, and get top n_colors
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(sample)

    colors = kmeans.cluster_centers_  # Get the colors
    colors = colors.astype(int)  # Converting to integer RGB values
    labels = kmeans.predict(reshaped_img)  # Labels for each pixel
    counts = np.bincount(labels)  # Percentage of each color
    percentages = counts / len(labels) * 100

    return colors, percentages


# %% [markdown]
# K-means looks for color groupings and intensity in a multi-dimensional space (RGB) and also intesity, by combining RGB so its not only 3 options; all which a simple color values aggregations can't replicate.
#
# In our use case it return [n_colors] number of most promenant colors in the image along side the percetage of thier respective appearances.

# %%
# Extract colors and percentages for n images

clusters = []
for i in range(200):
    colors, percentages = extract_colors(random_img_path(df))

    # RGB values in dataframe along side the Percetage
    for color, percentage in zip(colors, percentages):
        clusters.append(
            {
                "R": color[0],
                "G": color[1],
                "B": color[2],
                "Percentage": percentage,
                "Image": df.iloc[i]["image"],
            }
        )
clusters_df = pd.DataFrame(clusters)

# RGB columns into a single color column for color
clusters_df["color"] = clusters_df.apply(
    lambda row: f"rgb({row['R']}, {row['G']}, {row['B']})", axis=1
)

# 3D scatter plot
fig = px.scatter_3d(
    clusters_df,
    x="R",
    y="G",
    z="B",
    size="Percentage",
    hover_data=["Image"],
    title="3D Scatter Plot of RGB Values and Percentages",
    height=600,
)
# Set RGB color for each point
fig.update_traces(
    marker=dict(
        color=clusters_df["color"],  # Use the 'color' column for RGB values
        opacity=0.8,
    )
)

display(HTML(fig.to_html()))

# %% [markdown]
# The K-means clustering shows the most promenant colors in the images within the dataset; including hues and tones along side the hierarchy (percetange) of the appearance from the size:
#
# 1. **Green tones** are the most common, these mostly represent the leaves and stems which are almost always green
# 2. **Flower-specific colors** moving slighlty further away from the greens one can see its a lot more sparse values of pinks, purples, blues, red, and white all being well-represented; these are the colors of the petals and will be one of the main features in the distinguishing of flowers
# 3. **Lack on specific axis**: Looking at the graph if we where to draw diagonal from RGB (255, 255, 0) to (0, 0, 255), there is a clear split with almost all the petal colors tending towards the red-blue side rather than the blue-green. Indicating that just as real life, in this dataset there's very few green-blue petal colors, while there are many red-blue ones.
#
# This color chart is important to understand for the neural network, as color is one of the primary distinguishing feature for many flower species. The model will need to be able to use these color patterns to its advantage to classify the flowers.

# %% [markdown]
# #### Data Analysis -> CNN Design
# This section has been exploration of the Oxford 102 Flowers Dataset's attributes trough visualizing:
# * Distribution of classes
# * Image dimentions
# * RGB channels
# * Color petterns
#
# For the next section, there will be more of a fucus on a subset of the categories to showcase a NN building process. I;ll go over key concepts and visualization techniques while also keeping inmind constraits like training time and GPU VRAM limitations.
#
# ---

# %% [markdown]
# ## Convolutional Neural Network Design
# What is are Convolutional Neural Networks (CNNs)?
# * They represent a specialized type of deep learning NN's designed to process structures grid data like images. CNN's mantain spacial realtionships on multiple dimentions tough the use of multiple layers.
# * CNNs can learn hierarchical patterns
#     * Low-level features: Initial layers identify basic elements such as edges, corners, and colors
#     * Mid-level features: Middle layers combined into more complex patterns like textures and shapes
#     * High-level classification: Deeper layers put these patterns together into class-specific features
#
# CNNs are great for the task at hand as the flowers have subtle variations between some species which may require fine and detailed feature selection and detection. Other classification approaches would likley struggle with finding the differences in these images.

# %% [markdown]
# ### Preparing Dataset
# First step as always is to clean and prepare the dataset, for this project the focus will be on a subset of 5-10 flowers, to show key concepts while keeping training time and resources used up to a reasonable amount.

# %%
# Get unique flower names from the dataset


sample_categories = 5

# Select the first n samples
sample_categories = categories[:5]
print(f"Selected Categories: {sample_categories}")

sample_df = df[
    df["flower_name"].isin(sample_categories)
].copy()  # Filter the DataFrame for the selected categories
sample_df["image"] = sample_df["image"].apply(
    lambda x: data_dir / x
)  # Put in exact file location to editing later on
sample_df.sample(2)

# %% [markdown]
# #### Train Validation Test Split

# %%
# Train Test Val split the indexes at 70 / 15 / 15
train, temp = train_test_split(sample_df, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

train_len, val_len, test_len = len(train), len(val), len(test)

print(f"Total number of images:         {len(sample_df)}")
print(f"Number of classes:              {len(sample_df['flower_name'].value_counts())}")

# %% [markdown]
# #### Visualizing Proportionas (pie-chart)

# %%
# Subplot 1 row, 2 columns
fig = sp.make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["Dataset Split: Train, Test, Val", "Category Distribution"],
    specs=[[{"type": "domain"}, {"type": "domain"}]],
)

splits = [train_len, val_len, test_len]
split_label = ["Train Size", "Validation Size", "Test Size"]
fig_split = px.pie(values=splits, names=split_label)

sample_categories = sample_df["flower_name"].value_counts().index.tolist()
sample_cat_counts = sample_df["flower_name"].value_counts().values
fig_category = px.pie(values=sample_cat_counts, names=sample_categories)

fig.add_trace(fig_split.data[0], row=1, col=1)
fig.add_trace(fig_category.data[0], row=1, col=2)

display(HTML(fig.to_html()))

# %% [markdown]
# This sampled representatation of the full dataset contains the same potential problems with the full one; varying image dimentions, different frequencies of images per category and similar looking flower species.
# * The piechart on the left, shows a 70% 15% 15% split for Train Validation and Test, of our data, this makes sure that most of our data is used for training while we still keep a suffieicnt amount for hyperparamenter tuning based on Validation results and a final score using the Test split.
# * While the one on the right describes the class disctribution for the selected subset and as shown the imbalances remain as per the original dataset
#
# This upcoming section will be a preprocessing pipeline to make anymore necessary changes to the images themselfs before we can create and feed into a model.

# %% [markdown]
# ### Image Preprocessing Pipeline
# Before we feed the flower images into the NN, they need to go trough some preprocessing steps.
#
# The raw images in the datatset vary in, sizes and aspect ratios which would hold back the image training, the pipeline will attempt at standardising these images in a format optimes for CNNs.
#
# Previous dimentional analysis gave media width x height at 500 x 667, I'll scale it down to the last squared number at 256x256 then center it to 224x224 to retain center details, cropping larger images or padding smaller ones.
#
# NOTE: The following values where precalulates by me based on all the images, it takes too long to do everytime
# * Mean: [0.43553434, 0.37773397, 0.2879371]
# * Std:  [0.29582291, 0.24464045, 0.26921128]

# %%
# Pipeline for train with augmentation
transform_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(
            224, scale=(0.8, 1.0)
        ),  # Randomly crop to resize to 224x224
        # transforms.RandomVerticalFlip(),                      # Vertical Flip (doesnt make much sense flowers)
        transforms.RandomRotation(15),  # Random 20 degree turn
        transforms.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
        ),  # Randomly adjust Brightness, contrast, hue and saturation of the image
        transforms.ToTensor(),  # Image to tensors
        transforms.Normalize(  # Normalize pixel data for each channel
            mean=[0.43553434, 0.37773397, 0.2879371],  # ImageNet mean
            std=[0.29582291, 0.24464045, 0.26921128],  # ImageNet std
        ),
    ]
)

# Pipeline for test and validation which doesnt need augmentation since it isnt learning
transform_val_test = transforms.Compose(
    [
        transforms.Resize(256),  # Resize to resize to 224x224
        transforms.CenterCrop(224),  # Crop to 224x224 to match input size for the model
        transforms.ToTensor(),  # Image to tensor
        transforms.Normalize(  # Normalize with the same mean and std as training
            mean=[0.43553434, 0.37773397, 0.2879371],
            std=[0.29582291, 0.24464045, 0.26921128],
        ),
    ]
)

# Augmentation was visualized previously, this is putting that into a pipleine


# %%
def tensor_to_img(tensor):
    """Convert Tesor to NumPy image by de-normalizing"""
    mean = torch.tensor([0.43553434, 0.37773397, 0.2879371]).view(
        3, 1, 1
    )  # reshape to 3D from 1D, (Color, Height, Width)
    std = torch.tensor([0.29582291, 0.24464045, 0.26921128]).view(3, 1, 1)
    img = tensor * std + mean  # denormalization
    return np.clip(
        img.permute(1, 2, 0).numpy(), 0, 1
    )  # change shape from (C, H, W) -> (H, W, C)


def plot_transformed(ax, img, title):
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title)


# Visaliza the transformations used in the pipleine
transforms_list = [
    ("Original", lambda x: x),
    ("Resized Crop", transforms.RandomResizedCrop(224, scale=(0.8, 1.0))),
    ("Rotation", transforms.RandomRotation(15)),
    ("Color Jitter", transforms.ColorJitter(0.1, 0.1, 0.1, 0.05)),
    ("ToTensor", transforms.ToTensor()),
    # Normalized image won't look different, rather just the values of the pixels behind the scenes are standardized
]

fig, axes = plt.subplots(1, len(transforms_list), figsize=(12, 5))
img = Image.open(random_img_path(df))
for i, (title, transform) in enumerate(transforms_list):
    img = transform(img)
    if torch.is_tensor(img):  # Convert to img if its a tensor
        img = tensor_to_img(img)
    plot_transformed(axes[i], img, title)


plt.tight_layout()
plt.show()

# %% [markdown]
# Developed two distinct preprocessing pipelines:
# * Training Pipeline: Includes data augmentation to artificially expand the variey of our dataset and help the model generalize better. By adding variations in the training images, model learns to recognize flowers under different conditions.
# * Validation/Test Pipeline: Does not include augmentation because we want to evaluate the model on unmodified images for a stable benchmark for assessing model performance.
#
# Focussing on the Training Pipeline:
# * RandomResizedCrop: Crops a random portion of the image and centers it down to 224×224 pixels, showcasing different framing / positions on the frame
# * RandomRotation: Rotates the image by max 15 degrees in either direction, showcasing different orientations
# * ColorJitter: Slightly adjusts brightness, contrast, saturation, and hue to showcase different lighting conditions
#
# The pipeline ends with two important operations:
# * ToTensor: Converts the PIL image to a PyTorch tensor, pixel value range go from 0-255 to 0.0-1.0, and rearranges dimensions from (Height, Width, Channels) to (Channels, Height, Width)
# * Normalize: Standardizes each color channel using mean and standard deviation values. This centers the data distribution and helps the neural networks speed
#
# The visualization below it showcases some of the transformation step in the pipeline for a simple flower and how passing trough it changes how the computer "sees" it.

# %% [markdown]
# #### Batch Processing + DataLoader
# Before we can make / feed the preprocessed data into the Network we need to improve the loadinging mechanism. Processing images specifically is bound to be very slow for training. To better this we can use batch processing at the same time, which significatly speeds up training speed by making use of parallel computing (by using modern GPUs) and by optimizing memeory trasfers.
#
# Batch processing also updates the models weights based ont he average gradiesnts across multiple exmaples, which cleans to a more stable and smooth convergence compared to when images are processed individually.
#
# The big advangtage this is the reduction in of risks of being trapped in a local minima when calculating loss and allowing the use of larger learning rates due leading to faster convergence.
#
# PyTorch also provides a framework for its Datasets and DataLoaders. The Dataset class outlines how each image is acessed and transformed while Data Loader handles the batching process. This combination adds to the pipeline that augments the imges.

# %% [markdown]
# ##### Dataset


# %%
class FlowerDataset(Dataset):
    """Dataset Class to load, store and process images"""

    def __init__(self, dataframe, transform):
        # Loading data
        self.dataframe = dataframe
        self.transform = transform
        self.classes = sorted(self.dataframe["flower_name"].unique())

    def __len__(self):
        # Returns size of the dataframe
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get the indeth'th same
        path = self.dataframe.iloc[idx]["image"]
        label = self.dataframe.iloc[idx]["label"]

        image = Image.open(path)
        image = self.transform(image)

        return image, label


# %%
# Create dataset objects for the splits
train_dataset = FlowerDataset(train, transform=transform_train)
val_dataset = FlowerDataset(val, transform=transform_val_test)
test_dataset = FlowerDataset(test, transform=transform_val_test)

# %%
image_idx = random.randint(0, len(train_dataset))

fig, axes = plt.subplots(2, 5, figsize=(15, 5))

for ax in axes.flatten():
    image, label = train_dataset.__getitem__(image_idx)
    name = categories_dict[str(label)]
    ax.imshow(
        tensor_to_img(image)
    )  # Returns tensors which we can turn back into an array with a previously made funtion
    ax.axis("off")

fig.suptitle(f"Augmented variations of: {name} | {label}", fontsize=18)
plt.tight_layout()
plt.show()

# %% [markdown]
# The data augmentation is now applied dynamically during training. This means that, as shown in the visualizationthe model will see different variations of the same images across different epochs.

# %% [markdown]
# #### DataLoader

# %%
batch_size = 32  # Good balance for memeory usage and computational efficiency
workers = 2  # 4 Threads used for parallel data loading
gpu = torch.cuda.is_available()  # True if there is a GPU

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,  # Shuffle training data
    pin_memory=False,  # Faster GPU transfers if a GPU is available
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,  # No need to shuffle validation or testing
    pin_memory=False,
)

test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, pin_memory=False
)


# %%
def visualize_batch(loader, batch_num=1):
    # Get all items for a batch
    images, labels = next(iter(loader))
    images = [tensor_to_img(img) for img in images]

    # 16 by 2 for each batch
    fig, axes = plt.subplots(2, 16, figsize=(16, 3))
    axes = axes.flatten()

    # Plot
    for img, ax in zip(images, axes):
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.suptitle(f"Batch Number: {batch_num}", fontsize=16)
    plt.show()


visualize_batch(train_loader, batch_num=1)
visualize_batch(train_loader, batch_num=2)

# %% [markdown]
# This implementation completes the pipeline:
# 1. Dataset Class: The `FlowerDataset` class is a blueprint to acess images using `__getitem__`, with augmentation being applies everytime
# 2. DataLoader: The loader wraps the dataset class and handles the batching of our data
# The batching visualizations show some examples on what each batch would look like to us.
#
# Overall this approach gives us 3 advantages: Training Efficiency, Memory Optimization, Better Generalization; all the while making sure that the neural network gets a constant stream of properly formatted varied data.

# %% [markdown]
# ### CNN Architechture
#
# A good architechture balanecs the models complexity while adhering to the requirements of the specific dataset. This this Flower clssification model the input is 224x244 RGB images; the model needs to be able to catch small dedtails like petal texture and patterns as well as broad patternns like overall shape and color.
#
# NOTE: For this project I'll stick to developing 1 good model instead of iterating layers in and out to find best combinations.
#
# | Layer Type | Parameters | Output Dimensions |
# |------------|------------|-------------------|
# | Input | 224×224 RGB image | 224×224×3 |
# | Conv + BatchNorm + ReLU | 64 filters, 3×3 kernel, padding=1 | 224×224×64 |
# | Conv + BatchNorm + ReLU | 64 filters, 3×3 kernel, padding=1 | 224×224×64 |
# | MaxPooling | 2×2 pooling | 112×112×64 |
# | Conv + BatchNorm + ReLU | 128 filters, 3×3 kernel, padding=1 | 112×112×128 |
# | Conv + BatchNorm + ReLU | 128 filters, 3×3 kernel, padding=1 | 112×112×128 |
# | MaxPooling | 2×2 pooling | 56×56×128 |
# | Conv + BatchNorm + ReLU | 256 filters, 3×3 kernel, padding=1 | 56×56×256 |
# | Conv + BatchNorm + ReLU | 256 filters, 3×3 kernel, padding=1 | 56×56×256 |
# | MaxPooling | 2×2 pooling | 28×28×256 |
# | Conv + BatchNorm + ReLU | 512 filters, 3×3 kernel, padding=1 | 28×28×512 |
# | Conv + BatchNorm + ReLU | 512 filters, 3×3 kernel, padding=1 | 28×28×512 |
# | MaxPooling | 2×2 pooling | 14×14×512 |
# | Conv + BatchNorm + ReLU | 512 filters, 3×3 kernel, padding=1 | 14×14×512 |
# | Conv + BatchNorm + ReLU | 512 filters, 3×3 kernel, padding=1 | 14×14×512 |
# | MaxPooling | 2×2 pooling | 7×7×512 |
# | Flatten | Convert to 1D | 25,088 |
# | Dense + ReLU | 4096 neurons | 4096 |
# | Dropout | 0.5 dropout rate | 4096 |
# | Dense + ReLU | 1024 neurons | 1024 |
# | Dropout | 0.5 dropout rate | 1024 |
# | Dense | Number of flower classes | Number of classes |
#
# This follows VGG-style principles
#
#
# Components:
# - **Convolutional Layer**: Applies filters for edges, textures, shapes in the input. Early layers detect simple features, deeper layers are for more complex patterns.
# - **Kernel Size (3×3)**: Spatial extent / size of selection square of the convolution. 3×3 kernels can detect small grained features, this is also a very standard numbe
# - **Padding (padding=1)**: Adds a border of pixels (typically zeros), keeps spatial dimensions after convolution. Padding=1, a 3×3 kernel will keep the same height and width in the output feature map.
# - **BatchNorm**: Normalizes the outputs of a layer across the batch, making training more stable and efficient.
# - **ReLU (Rectified Linear Unit)**: Activation function for non-linearity by setting all negative values to zero. max(value, 0). Helps to learn non-linear relationships.
# - **MaxPooling (2×2)**: Downsamples the feature maps by taking the maximum value in each 2×2 region. Halfs spacial dimensions.
# - **Increasing Filter Counts (64→128→256→512)**: Since spatial dimensions decrease, the number of filters can be incresed for more complex features.
# - **Flatten**: 3D feature maps (height × width × channels) to a 1D one, that can be processed by fully connected layers.
# - **Dense Layer (Fully Connected)**: Connects every neuron to all neurons in the previous layer, let's the network to learn global patterns across the entire image.
# - **Dropout (0.5)**: Randomly deactivates 50% of neurons during each training iteration to prevent overfitting.
# - **Stacked Conv Layers**: While the outputs are the same they allow for the netwrok to learn more complex features.


# %%
class Flower_CNN_V1(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), n_classes=len(sample_categories)):
        super().__init__()

        # Conv block maker function as there is a repeating pattern
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
            )

        # Create convolutional layers using the function
        self.conv_set_1 = conv_block(input_shape[0], 64)
        self.conv_set_2 = conv_block(64, 128)
        self.conv_set_3 = conv_block(128, 256)
        self.conv_set_4 = conv_block(256, 512)
        self.conv_set_5 = conv_block(512, 512)

        # Flatten out the connected layers
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.dense_set_1 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5)
        )
        self.dense_set_2 = nn.Sequential(
            nn.Linear(4096, 1024), nn.ReLU(), nn.Dropout(0.5)
        )

        self.dense_set_3 = nn.Linear(1024, n_classes)

    def forward(self, x, return_intermediate=False):
        intermediate_outputs = {}

        # Pass through conv layers and save intermediate outputs
        x = self.conv_set_1(x)
        intermediate_outputs["conv_set_1"] = x
        x = self.conv_set_2(x)
        x = self.conv_set_3(x)
        x = self.conv_set_4(x)
        x = self.conv_set_5(x)

        # Flatten out the connected layers
        x = self.flatten(x)
        intermediate_outputs["flatten"] = x

        # Pass through fully connected layers
        x = self.dense_set_1(x)
        intermediate_outputs["dense_set_1"] = x
        x = self.dense_set_2(x)
        intermediate_outputs["dense_set_2"] = x
        x = self.dense_set_3(x)
        intermediate_outputs["dense_set_3"] = x

        if return_intermediate:
            # Return every step for visualizations sake
            return intermediate_outputs
        else:
            return x


# %% [markdown]
# #### Visualizing Intermidiate Steps

# %%
image, label = next(iter(train_dataset))
image = image.unsqueeze(
    0
)  # adhere to input shape [1, 3, 224, 224] 1 batch, 3 channels, 244x244

# Instance of the NN
model = Flower_CNN_V1()
# Get the intermediate results
intermediate_outputs = model(image, return_intermediate=True)
print(f"Layer images: {intermediate_outputs.keys()}")

# Remove a dimentions (batch) from tesnor, create new tesnor with detatch without gradient
conv1 = intermediate_outputs["conv_set_1"].squeeze(0).detach().cpu().numpy()

# Plot first 12 feature maps
fig, axes = plt.subplots(2, 6, figsize=(15, 5))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(conv1[i], cmap="viridis")
    ax.axis("off")
fig.suptitle("First 12 Feature Maps from Conv Layer 1", fontsize=18)
plt.tight_layout()
plt.show()

# %% [markdown]
# This is a plot of the first 12 feature maps fromt he first convolutional layer (as exmaple) of the CNN. These feature maps represent how the model is detecting different patterns like edges and tectures inthe in imput image after the first convolution and activation. There is a total of 64 in the first Conv layer, however here I visulize 12.

# %%
flat = intermediate_outputs["flatten"].squeeze(0).detach().cpu().numpy()

fig = px.line(y=flat, title="Flattened Output Visualization")
display(HTML(fig.to_html()))

# %% [markdown]
# This is a graph of the flattened outputs, it shows values of the activation neurons after the convolutional and pooling layers, before they are passed into the fully connected layers. At each point it corresponds to the activation value of a neuron.

# %%
# Extract and reshape activations from each dense layer
dense1 = intermediate_outputs["dense_set_1"].squeeze(0).detach().cpu()
dense2 = intermediate_outputs["dense_set_2"].squeeze(0).detach().cpu()
dense3 = intermediate_outputs["dense_set_3"].squeeze(0).detach().cpu()


# Reshape dense activations to 2D grids (for visualization)
def reshape_dense_to_2d(dense_tensor):
    size = dense_tensor.shape[0]
    side_length = int(np.sqrt(size))
    try:  # Try to reshape into a square grid
        return dense_tensor.reshape(side_length, side_length).cpu().numpy()
    except:  # Keep as 1D if it can't be reshaped to square  # noqa: E722 (ruff ignore)
        return dense_tensor.cpu().numpy().reshape(1, size)


dense1_reshaped = reshape_dense_to_2d(dense1)
dense2_reshaped = reshape_dense_to_2d(dense2)
dense3_reshaped = reshape_dense_to_2d(dense3)

fig = sp.make_subplots(
    rows=1, cols=3, subplot_titles=["Dense Set 1", "Dense Set 2", "Dense Set 3"]
)

# Add heatmaps for each dense layer
fig.add_trace(px.imshow(dense1_reshaped).data[0], row=1, col=1)
fig.add_trace(px.imshow(dense2_reshaped).data[0], row=1, col=2)
fig.add_trace(px.imshow(dense3_reshaped).data[0], row=1, col=3)

fig.update_layout(
    title_text="Activations from Dense Layers",
    xaxis_title="Neuron Index (X)",
    yaxis_title="Neuron Index (Y)",
)
display(HTML(fig.to_html()))

# %% [markdown]
# As previously shown this shows the streght of each neuron, however this is visualizes as a 2D heatmap with each combination of X and Y indexes pointing to one neuron. Ex (0,1) is the second neuron.
#
# What's also very visible is the rapid drop in neurons that are considered from 4096 -> 1024 -> 5 (which is the number of clases that we sampled)
# * Intrensity of a neuron (color) indicates its "reaction" to a feature in the input data, weather its important / dominant, or not.
#
# `dense_set_3` will output an "array" with values that represent the likleyhood that the image belongs to each class, for example an output could be: [0.100, 0.006, 0.160, -0.060, 0.299], we call these logits; raw scores.
# These logits are taken trough a `softmax` function that converts each into a value between 0-1 representing prababilty that it belongs to that class (all the probabilities sum to 1).
#
# The lastly a class is chosen based on this.

# %% [markdown]
# ## Training the Flower CNN
#
# Training a deep neural network has a couple of components and steps that we'll work trough in the upcoming section:
# 1. **Model Setup**
#    - Initialize model
#    - Move model to device (CPU/GPU)
#    - Training mode
#
# 2. **Optimizer Setup**
#    - Initialize optimizer
#    - Initialize loss function
#
# 3. **Training Loop**:
#    - For every batch within each epoch
#        1. Forward pass: Prcesses input data trough the model the get a prediction
#        2. Loss calculation: Finds the error between the prediction and the true labels
#        3. Zero the gradients: Resets all parameters abck to 0
#        4. Backward pass: Calculates the gradients, how much each weight contributes to the error rate
#        5. Optimize parameters: Adjust parameters like weights based on the gradient to minimize the loss
#
# 4. **Validation Steps**:
#    - Set model to evaluation mode
#    - Disable gradient calculation
#    - Run forward pass on validation data
#    - Calculate validation metrics
#    - Set model back to training mode
#
# 5. **Test Model**
#
# The most important part is the inner loop: zero gradients -> forward pass -> loss calculation -> backward pass -> optimizer step.

# %%
print(f"Using device: {device}")


# Function to calculate accuracy
def calc_accuracy(y_pred, y_true):
    """Find the accuracy based on the predicted and true labels"""
    y_pred = y_pred.argmax(dim=1)  # Find the index of the max value, prediction
    correct = (
        torch.eq(y_true, y_pred).sum().item()
    )  # Compare prediction with true label, get value
    acc = (correct / len(y_pred)) * 100

    return acc


# %%
def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, device):
    """
    Full function to train / validate a PyTorch model

    args:
    - model: PyTorch model
    - train_loader: DataLoader
    - val_loader: DataLoader
    - epochs: number of epochs
    - criterion: loss function
    - optimizer: optimizer
    - device: cuda or cpu
    """

    train_loss, val_loss, train_acc, val_acc = [], [], [], []

    for epoch in range(epochs):
        model.train()  # set model to training mode
        epoch_train_loss, epoch_train_acc = 0, 0
        size = len(train_loader)

        print(f"Epoch {epoch + 1}/{epochs} - Training...")

        # Train, i for
        for i, (image, label) in enumerate(train_loader):
            image, label = (
                image.to(device),
                label.to(device),
            )  # Move the data to gpu if available or cpu

            optimizer.zero_grad()  # Zero gradient (reset)

            y_logits = model(image)  # Forward pass

            loss = criterion(y_logits, label)  # Calculate loss, accuracy
            acc = calc_accuracy(y_logits, label)

            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            epoch_train_loss += loss.item()  # Update metrics
            epoch_train_acc += acc

            # i is from enumeration
            if (i + 1) % 10 == 0:  # Progress every 10 intervals
                print(f"Batch {i + 1}/{size}")

        train_loss.append(epoch_train_loss / size)
        train_acc.append(epoch_train_acc / size)

        # Validation
        model.eval()
        epoch_val_loss, epoch_val_acc = 0, 0
        size = len(val_loader)

        with torch.inference_mode():
            for image, label in val_loader:
                # Move data to device
                image, label = image.to(device), label.to(device)
                y_logits = model(image)  # Forward pass
                loss = criterion(y_logits, label)  # Calculate loss, accuracy
                acc = calc_accuracy(y_logits, label)

                epoch_val_loss += loss.item()  # Update metrics
                epoch_val_acc += acc

        val_loss.append(epoch_val_loss / size)
        val_acc.append(epoch_val_acc / size)

        print(
            f"Epoch {epoch + 1}/{epochs} complete - Val Loss: {val_loss[-1]:.5f}, Val Acc: {val_acc[-1]:.2f}%"
        )
        print("-" * 30)  # line

    return train_loss, val_loss, train_acc, val_acc


# %%
model = Flower_CNN_V1().to(device)

# Define loss function and optimizer
criterion = CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 10

# %% [markdown]
# Here Ive initialized the model and moved it to device, which is a GPU (with CUDA) if available, otherwise the CPU. A GPU would allow the model to train much faster.
#
# The loss functions is CrossEntropy Loss which is a fairly standard choice when the problem consists of multiple classes.
# * It applies softmax to the function to convert logits to probabilities
# * It calculates the negative log likelihood loss, mesuing how far the predicted probability distribution is form the true distribution

# %% [markdown]
# ![Negative Log Gif](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*umD9d_BlMsfWQl0as7IipQ.gif)
#
# Source: [Medium](https://medium.com/data-science/cross-entropy-negative-log-likelihood-and-all-that-jazz-47a95bd2e81)

# %%
train_loss, val_loss, train_acc, val_acc = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=epochs,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
)
print("Training complete!")
