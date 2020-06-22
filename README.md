# Deep Learning for Diversity Inclusion in Media

<img src='images\notebook_images\title.gif' align="center">

## Introduction

Disclaimer

This work is meant to fulfill educational purposes. The hypothetical business case, and results of modeling are not meant to be considered as representative of sentiments or aims of the Netflix brand, or any of its affiliates.

**With the creative force of  Generative Adversarial Networks (GANs), we are able to generate lifelike faces that are able to be animated and mapped on the current faces within movies, television shows, online learning platforms, advertisements, and any other media that showcases human faces.** Of course such a disruptive technology would need to be applied in a closed and controlled environment, and so for the purposes of this project, **the specific business case is in the frame of a pitch to Netflix executives to enable viewers to utilize GANs and First Order Motion Models to alter the cast of any video streamed on the Netflix Platform.** 

This project is intended to showcase the transformative power that GANs present to the current platforms available for viewing images and video content. The work is sectioned into 5 Jupyter Notebooks, each focusing on a seperate stage of the project, these are as follows:

1. **Mining & Cleaning LinkedIn Data**: Code associated with scraping LinkedIn for image data related to a person's social media to be fed into a GAN for face generation.
2. **Generating Faces with a DCGAN**: Using a Deep Convolutional Generative Adversarial Network to generate faces based on the scraped image data from LinkedIn.
3. **Generating High-Rez Faces with StyleGAN2**: Create a pre-trained instance of StyleGAN2 to generate high resolution 1024x1024 images of faces.
4. **Animating Faces with First Order Motion Model**: Code which allows mapping of generated faces to motion and expressions in a source video in order to animate the generated faces to match the original video content.
5. **Hypothesis Testing**: Here a hypothesis is made on the public perception of this model affecting change on views of diversity inclusion in media, and testing said hypothesis.

## Objectives

In demonstrating the Custom Cast, or Generate/Animate (G/A) System, there are two primary goals:

* Generate high resolution lifelike images of human faces with GANs
* Animate faces with motion and countenance with First Order Motion Models mapping faces to source video content

## 1. Mining & Cleaning LinkedIn Data

To demonstrate how GANs are capable of generating image data that is not only descriptive of people, but also personally relevant, the first two notebooks of this project will focus on obtaining image data from the profile pictures of a user's connection base (my own), and from these images generate artificial faces that are aggregated from my personal LinkedIn connections. Here we mine the social media facial data from LinkedIn with web scraping techniques and process this data to be fed into the GAN. In this notebook, the following was conducted:

- Navigated LinkedIn with `selenium` in order to locate the connections page of a particular user.
- Scraped LinkedIn for profile picture data related to a particular user with `BeautifulSoup`.
- Located and extracted faces from the profile picture data using `OpenCV2`. 
- Created visualizations with `wordCloud` & `matplotlib` based on the occupation information of the connections of a particular user.
- Resized these face images that will be fed into our GAN in the next notebook. 
- Visualized the entire dataset retrieved from LinkedIn and explored the pixel intensity values of a sample of images from the dataset.

### LinkedIn Dataset

<img src='images\notebook_images\linkedin_dataset.png' align="center">

### Image Analysis

<img src='images\notebook_images\face_pixels.png' align="center">


## 2. Generating Faces with a DCGAN

 **For the purposes of the hypothetical business case related to Netflix, we will use GANs to create images resembling the faces of a person's social media network, which can be used to supplant the original cast in a movie or television show being streamed on the Netflix platform**. This could **lead to creating casts that are more relatable to a viewer based on the images of those that the viewer interacts with on social media. This relatability which many viewers do not typically enjoy from the movies and shows they currently watch can lead to an increase in user satisfaction of the Netflix Platform, and thus to an increase in the member base**.
 
 ### Generative Adversarial Network Architecture
 
 <img src='images\notebook_images\gan_archi_2.png' align="center">
 
 
 #### Discriminator
 
 The discriminator referred to as  𝐷  is responsible for the adversarial nature of the GAN. The discriminator is a classifier which in most cases outputs a probability that a particular data instance it receives is either from the training data, or from the generator. For the purposes of this project, the real images will be that of faces scraped from LinkedIn. For the discriminator, these are considered the positive class while it is training. The negative class while training is therefore those images that have been created by the generator. 
 
  <img src='images\notebook_images\discriminator.png' align="center">
 
 #### Generator
 
 The generator known as  𝐺  is therefore responsible for the generative component of the GAN. The generator is able to create data which is meant to replicate the training data, and creates said data from a random input known as the Latent Noise Vector. This random noise referred to as  𝑧  is then transformed into an output that is similar to the training data by receiving feedback from the Discriminator. 
 
 <img src='images\notebook_images\generator.png' align="center">
 
### DCGAN Results

#### First Generated Image

<img src='images\notebook_images\first_generated_face.png' align="center">

#### DCGAN 50 Epochs

<img src='images\notebook_images\faces_dcgan_50e.gif' align="center">

#### DCGAN 200 Epochs

<img src='images\notebook_images\faces_dcgan_200e.gif' align="center">

#### DCGAN 500 Epochs

<img src='images\notebook_images\faces_dcgan_500e.gif' align="center">



## 3. Generating High-Rez Faces with StyleGAN2

In continuing the hypothetical business case of GAN usage with the Netflix platform, we explored the output of high quality images from StyleGAN2. It is with images of high definition that it will be possible to **create artificially generated casts indistinguishable from real actors for media streamed on the Netflix Platform to allow viewers customization of their media.** As demonstrated in the previous notebook, in order to produce high-quality images, we needed the performance of a GAN with deeper architecture, and access to a larger dataset of face images. To this end, we turn to StyleGAN2 and the FFHQ dataset.

In this notebook we:

* Implement a pretrained StyleGAN2 cloned from the Nvidia StyleGAN2 github repository
* Generate over 300 1024x1024 resolution images of faces that are near indistinguishable from samples within the FFHQ training set.
* Create interpolation footage of the latent space between two seed images generated by StyleGAN2


### StygeGAN2

StyleGAN2 is an improvement of the quality of images produced, and resolves these visible artifacts developed by a team of Nvidia Engineers, those members being Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, Timo Aila. StyleGAN is a method for generating high-resolution images with a similar architecture to ProGAN while adding a style transger methodology that alters the generator at each level of resolution by upscaling its resolution and feeding in a random input referred to as style noise. StyleGAN2 differs from StyleGAN by:

- Moving the noise module outside of the style module.
- Path Length Regularization.
- The mean is not needed in normalizing the features.
- Weight Demodulation, which simplifies the StyleGAN design by replacing the instant normalization design.

### FFHQ Dataset

As described by the NVlabs GitHub, the Flickr-Faces-HQ (FFHQ) dataset is a high-quality image dataset of human faces. It was first introduced in the StyleGAN paper A Style-Based Generator Architecture for Generative Adversarial Networks Tero Karras (NVIDIA), Samuli Laine (NVIDIA), Timo Aila (NVIDIA) http://stylegan.xyz/paper.

FFHQ is a relatively large dataset with around 70,000 1024x1024 resolution png images. The faces within the images contain variations in age, ethnicity, accessories like glasses and hats, as well as image backgrounds. The dataset was crawled from Flickr and is therefore a further proof of the usage of social media data with GANs. However because of its source, it also contains the bias within the website in terms of diversity. The images were cropped and aligned using dlib, and Amazon Mechanical Turk was used to remove noise from the images such as paintings, photos of photos, and statues.

### StyleGAN2 Results

#### High Resolution Image

<img src='images\notebook_images\man1.png' align="center">

#### Interpolation

<img src='images\notebook_images\morph_gif.gif' align="center">


## 4. Animating Faces with First Order Motion Model

Having successfully generated high resolution images of faces, the last functionality that needs to be demonstrated is mapping these faces to a video source. In this hypothetical business case the original video source will be in the form of a television show or movie being streamed on the Netflix platform. **First Order Motion Models will enable the viewer to imbue a new cast into the original video source that is animated to identically match the movement and gestures from the source.**

In this notebook we:

* Clone the First Order Motion Model repository from GitHub.
* Retrieve source images from StyleGAN2 and create driving video
* Feed these inputs to FOMM to animate the source image mapped to driving video

### First Order Motion Model

First Order Motion Models were introduced by Aliaksandr Siarohin et al. in their First Order Motion Model for Image Animation (http://papers.nips.cc/paper/8935-first-order-motion-model-for-image-animation.pdf). The power for the business case stated from the FOMM is the ability to create a video sequence that animates a source image based on motion of the driving video without the use of premade annotations or information shared about the objects to be mapped. The motion First Order Motion models are able to capture include facial expressions, head poses, and eye movements. While other frameworks required landmark detectors, and were object specific, the FOMM can animate multiple object categories, does not require an object model, and learns to only transfer motion.

### VoxCeleb Dataset

As in the case of the StyleGAN2 instance, we will use a pretrained First Order Motion Model to animate the our generated face images. This pretrained model used the VoxCeleb dataset which is comprised of face data from 22,496 videos scraped from YouTube. This particular formation of the VoxCeleb dataset contains 12,331 training videos and 444 test videos after preprocessing, which involved extracting initial bounding boxes aroung the faces within the video much like was achieved in the first notebook. The videos have all been resized to 256x256 dimensions. The distributions of country origin of the faces within these videos are shown below.

### First Order Motion Model Results

<img src='images\notebook_images\fomm_1.gif' align="center">

<img src='images\notebook_images\fomm_2.gif' align="center">


## 5. Hypothesis Testing

In bringing the work from the past notebooks to a conclusion here, we discuss the experience and impact of use of the Generate/Animate system for Netflix in our hypothetical business case. **There is great importance in understanding the public sentiment in regard to the G/A system affecting perceptions of diversity includion in media, and to this goal we conduct hypothesis testing in this final notebook.**

Within this notebook we visited several concepts:

* Made an observation and conducted research into the current trend of diversity within certain media
* Devised a hypothesis focused on the G/A system's perceived effects on diversity in media and general social stereotypes
* Demonstrated the video and survey used to mine data for analysis
* Cleaned said data and conduced exploratory data analysis on these responses
* Conducted sentiment analysis on the proceeding sentiments in regard to the G/A system
* Calculated the p-value for our experiment
* Rejected the null hypothesis and found that statistically the G/A system could effect the levels of diversity within media

#### Hypotheses

**_Null Hypothesis_** $ H_{0} : $ There is no relationship between an expected perceived increase of diversity in media and the Generate/Animate system. 

**_Alternative Hypothesis_** $ H_{1} : $ At least 7 out of 10 respondents would expect a perceived increase of diversity in media with the Generate/Animate system.

#### Alpha

Having set our hypotheses, we will need to establish the an alpha in order to set the significance threshold. Our alpha will set the distinguishing benchmark for our findings to inform us to either accept or reject the above stated null hypothesis. For our work here, the alpha value will be the generally accepted:

 - 𝛼 = $0.05$

### Data Mining

The process for collecting data was done in two parts:

- Video explanation
- Survey related to concepts

#### Video Explanation

[![Watch the video](https://img.youtube.com/vi/JcBvuPSKzsU/hqdefault.jpg)](https://youtu.be/JcBvuPSKzsU)

#### Survey

<img src='images\notebook_images\You Choose_html.png' align="center">

### Exploratory Data Analysis

<img src='images\notebook_images\sunburst.png' align="center">

### Sentiment Analysis

Sentiment Analysis, sometimes referred to as Opinion Mining is the process of understanding the opinion or emotion of an author associated with a particular subject. We will use the answers from the "How do you think this system could change general perceptions of stereotypes?" question to analyze the general sentiment towards the Generate/Animate system of the sample of respondents that answered the question. Here an opinion will be categorized into one of three categories:

* Positve
* Neutral
* Negative

Specifically, the subject of the sentiment analysis will be on the effects that the G/A system can have on general cultural stereotypes.

The opinion holder in this instance will be the 58 individuals that answered **"How do you think this system could change general perceptions of stereotypes?".**

<img src='images\notebook_images\semantic_1.png' align="center">

#### Sentiment Analysis Results

<img src='images\notebook_images\sentiment_results.png' align="center">

### WordCloud Visualization

Here we see that some of the most common words in our response are "people", 'to "see", "diversity", "AI", "media", and "think". Looking at these words alone yields the idea that the general opinion is that **AI can help people think and see diversity in media**, which is the sole purpose of this work. 

<img src='images\response_data\responses_wordcloud.png' align="center">

### Experiment Results

##### Expected Result
Claim of 7 out of 10 respondents will be of the opinion that this system can generate more diversity in media:

"In your opinion, could this format of viewing media create more diversity in the videos people see in general?"
* 7 out of 10 say yes
* 3 out of 10 say no

With 68 responses to the question, we would expect that:

* 47.6 respondents would reply- Yes
* 20.4 respondents would reply- No

##### Observed Results

<img src='images\notebook_images\results.png' align="center">

#### P-Value

As stated previously the p-value is the probability of observing a test statistic at least as large as the one observed, by random chance, assuming that the null hypothesis is true. Here we have 

$p-value=0.012864308543025699$

When compared with the significance threshold that we set we can see the comparison.

<img src='images\notebook_images\p_value.png' align="center">

## Conclusions

We were able to uncover several areas of interest as they pertained to the demographics of the sample and the opinions shared on diversity within media. Some of note include:

- This would mean that over 58% of our respondents would prefer that Netflix offered the option of using the G/A system.
- For the online learning medium we can see an increase in preference for the option of the G/A system being present, with exactly 2/3 of respondents preferring to have the system present in some form.
- Of all the individuals in our sample to answer the question, 83% believed that the G/A system could increase the amount of diversity within media.
- The groups of gender that with the highest ratio of those that feel the G/A system can increase diversity are those identifying as Gender Fluid and Gender non-conforming.
- When viewing the ethnicity category in relation to the diversity opinion, we can see again that no particular race felt that the G/A system could not increase diversity in media. While the group with the highest ratio of 'Yes' responses were identifying as other or unknown, both the 'African-American/Black' and 'Caucasian/White' groups had very high beliefs that the G/A system could increase diversity.
- Over 65% of the respondents had a positive view about the effects that the G/A system could have on the perception of stereotypes.

Overall our findings can be considered in favor of the G/A system bringing more diversity to the user base of Netflix, and to media in general. This system along with other actions could lead to an improvement in the representation of minorities in media, which could lead to countless changes in the current landscape of social relations around the world. Further more the G/A system presents us with an opportunity to view ourselves in a new light, one potentially freed from the limitations that our current culture have placed on many of us. 
At the time of this work, there have been countless occurrences of violence and hatred towards PoC and those individuals that have been classified as minorities across the globe, and especially within my home country of the United States. As someone that has been labeled as a minority, and has dealt with the under representation and misrepresentation of those that look like me, I personally feel that the G/A system offers a ray of hope that it is possible to change the way that humans as a whole interact with each other, and view each other. That all of those working to heal the wounds of the present and the past will succeed and that the end result will be a world of acceptance, appreciation, and peace not only with others, but within one's self.
