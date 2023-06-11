# Exoplanet-Detection-using-CNN
The project aims to leverage machine learning and deep learning techniques to analyse the flux data and accurately classify stars as either exoplanet-stars or non-exoplanet-stars. By training a model on the provided dataset, we seek to uncover patterns and features indicative of exoplanet presence, enabling the model to make predictions on unseen data.

![Keras](https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Pr0-C0der/exoplanet-detection/blob/main/LICENSE)

[![Linkedin Badge](https://img.shields.io/badge/-LinkedIn-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/prathamesh-gadekar-b7352b245/)](https://www.linkedin.com/in/prathamesh-gadekar-b7352b245/)
[![Hotmail Badge](https://img.shields.io/badge/-Hotmail-0078D4?style=flat-square&logo=microsoft-outlook&logoColor=white&link=mailto:prathamesh.gadekar@hotmail.com)](mailto:prathamesh.gadekar@hotmail.com)

## Table of Contents

- [What is Exoplanet](#what-is-exoplanet)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Contributing](#contributing)
- [License](#license)


# What is Exoplanet?
![Exoplanets](https://exoplanets.nasa.gov/rails/active_storage/blobs/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBaWtPIiwiZXhwIjpudWxsLCJwdXIiOiJibG9iX2lkIn19--a88308edf835b0b6d9d4399ffa1483554e9c3f64/Exoplanet_types_graphic.jpg?disposition=attachment)
[Image Source](https://exoplanets.nasa.gov/resources/2253/exoplanet-types-graphic/)

An exoplanet, or extrasolar planet, is a planet that orbits a star outside of our solar system. These celestial bodies are of great scientific interest as they provide valuable insights into the formation, composition, and diversity of planetary systems beyond our own. Exoplanets can vary in size, composition, and orbital characteristics, ranging from gas giants to rocky planets. Their detection is achieved through various indirect methods, such as observing the transit of a planet in front of its host star or measuring the gravitational influence on the star. The study of exoplanets plays a crucial role in advancing our understanding of planetary systems and the potential for extraterrestrial life.

# Methods for Detecting Exoplanets

- Indirect methods:
  - Transit method: Observing periodic dimming of a star's light as a planet passes in front of it.
  - Radial velocity method: Detecting the wobble of a star caused by the gravitational pull of an orbiting planet.
  - Gravitational microlensing: Measuring the bending of light due to a planet's gravity.
  - Astrometry: Detecting tiny changes in a star's position caused by an orbiting planet.
- Direct imaging: Capturing the actual light emitted or reflected by the exoplanet, although challenging due to the brightness of the host star.

# Transit Method used for Exoplanet Detection
![Light Curve](https://exoplanets.nasa.gov/rails/active_storage/blobs/eyJfcmFpbHMiOnsibWVzc2FnZSI6IkJBaHBBajBNIiwiZXhwIjpudWxsLCJwdXIiOiJibG9iX2lkIn19--c103b7858d358b954674f60aeedf8e5ba479e4bb/656348main_ToV_transit_diag.jpg?disposition=attachment)

[Image Source](https://exoplanets.nasa.gov/resources/280/light-curve-of-a-planet-transiting-its-star/)

Flux is a crucial parameter used in the detection and characterization of exoplanets. Flux is a measure of the number of electric or magnetic field lines passing through a surface in a given amount time. By monitoring the flux, which represents the light intensity emitted by a star, astronomers can identify subtle changes that indicate the presence of an exoplanet. The transit method relies on observing periodic dips in flux as an exoplanet passes in front of its host star, causing a temporary decrease in the observed light. Additionally, the radial velocity method measures the small shifts in spectral lines caused by the gravitational tug of an exoplanet, resulting in periodic variations in flux. Analyzing these flux variations provides valuable information about the presence, size, and orbital characteristics of exoplanets.

# Dataset Description
The dataset for the following project was collected by the NASA Kepler space telescope using the Transit method. By closely observing a star over extended periods, ranging from months to years, scientists can detect regular variations in the light intensity. These variations, known as "dimming," serve as evidence of the presence of an orbiting body around the star. Such stars exhibiting dimming can be considered potential exoplanet candidates. However, further study and investigation are required to confirm the existence of exoplanets. For example, employing satellites that capture light at different wavelengths can provide additional data to solidify the belief that a candidate system indeed harbors exoplanets.

The dataset provided is divided into Training and Testing data. The data describe the change in flux (light intensity) of several thousand stars. Each star has a binary label of 2 or 1. 2 indicated that that the star is confirmed to have at least one exoplanet in orbit; some observations are in fact multi-planet systems.

- Trainset:
  - 5087 rows or observations.
  - 37 confirmed exoplanet-stars and 5050 non-exoplanet-stars.
- Testset:
  - 570 rows or observations.
  - 5 confirmed exoplanet-stars and 565 non-exoplanet-stars.

# Exploratory Data Analysis

Using the flux values, we plot the waves with respect to time for both exoplanet-stars and non-exoplanet-stars.

![Waves with respect to time](https://github.com/Pr0-C0der/Exoplanet-Detection-using-CNN/assets/93116210/3b3dbdf9-ef29-46ee-92d5-cfdd0536bbb0)

By plotting the pairplot for first flux values, we can see that each one of them is highly correlated. 
![Pair Plot](https://github.com/Pr0-C0der/Exoplanet-Detection-using-CNN/assets/93116210/05c734a2-f55e-471a-a56d-9730f3dc5a1f)

By observing the below distribution, we can conclude the same. Since the dataset is highly imbalanced, the distribution of exoplanet-stars is barely visible. Hence we have highlighted it using blue ink.

![Probability Distribution Function](https://github.com/Pr0-C0der/Exoplanet-Detection-using-CNN/assets/93116210/bfffc11f-ad30-41a3-a6bc-224355f6b4b3)

# Data Preprocessing
In the data pre-processing phase, several steps are taken to prepare the dataset for the exoplanet detection project. 
1. Firstly, to address the issue of data imbalance, outliers are removed from the dataset. As the data contains a high imbalance between the number of exoplanet and non-exoplanet instances, this step helps in creating a more balanced representation of the classes. 
2. Secondly, to further handle the data imbalance, a technique called Random Over Sampler is employed, which increases the number of minority class instances through random duplication. This helps in improving the learning process and the performance of the models. 
3. Lastly, the labels in the dataset are transformed from 1 and 2 to 0 and 1, respectively, to ensure a consistent binary representation. 

The decision to not perform data scaling was taken while testing the models trained using scaled data. From the observations we found out that using the raw data produced better results compared to scaled data.

