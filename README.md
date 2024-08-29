# Music Subgenre Classification: Techno, House, Trance, and DnB
Our primary objective was to train a variety of machine learning and deep learning models to solve the multi-label classification problem of determining a song's genre among four prominent subgenres of electronic music: techno, house, trance, and drum & bass.

# Table of Contents
1. [Introduction](#Introduction)

## Introduction
Music genres are essential for organizing and categorizing music, making it easier for listeners to discover, enjoy, and connect with styles that resonate with them. Genres also carry historical, cultural, and sonic significance. Playlists, which often focus on a single subgenre, have become an increasingly popular way to discover new music.  

We address the multi-label classification problem to identify a song's genre(s) using acoustic features extracted from audio files. We train a variety of supervised learning models to determine genre. Rather than focusing on broad genres (_e.g._, jazz, hip hop, electronic), we concentrate on four subgenres of electronic music: techno, house, trance, and drum and bass. While these subgenres are distinct and well-defined, they can be challenging to differentiate.

## Data Collection and Data Set
We used a subset of the [AcousticBrainz](https://acousticbrainz.org/) data set, which contains a total of about 7.5 million unique songs.  Each song can be identified with its unique MusicBrainz ID (MBID), which comes from [MusicBrainz](https://musicbrainz.org/), a public database consisting of metadata on music.   Due to the massive size of the data set, extracting the entire data set was not practical.  Instead, we used an API to query MusicBrainz a list of MBIDs for each subgenre, then we used another API to extract the data from AcousticBrainz with the given MBID.  Not every song in MusicBrainz has data in AcouticBrainz -- indeed roughly half of the queried songs from MusicBrainz had corresponding data in AcousticBrainz.  After preliminary data cleaning, our data set had about 37,000 data points and about 2,300 acoustic features.

AcousticBrainz does not store any audio files.  Rather, audio characteristics, such as loudness, dynamics, spectrum, beats, and chords, are extracted using [Essentia](https://essentia.upf.edu/streaming_extractor_music.html#music-descriptors) and stored in AcousticBrainz.  Many features are split based on bands of frequency, and then various statistics within each band (e.g. mean, variance), resulting in the many features.  Genre labels were obtained from MusicBrainz, which are user submitted and then voted by the community.  We consider the genre labels to be reasonably accurate.

## Exploratory Data Analysis
We did some initial data cleaning including:
- Removing duplicates based on MBID, followed by artist id and song title.
- Removing songs with all four genre labels.  We concluded that these were likely inaccurately tagged.
- Some data points were part of audiobooks, which we removed.
- Some features were trivial (e.g. were all the same value), which we removed.
- Some features were naturally scalar-valued, while others were vectors, i.e., lists—particularly those measured across various frequency bands. We expanded these vector features, creating a separate feature for each dimension.  **This resulted in roughly 2500 features.**
  
The metadata from MetaBrainz contains a 'tag' attribute, which contains the genre labels.  If any of the tags contained 'techno', 'house', 'trance', and/ or 'drum and bass' as a substring, then we labelled the song with that genre.

We did some preliminary data analysis to get a rough idea of how our data looked.  However, due to the large amount of features, it was challenging to get a detailed sense of them.

Below illustrates a histogram of the means and variances of each standardized feature for each genre.
<img src = "https://github.com/user-attachments/assets/59340f6f-a7ec-4c23-b8e6-83728fb8abb7" width = 500>
<img src = "https://github.com/user-attachments/assets/f43ec3f4-63f1-4aee-b4f3-ff16ece9caf4" width = 500>

We see that techno has greater mixture of variances and a flatter distribution of means, indicating that techno may be more diverse sonically while house has means concentrated closer to 0 and variances closer to 1, indicating that house may sound more 'generic'.

To reduce features, we used a mixture of correlation analysis and PCA.  We considered a few different versions of the dataset with reduced features.  We outline the version we trained most of our models with:
1. We removed all features with correlation above 0.90.
2. We split the features into two groups: features which were initially scalar valued and features which were initially vector valued.  Among the vector valued ones, we further grouped the features based on if they were describing similar acoustic information (e.g. chords, melbands, loudness, etc...).
3. We performed PCA on each of the groups, accounting for 90% of the variation.
4. The new features coming from PCA along with the scalar features comprised the new set of **476 features**.

## Modeling Approach

We first tried a few standard machine learning models.  The following table summarizes our accuracies for each genre class:
|                 | Trance   | House  | Techno | Drum and Bass |  
| ---------       | -------- | ------ | ------ | ------------- |
| Dummy Stratified|   0.591  | 0.524  | 0.583  |   0.633       |
|      kNN        |   0.73   |  0.68  |  0.80  | 0.81          |
| Gaussian NB     |   0.72   |  0.49  | 0.49   |    0.81       |
| Bernoulli NB    |   0.74   | 0.68   | 0.80   |  0.84         |
| Random Forest   |   0.854  | 0.758  |  0.785 | 0.887         |
|     **XGBoost**     |  **0.883**   | **0.789**  | **0.803**  |  **0.909**        |

We can see that XGBoost outperforms all other models for each genre.  The following table summarizes the F1 score for each genre class:

|                 | Trance   | House  | Techno | Drum and Bass |  
| ---------       | -------- | ------ | ------ | ------------- |
| Dummy Stratified|   0.28  | 0.41  | 0.30  |   0.24       |
|      kNN        |   0.18   |  0.57  |  **0.68**  | 0.44          |
| Gaussian NB     |   0.41   |  0.57  | 0.50   |    0.58       |
| Bernoulli NB    |   0.47   | 0.56   | 0.62   |  0.63         |
| Random Forest   |  0.68   | 0.64  | 0.53  |  0.71        |
|  **XGBoost**    |  **0.77**   | **0.72**  | 0.60  |  **0.79**        |

We can see that XGBoost outperforms in most except for Techno.  Please refer to the model notebooks for results on recall and precision.

We used a neural network with hidden layers of sizes 500, 100, and 20. The output layer was of size 4. The input layer was of size 2614, the number of numerical features available after deleting features which had constant values throughout the data. The input data is first normalized using the calculated mean and standard deviation of the training data, before being input into the model.

The training data was first split into a training set of size 30000 and a test set of size 6885. The neural network was trained using a gradient descent algorithm with a learning rate of 0.005 and a momentum of 0.9. The batch size during training was 128, to help speed up optimization, and the loss was calculated using the binary cross entropy loss. The training took place over 50 epochs and took approximately 24.5 seconds for all 50 epochs.

We then performed a 5-fold cross validation on the training set and found that each iteration gave similar accuracy scores. The average scores after cross validation are:

|           | Trance   | House  | Techno | Drum and Bass |
| --------- | -------- | ------ | ------ | ------------- |
| Accuracy  | 0.835    | 0.733  | 0.787  | 0.911         |
| Recall    | 0.712    | 0.684  | 0.678  | 0.842         |
| Precision | 0.742    | 0.656  | 0.566  | 0.777         |

Further training on the test set did not significantly improve the accuracy scores of the model. The model was finally trained on the full training set and then evaluated on the test set:

|           | Trance   | House  | Techno | Drum and Bass |
| --------- | -------- | ------ | ------ | ------------- |
| Accuracy  | 0.863    | 0.747  | 0.798  | 0.919         |
| Recall    | 0.760    | 0.682  | 0.642  | 0.839         |
| Precision | 0.747    | 0.696  | 0.664  | 0.838         |

The dataset with 476 features detailed in the Exploratory Data Analysis section was also used to train a model with the same sizes of hidden layers, only differing in the input layer which now has size 476. This model was trained, as detailed previously, on 30000 samples and tested on the remaining 6885 samples. The results were:

|           | Trance   | House  | Techno | Drum and Bass |
| --------- | -------- | ------ | ------ | ------------- |
| Accuracy  | 0.801    | 0.751  | 0.861  | 0.891         |
| Recall    | 0.711    | 0.708  | 0.807  | 0.811         |
| Precision | 0.547    | 0.641  | 0.678  | 0.714         |

## Conclusion and Future Directions

## Description of Repository
The building_AcousticBrainz_dataset.ipynb notebook illustrates how we use the MusicBrainz and AcousticBrainz API to create our dataset.  The dataset is presented as a folder of json files; each file represents a song.

The Cleaning_and_EDA folder contains scripts and notebooks showing how we built our dataset from these json files into a pandas DataFrame.  In addition, it also demonstrates how cleaning process and exploratory data analysis we considered.

The Models folder contains models.
