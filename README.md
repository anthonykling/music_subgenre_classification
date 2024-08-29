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


## Modeling Approach

## Conclusion and Future Directions

## Description of Repository
The building_AcousticBrainz_dataset.ipynb notebook illustrates how we use the MusicBrainz and AcousticBrainz API to create our dataset.  The dataset is presented as a folder of json files; each file represents a song.

The Cleaning_and_EDA folder contains scripts and notebooks showing how we built our dataset from these json files into a pandas DataFrame.  In addition, it also demonstrates how cleaning process and exploratory data analysis we considered.

The Models folder contains models.
