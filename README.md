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
- 

## Modeling Approach

## Conclusion and Future Directions

## Description of Repository
