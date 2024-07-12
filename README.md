# Music Recommendation System
[![Python](https://img.shields.io/badge/Python-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Spotify](https://img.shields.io/badge/Spotify-3DDC84?style=for-the-badge&logo=spotify&logoColor=white)](https://developer.spotify.com/documentation/web-api)
[![Sklearn](https://img.shields.io/badge/Sklearn-blue?style=for-the-badge&logo=sklearn&logoColor=white)](https://scikit-learn.org/stable/)
[![Librosa](https://img.shields.io/badge/Librosa-039BE5?style=for-the-badge&logo=librosa&logoColor=white)](https://librosa.org/doc/latest/index.html)
## Overview
Welcome to our Music Genre Classifier and Recommendation System! This versatile tool utilizes machine learning to not only classify music genres based on various audio features but also recommend similar songs tailored to your preferences. Upload your audio file and discover its genre, as well as receive personalized recommendations based on your favorite tunes!
## Built using:
- [Python: ](https://www.python.org/doc/) Python documentation
- [Librosa: ](https://librosa.org/doc/latest/index.html) Librosa documentation
- [Sklearn: ](https://scikit-learn.org/stable/) Sklearn documentation
## Dataset Used:
- GTZAN Dataset
- Spotify Dataset(Manually Extracted)
## Music Recommendation System Features:
- Accurately classifies audio genres.
- Recommends Music on giving audio name

## Collaborators:
| Name | Year | Branch|
| ------------- | ------------- | ------------- |
| Salla Kaushik (B22EE058)  | Sophomore  | EE |
| Aditya Trivedi (B22CS055) | Sophomore  | CSE |
| Abhijan Thaja (B22AI025) | Sophomore  | AI |
| Rahul Reddy (B22CS041) | Sophomore  | CSE |
| Sri Ganesh (B22CS054) | Sophomore  | CSE |
| Karan Reddy (B22AI023) | Sophomore  | AI |
| Shubham Kumar (B22EE064) | Sophomore  | EE |

## Using Command Line Interface
```js
  cd App
```
### For using recommender
- train the model and save it
```js
  python recommender.py
```
- ssing the recommnder model
```js
  python Cli.py --recommend <song_name> --n <no_of_components>
```
### For using Classifier
- train the model and save it
```js
  python classifier.py
```
- using the classifier model
```js
  python Cli.py --classify <path_to_song>
```


