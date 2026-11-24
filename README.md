# Audio feature extraction code for manuscript "The role of audio features for music-induced analgesia"

This repository contains custom MATLAB and Python functions used to extract
audio features for the manuscript:

> *The role of audio features for music-induced analgesia*. [under review].

The functions were used to extract acoustic and music-related audio features
for analyses of music-induced analgesia. They include selected feature
implementations intended to reproduce measures reported by Knox et al. (2011)
as well as further exploratory audio features.

## Scope of this repository

This repository contains the custom feature-extraction functions used in the
study. It does not contain:

- the audio stimuli;
- participant-level data;
- third-party software packages or pre-trained model files.

The complete batch-processing code is not included because it consisted primarily of
study-specific file handling and iteration over the audio collection. Users may
apply the provided functions within their own batch-processing workflow.

## Repository contents

```text
matlab_feature_extraction_functions.m
    MATLAB functions based primarily on MIRtoolbox.

python_feature_extraction_functions.py
    Python functions based on librosa, Essentia, and AudioCommons Timbral
    Models.

requirements.txt
    Python package versions required for the Python functions.
```

## Requirements

### MATLAB functions

The MATLAB functions require:

- MATLAB [e.g. R2023b];
- MIRtoolbox [v1.7.2].

MIRtoolbox must be installed separately and added to the MATLAB path before
using the functions.

For example:

```matlab
addpath(genpath('/path/to/MIRtoolbox'));
```

### Important note on the MATLAB function collection

The file `matlab_feature_extraction_functions.m` contains multiple MATLAB
function definitions in a single file. It is provided as a collection of the
feature-extraction functions used in this study.

To call an individual function directly from the MATLAB command window or from
another MATLAB script, users may need to copy the relevant function into a
separate `.m` file whose filename matches the function name.

For example, to call:

```matlab
extract_spectralCentroid(filepath)
```

the corresponding function can be saved as:

```text
extract_spectralCentroid.m
```

This repository retains the functions together in one file to document the
original feature-extraction implementation.

### Python functions

The Python functions were developed using:

- NumPy;
- librosa;
- Essentia;
- AudioCommons Timbral Models.

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

The exact package versions used for the study should be specified in
`requirements.txt`.

Depending on the operating system and Python installation, Essentia may require
a platform-specific installation procedure. Please consult the official
Essentia installation documentation if needed.

## Input audio

The functions expect either:

- a path to an audio file (`filepath`); or
- a one-dimensional mono audio signal stored as a NumPy array (`sig`).

For Python functions that accept an audio signal, audio can for example be
loaded using librosa:

```python
import librosa
import python_feature_extraction_functions as features

sig, sr = librosa.load(
    "path/to/audio.wav",
    sr=44100,
    mono=True,
)

hardness = features.extract_timbral_hardness(sig, sr)
```

The default sampling rate used by relevant Python functions is 44100 Hz.
Users should ensure that the sampling rate supplied to each function matches
the sampling rate of the audio signal.

The Essentia instrument-classification function resamples audio internally to
16000 Hz.

## MATLAB feature extraction functions

The MATLAB function collection uses MIRtoolbox and contains functions for
extracting the following features:

- chromagram peak-position standard deviation;
- HCDF periodicity entropy;
- mel-frequency cepstral coefficients (MFCCs);
- standard deviations of MFCCs;
- spectral centroid;
- brightness;
- envelope flatness;
- metrical centroid;
- onset rate;
- RMS energy after gammatone filtering;
- inharmonicity;
- roughness (Sethares).

Several functions were implemented to reproduce selected audio features
described by Knox et al. (2011). Further features were used in exploratory
analyses.

## Python feature extraction functions

The Python function collection includes:

- timbral hardness using AudioCommons Timbral Models;
- percussive-to-harmonic energy ratio;
- spectral contrast;
- spectral flatness;
- spectral roll-off;
- descriptors extracted using Essentia MusicExtractor;
- instrument-classification scores based on Discogs-EffNet embeddings and the
  MTG-Jamendo instrument-classification model.

Example usage:

```python
import librosa
import python_feature_extraction_functions as features

sig, sr = librosa.load(
    "path/to/audio.wav",
    sr=44100,
    mono=True,
)

timbral_hardness = features.extract_timbral_hardness(sig, sr)
perc_harm_ratio = features.extract_perc_harm_ratio(sig, sr)
spectral_contrast = features.calc_spectral_contrast_librosa(sig, sr)
spectral_flatness = features.extract_spectral_flatness_librosa(sig)
spectral_rolloff = features.extract_spectral_rolloff_librosa(sig, sr)
```

## Essentia MusicExtractor

The function:

```python
extract_essentia_features(filepath)
```

uses Essentia's `MusicExtractor` with:

```python
lowlevelStats=["mean"]
```

This configuration requests mean statistics for a selection of music descriptors. 
The function returns an Essentia `Pool` object containing the extracted
descriptors.

Example:

```python
import python_feature_extraction_functions as features

essentia_features = features.extract_essentia_features(
    "path/to/audio.wav"
)

print(essentia_features.descriptorNames())
```

## Essentia instrument-classification models

The function:

```python
extract_essentia_models_jamendo_instrument_discogs_effnet(
    filepath,
    instr_index,
)
```

requires the following Essentia TensorFlow model files:

```text
models_essentia/discogs-effnet-bs64-1.pb
models_essentia/mtg_jamendo_instrument-discogs-effnet-1.pb
```

These files are not distributed with this repository. Users should download
them from the official Essentia model repository or model documentation and
place them in the `models_essentia/` directory.

The mapping between `instr_index` values and instrument labels is provided in
the model metadata:

https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1.json

Before downloading, using, or redistributing model files, users must check and
comply with the licence terms applicable to the respective model files.

## Audio materials and data availability

The audio materials used in the study are not included in this repository
because they are subject to copyright and third-party licensing restrictions.

No participant-level data are included in this repository.

## Reproducibility notes

This repository provides the custom feature-extraction functions used in the
study. Exact numerical reproduction of the study results requires access to:

1. the same source audio files;
2. the relevant third-party software and model files;
3. the software versions specified in this repository; and
4. the study-specific processing, feature-selection, and statistical-analysis
   workflow.

The functions provided here are intended to document the custom audio-feature
extraction procedures and to enable their reuse in other workflows.

## Software and third-party dependencies

This repository contains original code written for the study. It does not
redistribute MATLAB, MIRtoolbox, NumPy, librosa, Essentia, AudioCommons Timbral
Models, or Essentia TensorFlow model files.

These software packages and model files are subject to their respective
licences and terms of use. Users are responsible for installing and using all
dependencies in accordance with the applicable licence conditions.

Relevant software and methodological references include:

- Lartillot, O., Toiviainen, P., & Eerola, T. (2008). A MATLAB toolbox for
  music information retrieval. In *Data Analysis, Machine Learning and
  Applications* (pp. 261–268). Springer.

- Knox, D., Beveridge, S., Mitchell, L. A., & MacDonald, R. A. (2011).
  Acoustic analysis and mood classification of pain-relieving music.
  *The Journal of the Acoustical Society of America, 130*(3), 1673–1682.

- McFee, B., et al. (2015). librosa: Audio and music signal analysis in
  Python. In *Proceedings of the 14th Python in Science Conference*
  (pp. 18–25).

- Bogdanov, D., et al. (2013). ESSENTIA: An audio analysis library for music
  information retrieval. In *Proceedings of the 14th International Society
  for Music Information Retrieval Conference* (pp. 493–498).

- Alonso-Jiménez, P., Bogdanov, D., Pons, J., & Serra, X. (2020).
  Tensorflow audio models in Essentia. In *Proceedings of ICASSP 2020*
  (pp. 266–270).

- Pearce, A., et al. (2019). *AudioCommons: D5.8 Release of timbral
  characterisation tools for semantically annotating non-musical content*.
  AudioCommons Initiative Technical Report.

## Citation

If you use this code, please cite the associated manuscript:

> Anonymized et al. (2026). *The role of audio features for music-induced
> analgesia*. [Journal after publication].

Please also cite the relevant third-party software and methodological
references listed above.

The archived version of this code associated with the publication is:

> Anonymized (2026). *Audio feature extraction code for
> music-induced analgesia* (Version 1.0.0). Zenodo.
> https://doi.org/[ZENODO-DOI]

## Licence

The original code in this repository is made available under the MIT License.

Third-party dependencies and model files are not covered by the MIT License
and remain subject to their respective licences and terms of use.
