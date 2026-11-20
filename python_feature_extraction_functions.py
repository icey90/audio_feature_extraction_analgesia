"""
Python functions for extracting audio features.
Author: Isabella Czedik-Eysenberg

These functions were used for feature extraction in the manuscript:
Maidhof et al. (2026). "The role of audio features for music-induced
analgesia".

The module uses the following audio feature extraction libraries:

LibROSA:
McFee, B. et al. librosa: Audio and music signal analysis in python. SciPy 2015, 7 (2015).

Bogdanov, D. et al. in Britto A, Gouyon F, Dixon S, editors. 14th Conference of the International Society for Music Information Retrieval (ISMIR); 2013 Nov 4-8; Curitiba, Brazil. ISMIR; 2013. p. 493-8. (International Society for Music Information Retrieval (ISMIR)).

Essentia Tensorflow Models:
Alonso-Jiménez, P., Bogdanov, D., Pons, J. & Serra, X. in ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).  266–270 (IEEE).
 
AudioCommons Timbral Models:
Pearce, A. et al. AudioCommons: D5. 8 Release of timbral characterisation tools for semantically annotating non-musical content. (AudioCommons Initiative, Tech. Report, 2019).

"""


sr_default = 44100

def extract_timbral_hardness(sig, sr=sr_default):
    """
    Extract timbral hardness using AudioCommons Timbral Models.

    Parameters
    ----------
    sig : numpy.ndarray
        One-dimensional mono audio signal.
    sr : int, optional
        Sampling rate of 'sig' in Hz. The default is 44100 Hz.

    Returns
    -------
    float
        Timbral hardness estimate returned by
        'timbral_models.timbral_hardness'.

    Notes
    -----
    This function requires the 'timbral_models' Python package.
    """
    
    value = timbral_models.timbral_hardness(sig, fs=sr)
    return value


def extract_perc_harm_ratio(sig, sr=sr_default):
    """
    Calculate the ratio of percussive to harmonic signal energy.

    Harmonic and percussive components are estimated using librosa's
    harmonic-percussive source separation (HPSS). The feature is calculated
    as the ratio between the mean RMS energy of the percussive component and
    the mean RMS energy of the harmonic component.

    Parameters
    ----------
    sig : numpy.ndarray
        One-dimensional mono audio signal.
    sr : int, optional
        Sampling rate of 'sig' in Hz. This parameter is retained for a
        consistent function interface but is not directly used in the current
        calculation.

    Returns
    -------
    float
        Mean RMS energy of the percussive component divided by mean RMS
        energy of the harmonic component.

    Notes
    -----
    If the harmonic component has zero mean RMS energy, the result may be
    infinite or undefined.
    """
    
    # Separate harmonic and percussive components
    sig_harmonic, sig_percussive = librosa.effects.hpss(y=sig)
        
    # Calculate ratio between RMS of harmonic and percussive component
    perc_energy = np.mean(librosa.feature.rms(y=sig_percussive))
    harm_energy = np.mean(librosa.feature.rms(y=sig_harmonic))
    perc_harm_ratio = perc_energy / harm_energy
        
    return perc_harm_ratio


def calc_spectral_contrast_librosa(sig, sr=sr_default):
    """
    Calculate mean spectral contrast using librosa.

    Spectral contrast measures the difference in energy between spectral peaks
    and valleys across frequency bands. Mean spectral contrast is calculated
    separately for each frequency band across all analysis frames.

    Parameters
    ----------
    sig : numpy.ndarray
        One-dimensional mono audio signal.
    sr : int, optional
        Sampling rate of 'sig' in Hz. The default is 44100 Hz.

    Returns
    -------
    numpy.ndarray
        One-dimensional array containing mean spectral contrast values for
        the frequency bands returned by librosa.
    """
    
    feature_values = librosa.feature.spectral_contrast(y=sig, sr=sr)
    feature_means = np.mean(feature_values, 1)
    return feature_means
    
    
def extract_spectral_flatness_librosa(sig):
    """
    Calculate mean spectral flatness using librosa.
    Spectral flatness describes the degree to which a spectrum resembles a
    flat, noise-like spectrum rather than a tonal spectrum. The function
    calculates spectral flatness frame-wise and returns its mean across frames.

    Parameters
    ----------
    sig : numpy.ndarray
        One-dimensional mono audio signal.

    Returns
    -------
    float
        Mean spectral flatness across analysis frames.
    """
        
    feature_values = librosa.feature.spectral_flatness(y=sig)     
    feature_mean = np.mean(feature_values)
    return feature_mean

    
def extract_spectral_rolloff_librosa(sig, sr=sr_default):
    """
    Calculate mean spectral roll-off using librosa.

    Spectral roll-off is the frequency below which a specified proportion of
    the total spectral energy is contained. librosa uses its default roll-off
    proportion unless otherwise configured. The function returns the mean
    roll-off frequency across analysis frames.

    Parameters
    ----------
    sig : numpy.ndarray
        One-dimensional mono audio signal.
    sr : int, optional
        Sampling rate of 'sig' in Hz. The default is 44100 Hz.

    Returns
    -------
    float
        Mean spectral roll-off frequency in Hz.
    """
    
    feature_values = librosa.feature.spectral_rolloff(y=sig, sr=sr)     
    feature_mean = np.mean(feature_values)
    return feature_mean


def extract_essentia_features(filepath):
""" Extract audio features using the Essentia MusicExtractor.
    The function returns an Essentia Pool object containing the
    extracted descriptors.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to an audio file readable by Essentia.

    Returns
    -------
    essentia.Pool
        Pool containing the extracted Essentia descriptors.
    """
    essMusicExtractor = ess.MusicExtractor(lowlevelStats=["mean"])
    ess_features, features_frames = essMusicExtractor(filepath)
    return ess_features
    

def extract_essentia_models_jamendo_instrument_discogs_effnet(filepath, instr_index):
    """ Extract classification of music by instrument from discogs-effnet embeddings.
    The function first generates Discogs-EffNet embeddings for an audio file.
    It then applies the MTG-Jamendo instrument-classification model and
    returns the mean predicted score for one selected instrument class over
    all model output frames.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to an audio file readable by Essentia.
    instr_index : int
        Index of the instrument class in the model output. The mapping from
        indices to instrument labels is specified in the model metadata:

        https://essentia.upf.edu/models/classification-heads/
        mtg_jamendo_instrument/
        mtg_jamendo_instrument-discogs-effnet-1.json

    Returns
    -------
    float
        Mean predicted score for the requested instrument class. If feature
        extraction or model inference fails, 'numpy.nan' is returned.
    
    Notes
    -----
    This function expects the following Essentia TensorFlow model files to be
    present relative to the working directory:

    - 'models_essentia/discogs-effnet-bs64-1.pb'
    - 'models_essentia/mtg_jamendo_instrument-discogs-effnet-1.pb'

    The input audio is internally resampled to 16000 Hz. The model files are
    not included in this code file and are subject to their own licence terms.
    """
    
    curr_values = {}
    try:
        audio = MonoLoader(filename=filepath, sampleRate=16000, resampleQuality=4)()
        embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="models_essentia/discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
        embeddings = embedding_model(audio)
    
        model = TensorflowPredict2D(graphFilename="models_essentia/mtg_jamendo_instrument-discogs-effnet-1.pb")
        predictions = model(embeddings)
    
        value = np.mean(predictions, axis=0)[instr_index]
    except Exception as e:
        print(f"Error extracting the value, setting NaN-values: {e}")
        value = np.nan
    return value
