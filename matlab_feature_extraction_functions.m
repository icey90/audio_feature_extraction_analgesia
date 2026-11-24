% Feature extraction functions based on MIRtoolbox (Lartillot et al., 2008),
% used for replicating results reported in Knox et al. (2011),
% as well as exploratory analyses, as reported in manuscript:
% Anonymized et al. (2026): "The role of audio features for music-induced analgesia"
% Author: Anonymized
%
% MIRtoolbox: Lartillot, O., Toiviainen, P., & Eerola, T. (2008).
% A matlab toolbox for music information retrieval. In Data Analysis, 
% Machine Learning and Applications: Proceedings of the 31st Annual 
% Conference of the Gesellschaft für Klassifikation eV,
% Albert-Ludwigs-Universität Freiburg, March 7–9, 2007 (pp. 261-268).
% Berlin, Heidelberg: Springer Berlin Heidelberg.
%
% Knox, D., Beveridge, S., Mitchell, L. A., & MacDonald, R. A. (2011).
% Acoustic analysis and mood classification of pain-relieving music.
% The Journal of the Acoustical Society of America, 130(3), 1673-1682.
%

% ------------------------------------------------------------------
% Features replicating Knox et al. (2011):
% ------------------------------------------------------------------

% Chromagram peak position standard deviation
function value = extract_chromagramPeakPositionStd(filepath)
    feat = mirfeatures(filepath, 'stat');
    value = feat.tonal.chromagram.peak.PeakPosStd;
end

% HCDF periodicity entropy
function value = extract_hcdfPeriodicityEntropy(filepath)
    feat = mirfeatures(filepath, 'stat');
    value = feat.tonal.hcdf.PeriodEntropy;
end

% Mel-frequency cepstral coefficients (MFCC)
function value = extract_mfcc(filepath)
    value = mirgetdata(mirmfcc(filepath));
end
        
% Extract standard deviation of each MFCC
function value = extract_mfccStd(filepath)
    value = mirgetdata(mirstd(mirmfcc(filepath, 'Frame')));
end

% ------------------------------------------------------------------
% Further exploratory features:
% ------------------------------------------------------------------

% Spectral centroid
function value = extract_spectralCentroid(filepath)
    value = mirgetdata(mircentroid(filepath));
end

% Brightness
function value = extract_brightnessHighFrequency(filepath)
    value = mirgetdata(mirbrightness(filepath));
end

% Envelope flatness
function value = extract_envelopeFlatness(filepath)
    value = mirgetdata(mirflatness(mirenvelope(filepath)));
end

% Metrical centroid, see:
% Lartillot, O., Cereghetti, D., Eliard, K., Trost, W. J., Rappaz, M.-A.,
% Grandjean, D., "Estimating tempo and metrical features by tracking 
% the whole metrical hierarchy", 3rd International Conference on 
% Music & Emotion, Jyvaskyla, 2013.
function value = extract_tempoMetroid(filepath)
    value = median(mirgetdata(mirmetroid(filepath)));
end

% Onset rate (per minute)
function value = extract_tempoOnsets(filepath)
    onsets = mirgetdata(mironsets(filepath));
    duration = mirgetdata(mirlength(filepath));
    onsetsPerSecond = size(onsets, 1) / duration;
    value = onsetsPerSecond * 60; % convert to onsets/minute (similar to bpm)
end

% RMS energy of each band after gammatone filtering
function value = extract_rmsGammatone(filepath)
    rmsBands = mirgetdata(mirrms(mirfilterbank(filepath, 'Gammatone')));
    value = reshape(rmsBands, [length(rmsBands), 1]);
end

% Mean Inharmonicity
function value = extract_inharmonicity(filepath)
    try
        value = mirgetdata(mirinharmonicity(filepath));
    catch
        value = nanmean(mirgetdata(mirinharmonicity(filepath, 'Frame')));
    end
end

% Mean Roughness (Sethares)
function value = extract_roughnessSetharesMean(filepath)
    % Trim pseudo-silence from beginning and end:
    trimmed = miraudio(filepath, 'Trim');
            
    roughness_values = mirgetdata(mirroughness(trimmed));
    value = mean(roughness_values);
end

