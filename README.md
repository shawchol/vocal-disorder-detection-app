# Vocal Disorder Detection (Streamlit app)

A web app that screens a short voice recording for vocal disorders using a
dual-branch CNN-BiLSTM-Attention model. Upload a `.wav` or `.mp3` file and the app
returns the predicted class, confidence scores, and waveform / MFCC visualizations.

**Modes**
- Multi-class: Healthy, Laryngozele, Vox senilis
- Binary: Healthy vs Diseased

> **Disclaimer:** This is a university capstone project and a screening aid only.
> It is not clinically validated and must not be used for diagnosis.

## Dataset
Voice recordings from the Kaggle dataset
[Patient Health Detection using Vocal Audio](https://www.kaggle.com/datasets/subhajournal/patient-health-detection-using-vocal-audio)
(referred to as the SVD dataset in the app). The dataset is not included in this repo.

## How it works
1. Audio is loaded as 16 kHz mono, cut or zero-padded to 3 seconds, and peak-normalized.
2. Two inputs are built from the signal:
   - a 40 x 174 MFCC map (CNN-BiLSTM-Attention branch)
   - a 390-dimensional handcrafted feature vector: MFCC, mel-spectrogram, chroma,
     tonnetz and spectral-contrast statistics (mean and std), plus ZCR and RMS,
     standardized with `scaler.pkl`
3. The dual-branch model outputs class probabilities (softmax).

The app sidebar reports about 97.73% accuracy from 5-fold cross-validation during training.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit app (preprocessing, feature extraction, inference, plots) |
| `best_dual_branch_overall.h5` | 3-class model |
| `best_binary_overall.h5` | Binary model |
| `scaler.pkl` | Feature scaler for the handcrafted branch |
| `colab_deploy_notebook.ipynb` | Colab notebook used to deploy the app with ngrok |
| `requirements.txt` | Python dependencies |

## Run locally
Python 3.10 or 3.11 is recommended (TensorFlow 2.16.1).

```bash
pip install -r requirements.txt
streamlit run app.py
```

Keep the two `.h5` files and `scaler.pkl` in the same folder as `app.py`.

If `scaler.pkl` fails to load, the scikit-learn version differs from the one used to
create it; install the version used in the training environment.

## Run on Google Colab
Open `colab_deploy_notebook.ipynb`, upload the model files and `scaler.pkl`, and add your
own ngrok auth token (use Colab Secrets; do not paste tokens into the notebook).

## Limitations
- Trained and tested on a single public dataset; performance on other microphones,
  languages or recording conditions is unknown.
- Only 3 seconds of audio are analyzed per recording.
- Screening aid only, not a medical device.
