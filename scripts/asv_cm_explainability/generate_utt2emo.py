"""
Generate utterance to emotion mapping via SSL model
"""

import os
import glob

import torch
from speechbrain.inference.interfaces import foreign_class


class EmotionClassifier(object):
    def __init__(self):
        self.classifier = foreign_class(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
        )
        self.classifier.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classifier = self.classifier.to(device)

    def classify_emotion(self, audio_path):
        # Load the audio file
        out_prob, score, index, text_lab = self.classifier.classify_file(audio_path)
        emotion = text_lab[0]
        return emotion


if __name__ == "__main__":
    asvspoof_wav_dir = "../synthetiq-audio-io-board/data/Database/LA"

    # Apart from the metadata statiscal attributes, we also would like to
    # investigate emotion. We generate labels using speechbrain and add
    # the labels to the metadata
    emotion_recognizer = EmotionClassifier()

    asvspoof_wavs = glob.glob(asvspoof_wav_dir + "/*/*/*.flac")

    with open("data/Database/ASVspoof_VCTK_VCC_MetaInfo/utt2emo", "w") as u:
        for wav_file in asvspoof_wavs:
            utt_name = os.path.basename(wav_file).split(".")[0]
            emotion = emotion_recognizer.classify_emotion(wav_file)
            u.write("{} {}\n".format(utt_name, emotion))
