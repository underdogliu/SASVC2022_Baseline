"""
Speaking rate estimator based on whisper and parselmouth

We need below toolkits to be installed:
    !pip install praat-parselmouth
    !pip install openai-whisper
"""

import os
import sys
import glob

import whisper
import parselmouth


def transcribe_audio_with_whisper(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]


def compute_speaking_rate_with_whisper(audio_path):
    # Transcribe the audio using Whisper
    transcription = transcribe_audio_with_whisper(audio_path)

    # Load the sound file
    snd = parselmouth.Sound(audio_path)

    # Split the transcription into words
    words = transcription.split()
    word_count = len(words)

    # Get the total duration of the audio
    total_duration = snd.get_total_duration()

    # Calculate the speaking rate (time per word)
    if word_count > 0:
        speaking_rate = total_duration / word_count
        print(f"Speaking rate: {round(speaking_rate, 4)} seconds per word")
        return round(speaking_rate, 4)
    else:
        print("No words found in the transcription.")
        return 0.0


if __name__ == "__main__":
    data_dir = sys.argv[1]
    flac_files = glob.glob(data_dir + "/trimmed/*.wav")

    with open(data_dir + "/utt2spkrate", "w") as s:
        for f in flac_files:
            utt = os.path.basename(f).split(".")[0]
            spk_rate = compute_speaking_rate_with_whisper(f)
            s.write("{} {}\n".format(utt, spk_rate))