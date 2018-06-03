import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import wave
import os


def generateSpectogram(filename):
    y, sr = librosa.load(filename)
    librosa.feature.melspectrogram(y=y, sr=sr)
    D = np.abs(librosa.stft(y)) ** 2
    S = librosa.feature.melspectrogram(S=D)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    plt.figure(figsize=(0.22, 1.28))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), fmax=8000)
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Mel spectrogram')
    # plt.tight_layout()
    plt.draw()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(filename[:-4] + '.png', dpi=100, bbox_inches=0)


def mergeFiles():
    infiles = os.listdir('data_ready')
    cats = []
    dogs = []
    for file in infiles:
        if str.startswith(file, 'cat'):
            cats.append(file)
        if str.startswith(file, 'dog'):
            dogs.append(file)
    catsoutfile = "cats.wav"
    dogsoutfile = "dogs.wav"

    datacats = []
    for infile in cats:
        w = wave.open('data_ready/' + infile, 'rb')
        datacats.append([w.getparams(), w.readframes(w.getnframes())])
        w.close()

    datadogs = []
    for infile in dogs:
        w = wave.open('data_ready/' + infile, 'rb')
        datadogs.append([w.getparams(), w.readframes(w.getnframes())])
        w.close()

    catsoutput = wave.open(catsoutfile, 'wb')
    catsoutput.setparams(datacats[0][0])

    dogsoutput = wave.open(dogsoutfile, 'wb')
    dogsoutput.setparams(datadogs[0][0])
    i = 0
    for file in cats:
        catsoutput.writeframes(datacats[i][1])
        i += 1
    catsoutput.close()
    i = 0
    for file in dogs:
        dogsoutput.writeframes(datadogs[i][1])
        i += 1
    dogsoutput.close()


def splitEqually():
    from pydub import AudioSegment
    dogs = AudioSegment.from_wav("dogs.wav")
    for i in range(500, dogs.__len__(), 500):
        # print(i-500,i)
        # for(dogs.__len__())
        newAudio = dogs[i - 500:i]
        newAudio.export('data_samples/dogs' + str(i // 500) + '.wav', format="wav")

    cats = AudioSegment.from_wav("cats.wav")
    for i in range(500, cats.__len__(), 500):
        # print(i-500,i)
        # for(dogs.__len__())
        newAudio = cats[i - 500:i]
        newAudio.export('data_samples/cats' + str(i // 500) + '.wav', format="wav")


def generateSpectograms(directory):
    infiles = os.listdir(directory)
    for file in infiles:
        generateSpectogram(directory + '/' + file)


def getLabels(directory):
    files = os.listdir(directory + '/train')
    trainlabels = []
    for file in files:
        if str.startswith(file, 'cat'):
            trainlabels.append(1)
        if str.startswith(file, 'dog'):
            trainlabels.append(0)
    testfiles = os.listdir(directory + '/test')
    testlabels = []
    for file in testfiles:
        if str.startswith(file, 'cat'):
            testlabels.append(1)
        if str.startswith(file, 'dog'):
            testlabels.append(0)
    return (np.array(trainlabels), np.array(testlabels))

# generateSpectograms('data_samples')
# generateSpectogram('data_samples/cats1.wav')
