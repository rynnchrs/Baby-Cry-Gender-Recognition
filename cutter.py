from pydub import AudioSegment

# Works in milliseconds
t1 = 1000
t2 = 4000
newAudio = AudioSegment.from_wav(r"C:\Users\Administrator\Documents\Thesis\BABY CRY\(UNTRIMMED) BABY CRY\ish baby cry (UNTRIMMED)\HUNGRY1.wav")
newAudio = newAudio[t1:t2]

# Exports to a wav file in the current path.
newAudio.export('newSong.wav', format="wav")