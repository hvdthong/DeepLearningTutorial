import glob
files = glob.glob('*.txt')

words = []
for f in files:
    file = open(f)
    words.append(file.read())
    file.close()

print len(words)