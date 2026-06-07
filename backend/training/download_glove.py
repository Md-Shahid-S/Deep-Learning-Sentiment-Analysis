import urllib.request, zipfile, os

URL = "https://nlp.stanford.edu/data/glove.6B.zip"
os.makedirs("../data/glove", exist_ok=True)
urllib.request.urlretrieve(URL, "../data/glove/glove.6B.zip")
with zipfile.ZipFile("../data/glove/glove.6B.zip") as z:
    z.extract("glove.6B.100d.txt", "../data/glove/")
print("Done!")