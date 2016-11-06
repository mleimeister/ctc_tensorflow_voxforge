import os
import urllib2
import tarfile
from BeautifulSoup import BeautifulSoup

# set two speakers for training/testing sets
speaker = 'Aaaron'  # defines substring that is searched for in the tar file names
voxforge_url = 'http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit'
target_folder = './Voxforge'


def download_file(url, target_folder):
    """
    Downloads and extracts a tar file given a URL and a target folder.
    """
    stream = urllib2.urlopen(url)
    tar = tarfile.open(fileobj=stream, mode="r|gz")

    for item in tar:
        tar.extract(item, target_folder)


if __name__ == '__main__':

    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)

    html_page = urllib2.urlopen(voxforge_url)
    soup = BeautifulSoup(html_page)

    # list all links
    links = soup.findAll('a')

    # download files for the specified speaker
    speaker_refs = [l['href'] for l in links if speaker in l['href']]

    for i, ref in enumerate(speaker_refs):
        print('Downloading {} / {} files'.format(i, len(speaker_refs)))
        download_file(voxforge_url + '/' + ref, target_folder)








