### Essence of the data


Every minute, the world loses an area of forest the size of 48 football fields. And deforestation in the Amazon Basin accounts for the largest share, contributing to reduced biodiversity, habitat loss, climate change, and other devastating effects. But better data about the location of deforestation and human encroachment on forests can help governments and local stakeholders respond more quickly and effectively.

Planet, designer and builder of the world’s largest constellation of Earth-imaging satellites, will soon be collecting daily imagery of the entire land surface of the earth at 3-5 meter resolution. While considerable research has been devoted to tracking changes in forests, it typically depends on coarse-resolution imagery from Landsat (30 meter pixels) or MODIS (250 meter pixels). This limits its effectiveness in areas where small-scale deforestation or forest degradation dominate.

Furthermore, these existing methods generally cannot differentiate between human causes of forest loss and natural causes. Higher resolution imagery has already been shown to be exceptionally good at this, but robust methods have not yet been developed for Planet imagery.

### File formats
train.csv - a list of training file names and their labels, the labels are space-delimited
sample_submission.csv - correct format of submission, contains all the files in the test set. For more information about the submission file, please go to the Evaluation page
[train/test]-tif-v2.tar.7z - tif files for the training/test set (updated: May 5th, 2017)
[train/test]-jpg[-additional].tar.7z - jpg files for the trainin/test set (updated: May 5th, 2017)
Kaggle-planet-[train/test]-tif.torrent - a BitTorrent file for downloading [train/test]-tif-v2.tar.7z (updated: May 5th, 2017)
For all the tar.7z files, you can extract them with:

7za x train-jpg.tar.7z

tar xf train-jpg.tar


### The training was done on kaggle notebook
