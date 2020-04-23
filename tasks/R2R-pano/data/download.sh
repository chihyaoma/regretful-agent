#!/bin/sh

# The download links are from Peter Anderson's original GitHub repo.
# Check the below link if the dropbox links are not working.
# https://github.com/peteanderson80/Matterport3DSimulator/blob/master/tasks/R2R/data/download.sh

wget https://www.dropbox.com/s/hh5qec8o5urcztn/R2R_train.json -P tasks/R2R-pano/data/
wget https://www.dropbox.com/s/8ye4gqce7v8yzdm/R2R_val_seen.json -P tasks/R2R-pano/data/
wget https://www.dropbox.com/s/p6hlckr70a07wka/R2R_val_unseen.json -P tasks/R2R-pano/data/
wget https://www.dropbox.com/s/w4pnbwqamwzdwd1/R2R_test.json -P tasks/R2R-pano/data/