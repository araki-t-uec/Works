#!/usr/bin/env sh

rsync -avz --delete --exclude "Models/*" --exclude "Modelsrepo/*" --backup --backup-dir=../Backup_Tensor /c/Users/jiro/Araki/Tensorflow/ im00:~/Git/Works/Tensorflow/
