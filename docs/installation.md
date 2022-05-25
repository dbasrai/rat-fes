# how to install

this is how to install on a linux or macOS machine. i think similar steps should work in a windows, but i'd suggest using windows subsystem for linux if so. 

## prerequesities
you need to install conda and git

## clone repo

go to https://github.com/dbasrai/rat-fes and then clone repo locally.

navigate inside rat-fes directory and create environment  with `conda env create -f env.yml`

activate env with `conda activate rat-fes`

## install src files

activate environment, and then do `pip install -e`. this activates all src files.

you should be good to go now!

## getting started
set-up a data folder according to layout specified in data_folder_layout.md in docs.

go to rat-fes/scripts, and then open in jupyter notebook or jupyter-lab. open up 'demo' script, and start functions.
