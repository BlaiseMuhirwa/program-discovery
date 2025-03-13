## Pruning Program Search 

This repo contains the experimental setup for running pruning heuristics/program discovery. It is built on top of [flatnav](https://github.com/BlaiseMuhirwa/flatnav.git). All the code for pruning program discovery is under [program_search](/program_discovery/).

### Getting Started

Currently, there is only one example for running program discovery on mnist. First, let's grab this dataset from the ANN benchmarks.

```shell
$ ./bin/download_ann_benchmarks_datasets.sh mnist-784-euclidean
```

Next, let's install flatnav. 

```shell
$ cd python-bindings
$ pip install . 
```

Lastly, let's run the example

```shell
$ cd program_discovery 
$ make mnist
``` 

