# IndexDistillationANN


IndexDistillationANN is a library of approximate nearest neighbor search algorithms that validates our Index Distillation and Diversity-Aware approaches. To facilitate experimentation, our code is forked from [ParlayANN](https://github.com/cmuparlay/ParlayANN) and further implements our methods on its basis. This repository was built for our paper titled “Improving Graph-based Approximate Nearest Neighbor Search via Index Distillation and Diversity-Aware Pruning.”

To install, [clone the repo](https://github.com/XXXX) and then initiate the IndexDistillationANN submodule:

```bash
git submodule init
git submodule update
```

## Data Processing

The following is a crash course in quickly building and querying an index using IndexDistillationANN.

First, download a 100K slice of the BIGANN dataset.

```bash
mkdir -p data && cd data
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xf sift.tar.gz
mkdir -p data/sift/res && mkdir -p data/sift/graph
```

Next, convert it from the .u8vecs format to binary format:

```bash
cd ./data_tools
make fvec_to_uvec
make vec_to_bin
./fvec_to_uvec ../data/sift/sift_base.fvecs ../data/sift/sift_base.u8vecs
./fvec_to_uvec ../data/sift/sift_query.fvecs ../data/sift/sift_query.u8vecs
./vec_to_bin uint8 ../data/sift/sift_base.u8vecs ../data/sift/sift_base.u8bin
./vec_to_bin uint8 ../data/sift/sift_query.u8vecs ../data/sift/sift_query.u8bin
```

We also support vector data in floating-point format.

```bash
./vec_to_bin float ../data/sift/sift_base.fvecs ../data/sift/sift_base.fbin
./vec_to_bin float ../data/sift/sift_query.fvecs ../data/sift/sift_query.fbin
```

Next, calculate its ground truth up to $k=100$.

```bash
cd ./data_tools
make compute_groundtruth
./compute_groundtruth -base_path ../data/sift/sift_base.u8bin -query_path ../data/sift/sift_query.u8bin -data_type uint8 -k 100 -dist_func Euclidian -gt_path ../data/sift/sift-1M
```

If the vector data format is floating-point numbers：

```bash
./compute_groundtruth -base_path ../data/sift/sift_base.fbin -query_path ../data/sift/sift_query.fbin -data_type float -k 100 -dist_func Euclidian -gt_path ../data/sift/sift-1M
```

## Algorithms

Next we provide some descriptions and example commandline arguments for each algorithm in the implementation.

### Universal Parameters

#### Parameters for building:
1. **-graph_outfile** (optional): if graph is not already built, path the graph is written to. This is optional; if not provided, the graph will be built and will print timing and statistics before terminating.
2. **-data_type**: type of the base and query vectors. Currently "float", "int8", and "uint8" are supported.
3. **-dist_func**: the distance function to use when calculating nearest neighbors. Currently Euclidian distance ("euclidian") and maximum inner product search ("mips") are supported.
4. **-base_path**: path to the base file. We only work with files in the .bin format; for your convenience, a converter from the popular .vecs format has been provided in the data tools folder.

#### Parameters for searching:

1. **-gt_path**: path to the ground truth, in .ibin format.
2. **-graph_path** (optional): path to the ANNS graph in the case of using an already built graph.
3. **-query_path**: path to the queries in .bin format.
4. **-res_path** (optional): path where a CSV file of results can be written (it is written to in append form, so it can be used to collect results of multiple runs).
5. **-k** (`long`): the number of nearest neighbors to search for.

### Contrastive Experiment

####  Vamana (DiskANN)

Vamana, also known as DiskANN, is an algorithm introduced in [DiskANN: Fast Accurate Billion-point Nearest
Neighbor Search on a Single Node](https://proceedings.neurips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf) by Subramanya et al., with original code in the [DiskANN repo](https://github.com/microsoft/DiskANN). It builds a graph incrementally, and its insert procedure does a variant on greedy search or beam search with a frontier size $L$ on the existing graph and uses the nodes visited during the search as edge candidates. The visited nodes are pruned to a list of size $R$ by pruning out points that are likely to become long edges of triangles, with a parameter $a$ that is used to control how aggressive the prune step is. 

1. **R** (`long`): the degree bound.
2. **L** (`long`): the beam width to use when building the graph.
3. **alpha** (`double`): Use the parameters of the distance-based pruning strategy.


To build a Vamana graph on Sift1M and save it to memory, use the following commandline:

```bash
cd ./algorithms/vamana
make
./neighbors -R 50 -L 70 -alpha 1.1 -graph_outfile ../../data/sift/graph/vamana_50_70 -data_type uint8 -dist_func Euclidian -delta 0.05 -base_path ../../data/sift/sift_base.u8bin -query_path ../../data/sift/sift_query.u8bin -gt_path ../../data/sift/sift-1M -res_path ../../data/sift/res/vamana.csv
```


####  Global Index Distillation（Ours）

1. **gM** (`long`): Global Index Distillation Frequency

```bash
cd ./algorithms/GID
make
./neighbors -R 50 -L 70 -alpha 1.1 -gM 2 -graph_outfile ../../data/sift/graph/GID_50_70 -data_type uint8 -dist_func Euclidian -delta 0.05 -base_path ../../data/sift/sift_base.u8bin -query_path ../../data/sift/sift_query.u8bin -gt_path ../../data/sift/sift-1M -res_path ../../data/sift/res/GID.csv
```

#### pyNNDescent and NSSG

1. **cluster_size** (`long`):  The maximum number of elements that can be contained in each Voronoi cell of the clustering.
2. **num_clusters** (`long`): The number of times clustering is performed.
3. **angle** (`double`):A parameter for the angle-based pruning strategy.

```bash
cd ./algorithms/NSSG
make
# pyNNDescent
./neighbors -R 50 -L 70 -alpha 1.1 -graph_outfile ../../data/sift/graph/pyNNDescent_50_70 -data_type uint8 -dist_func Euclidian -delta 0.05 -cluster_size 1000 -num_clusters 32 -base_path ../../data/sift/sift_base.u8bin -query_path ../../data/sift/sift_query.u8bin -gt_path ../../data/sift/sift-1M -res_path ../../data/sift/res/pyNNDescent.csv
# NSSG
./neighbors -R 50 -L 70 -angle 0.5 -graph_outfile ../../data/sift/graph/NSSG_50_70 -data_type uint8 -dist_func Euclidian -delta 0.05 -cluster_size 1000 -num_clusters 32 -base_path ../../data/sift/sift_base.u8bin -query_path ../../data/sift/sift_query.u8bin -gt_path ../../data/sift/sift-1M -res_path ../../data/sift/res/NSSG.csv
```

#### Local Index Distillation（Ours）

1. **lM** (`long`): The number of local index distillation times

```bash
cd ./algorithms/LID
make
./neighbors -R 50 -L 70 -alpha 1.1 -lM 5 -graph_outfile ../../data/sift/graph/LID_50_70 -data_type uint8 -dist_func Euclidian -delta 0.05 -cluster_size 1000 -num_clusters 32 -base_path ../../data/sift/sift_base.u8bin -query_path ../../data/sift/sift_query.u8bin -gt_path ../../data/sift/sift-1M -res_path ../../data/sift/res/LID.csv
```

#### Diversity-Aware Pruning

Limit the number of edges to 20 and test the performance of the search.

```bash
cd ./algorithms/vamana
./neighbors -R 20 -L 70 -alpha 1.1 -graph_outfile ../../data/sift/graph/vamana_20_70 -data_type uint8 -dist_func Euclidian -delta 0.05 -base_path ../../data/sift/sift_base.u8bin -query_path ../../data/sift/sift_query.u8bin -gt_path ../../data/sift/sift-1M -res_path ../../data/sift/res/vamana_20.csv
```

Then the effectiveness of our pruning algorithm can be tested.

```bash
cd ./algorithms/vamana
./neighbors -PR 50 -R 20 -L 70 -alpha 1.1 -graph_outfile ../../data/sift/graph/vamana_diversity_aware_20_70 -data_type uint8 -dist_func Euclidian -delta 0.05 -base_path ../../data/sift/sift_base.u8bin -query_path ../../data/sift/sift_query.u8bin -gt_path ../../data/sift/sift-1M -res_path ../../data/sift/res/vamana_diversity_aware_20.csv
```


#### Draw Result

Draw a line chart of the test results.

```bash
pip install matplotlib pandas seaborn scipy numpy
cd ./test
python draw_res.py
```

