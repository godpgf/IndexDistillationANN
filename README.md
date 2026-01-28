# IndexDistillationANN


IndexDistillationANN is a library of approximate nearest neighbor search algorithms that validates our Multi-Start Strategy and Diversity-Aware Pruning. To facilitate experimentation, our code is forked from [ParlayANN](https://github.com/cmuparlay/ParlayANN) and further implements our methods on its basis. This repository was built for our paper titled “Improving Graph-based Approximate Nearest Neighbor Search via Multi-Start Strategy and Diversity-Aware Pruning”

To install, [clone the repo](https://github.com/godpgf/IndexDistillationANN) and then initiate the IndexDistillationANN submodule:

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
mkdir -p sift/res && mkdir -p sift/graph
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

All algorithms are in the algorithms directory.

```bash
cd algorithms
```

#### nnDescent

Many algorithms (such as NSG, NSSG, etc.) use the nearest neighbor graph constructed by the nnDescent algorithm as their initial graph structure.

1. **R** (`long`): the degree bound.
2. **L** (`long`): the beam width to use when building the graph.

```bash
cd nnDescent
make
./neighbors -alg_type nnDescent -R 50 -L 70 -alpha 0 -cos_angle 0 -cluster_size 1000 -num_clusters 32 -graph_outfile ../../data/sift/graph/sift_base.nnDescent -data_type uint8 -dist_func Euclidian -delta 0.05 -base_path ../../data/sift/sift_base.u8bin -query_path ../../data/sift/sift_query.u8bin -gt_path ../../data/sift/sift-1M -res_path ../../data/sift/res/nnDescent.csv
```

#### HNSW

We have integrated HNSW into the same code framework, with implementation methods and experimental parameters referencing the pseudocode provided in the HNSW paper.

```bash
cd HNSW
make
./neighbors -alg_type hnsw -R 50 -L 70 -alpha 1 -cos_angle 0 -graph_outfile ../../data/sift/graph/sift_base.hnsw -data_type uint8 -dist_func Euclidian -delta 0.05 -base_path ../../data/sift/sift_base.u8bin -query_path ../../data/sift/sift_query.u8bin -gt_path ../../data/sift/sift-1M -res_path ../../data/sift/res/hnsw.csv -meta_path ../../data/sift/graph/sift_base.hid
```

#### Vamana、NSG and NSSG 

Since the construction methods of Vamana, NSG, and NSSG are very similar to NSW, we integrated these algorithms together and use different parameters to activate each algorithm.
```bash
cd NSW && make && cd NSW
```

##### Vamana (DiskANN)

Vamana, also known as DiskANN, is an algorithm introduced in [DiskANN: Fast Accurate Billion-point Nearest
Neighbor Search on a Single Node](https://proceedings.neurips.cc/paper_files/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf) by Subramanya et al., with original code in the [DiskANN repo](https://github.com/microsoft/DiskANN). It builds a graph incrementally, and its insert procedure does a variant on greedy search or beam search with a frontier size $L$ on the existing graph and uses the nodes visited during the search as edge candidates. The visited nodes are pruned to a list of size $R$ by pruning out points that are likely to become long edges of triangles, with a parameter $a$ that is used to control how aggressive the prune step is. 

1. **R** (`long`): the degree bound.
2. **L** (`long`): the beam width to use when building the graph.
3. **alpha** (`double`): Use the parameters of the distance-based pruning strategy.


To build a Vamana graph on Sift1M and save it to memory, use the following commandline:

```bash
./neighbors -alg_type vamana -R 50 -L 70 -alpha 1.1 -graph_outfile ../../data/sift/graph/sift_base.vamana -data_type uint8 -dist_func Euclidian -delta 0.05 -base_path ../../data/sift/sift_base.u8bin -query_path ../../data/sift/sift_query.u8bin -gt_path ../../data/sift/sift-1M -res_path ../../data/sift/res/vamana.csv
```

##### NSG


```bash
./neighbors -alg_type nsg -R 50 -L 70 -alpha 1 -cos_angle 0 -graph_outfile ../../data/sift/graph/sift_base.nsg -data_type uint8 -dist_func Euclidian -delta 0.05 -base_path ../../data/sift/sift_base.u8bin -query_path ../../data/sift/sift_query.u8bin -gt_path ../../data/sift/sift-1M -res_path ../../data/sift/res/nsg.csv -graph_path ../../data/sift/graph/sift_base.nnDescent -sp_type 1 -rebuild
```

##### NSSG

```bash
./neighbors -alg_type nssg -R 50 -L 70 -alpha 0 -cos_angle 0.5756 -graph_outfile ../../data/sift/graph/sift_base.nssg -data_type uint8 -dist_func Euclidian -delta 0.05 -base_path ../../data/sift/sift_base.u8bin -query_path ../../data/sift/sift_query.u8bin -gt_path ../../data/sift/sift-1M -res_path ../../data/sift/res/nssg.csv -graph_path ../../data/sift/graph/sift_base.nnDescent -sp_type 2 -rebuild
```

#### SAG：Sparse and Accessible Graph（Ours）

To achieve the search performance of NSG and reduce the time required for building graph indices, we designed SAG based on the Vamana code. SAG involves two graph - construction operations: First, it builds a sparse graph and searches for neighbors for each node on this sparse graph to enhance the connectivity of nodes in all directions. Second, it utilizes the sparse graph with good connectivity to further search for neighbors, thereby improving the accuracy of neighbors (nodes that are closer in distance should be more likely to be neighbors).

```bash
./neighbors -alg_type vamana -R 50 -L 70 -alpha 1.1 -graph_outfile ../../data/sift/graph/sift_base.sag -data_type uint8 -dist_func Euclidian -delta 0.05 -base_path ../../data/sift/sift_base.u8bin -query_path ../../data/sift/sift_query.u8bin -gt_path ../../data/sift/sift-1M -res_path ../../data/sift/res/sag.csv -num_passes 2
```

####  Multi-Start（Ours）

Return to the root directory and compile our method according to the following script.

```bash
mkdir build
cd build
cmake ..
make
```

##### Coarse Mapping

We perform clustering using the Product Quantization (PQ) algorithm and select a node in each cluster as its centroid. For each vector to be inserted into the graph, the centroid is used as the search starting point to find its neighbors, and edges are then added between the vector and these neighbors. The process of finding this search starting point for each vector to be inserted into the graph is referred to as "Coarse Mapping". Note that multiple search starting points are possible.


```bash
./PQ train --data_type uint8 --dist_func Euclidian --ifile ../data/sift/sift_base.u8bin --pivotFile ../data/sift/sift_key_8_6_8.piv --times 8 --pivots_num 6 --chunk 8 --max_reps 16 --max_cache 0
./PQ train --data_type uint8 --dist_func Euclidian --ifile ../data/sift/sift_base.u8bin --pivotFile ../data/sift/sift_key_2_5_8.piv --times 2 --pivots_num 5 --chunk 8 --max_reps 16 --max_cache 0

./PQ infer --data_type uint8 --dist_func Euclidian --ifile ../data/sift/sift_base.u8bin --pivotFile ../data/sift/sift_key_8_6_8.piv --quantFile ../data/sift/sift_key_8_6_8.quant --quantLossFile ../data/sift/sift_key_8_6_8.loss --max_cache 0
./PQ infer --data_type uint8 --dist_func Euclidian --ifile ../data/sift/sift_base.u8bin --pivotFile ../data/sift/sift_key_2_5_8.piv --quantFile ../data/sift/sift_key_2_5_8.quant --quantLossFile ../data/sift/sift_key_2_5_8.loss --max_cache 0

./ExpPQDict --pivotFile ../data/sift/sift_key_2_5_8.piv --quantFile ../data/sift/sift_key_2_5_8.quant --quantLossFile ../data/sift/sift_key_2_5_8.loss --dictFile ../data/sift/sift_key_2_5_8.dict
./ExpPQDict --pivotFile ../data/sift/sift_key_8_6_8.piv --quantFile ../data/sift/sift_key_8_6_8.quant --quantLossFile ../data/sift/sift_key_8_6_8.loss --dictFile ../data/sift/sift_key_8_6_8.dict

./ExpSP --base_sp_file ../data/sift/sift_key.sp --query_sp_file ../data/sift/sift_query.sp --data_type uint8 --dist_func Euclidian --query_path ../data/sift/sift_query.u8bin --meta_path ../data/sift/[sift_key_2_5_8,sift_key_8_6_8]
```

#####  Multi-Start

```bash
./MultiStart -alg_type scour -R 50 -L 70 -alpha 1.1 -cos_angle 0 -graph_outfile ../data/sift/graph/sift_base.scour -data_type uint8 -dist_func Euclidian -delta 0.05 -base_path ../data/sift/sift_base.u8bin -query_path ../data/sift/sift_query.u8bin -gt_path ../data/sift/sift-1M -res_path ../data/sift/res/multi-start.csv -graph_path ../data/sift/graph/sift_base.nnDescent -rebuild -meta_path ../data/sift/[sift_key.sp,sift_query.sp]
```



#### Draw Result

Draw a line chart of the test results.

```bash
pip install matplotlib pandas seaborn scipy numpy
cd ./test
python draw_res.py
```

