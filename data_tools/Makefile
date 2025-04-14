include ../algorithms/bench/parallelDefsANN

vec_to_bin : vec_to_bin.cpp
	$(CC) $(CFLAGS) -o vec_to_bin vec_to_bin.cpp $(LFLAGS)

fvec_to_uvec : fvec_to_uvec.cpp
	$(CC) $(CFLAGS) -o fvec_to_uvec fvec_to_uvec.cpp $(LFLAGS)

compute_groundtruth : compute_groundtruth.cpp
	$(CC) $(CFLAGS) -o compute_groundtruth compute_groundtruth.cpp $(LFLAGS) -I ../algorithms

compute_range_groundtruth : compute_range_groundtruth.cpp
	$(CC) $(CFLAGS) -o compute_range_groundtruth compute_range_groundtruth.cpp $(LFLAGS) 

crop : crop.cpp
	$(CC) $(CFLAGS) -o crop crop.cpp $(LFLAGS) 

random_sample : random_sample.cpp
	$(CC) $(CFLAGS) -o random_sample random_sample.cpp $(LFLAGS) 