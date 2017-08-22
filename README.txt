----------
Launching:
----------
Launch the prototype to create models with a <dataset> with 

1) TransE-SG.
	$ ./launcher_.sh launcher.py --fin <input/dataset.bin> --test-all 50 --nb 100 --me 300 --margin 2.0 --lr 0.1 --ncomp 50 --extvar True  --sampler=subgraph --resultsdir <path/to/resultDIR> --fout <path/to/save/models/file> --resultsfilename <outputfilename>

2) HOLE-SG:
	$ ./launcher_.sh launch_hole.py --fin <input/dataset.bin> --test-all 50 --nb 100 --me 300 --margin 0.2 --rparam 0.0 --lr 0.1 --ncomp 150 --extvar True  --sampler=subgraph --resultsdir <path/to/resultDIR> --fout <path/to/save/models/file> --resultsfilename <outputfilename>

--------------
Testing Model:
--------------

This runs with the parameters to run SRL models on given dataset.bin enabling creation of new variables (Subgraph Embeddings) with --extvar <True> option and also enbales negative sampling with --sampler=subgraph. To run Transe/Hole baseline make --extvar <False>, sampler=random-mode.

We also created a program to test each models, with various settings; this saves time from creating the models everytime. To test a model with;

	$./launcher_.sh <testmodel.py or testmodel_hole.py > --input <input/path/dataset.bin> --testValid TEST --testmode True --vectors <model> --fileTestValid <dataset/test.bin> --sampletest 1.0 --resultsdir <path/to/resultsDIR> --resultsfilename <outputfilename> --exvars True --post_threshold 5 --post2_enable True
	
To test a model with enriched dataset enable --exvars <True>. --sampletest <1.0> specifies the test data size (>=1.0) takes the full test data. The post processing parameters can be set here. 
