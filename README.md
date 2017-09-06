A general purpose tagger and a greedy transition-based dependency parser, both with CNN as character composition model.

The tagger is described in the SCLeM2017 Paper:
    https://arxiv.org/abs/1706.01723
    
The parser is described in the ACL2017 paper (with slight improvement):
    https://arxiv.org/abs/1705.10814

Requirements:

    - Python 2.7

    - Numpy
    
    - Theano 0.9.0
    
    - Lasagne 0.2-dev1

Basic usages:
  
    # Train the tagger
    python main.py -tool tagger -train [train_file] -dev [dev_file] -model_to [model_dir] -source_feat word,char -target_feat utag

    # Test the tagger
    python main.py -test [test_file] -model_from [model_dir] -out [output_file]

    # Train the parser
    python main.py -tool parser -train [train_file] -dev [dev_file] -model_to [model_dir] -source_feat word,char

    # Test the parser
    python main.py -test [test_file] -model_from [model_dir] -out [output_file]
