# deeplearning-biology

This is a list of implementations of deep learning methods to biology, originally published on [Follow the Data](https://followthedata.wordpress.com/). There is a slant towards genomics because that's the subfield that I follow most closely.

Please, contribute to this growing list, especially in categories that I haven't covered well! Also, do add your contributions to [GitXiv](http://gitxiv.com/) as well if you can.

You might also want to refer to the [awesome deepbio](https://github.com/gokceneraslan/awesome-deepbio) list.

## Table of contents
* [Reviews](#reviews)
* [General](#general)
* [Model repositories and resources](#repositories)
* [Chemoinformatics and drug discovery](#chemo)
* [Biomarker discovery](#biomarker)
* [Proteomics](#proteomics)
* [Metabolomics](#metabolomics)
* [Generative models](#generative)
* [Generic 'omics tools](#omics)
  - [NLP inspired genomics applications](#omics_nlp)
* [Genomics](#genomics)
  - [Variant calling](#genomics_variant-calling)
  - [Gene expression](#genomics_expression)
  - [Predicting enhancers and regulatory elements](#genomics_enhancers)
  - [Methylation](#genomics_methylation)
  - [Single-cell applications](#genomics_single-cell)
  - [Non-coding RNA](#genomics_non-coding)
  - [Population genetics](#genomics_pop)
* [Systems biology](#sysbio)
* [Neuroscience](#neuro)

## Reviews <a name="reviews"></a>

These are not implementations as such, but contain useful pointers. Because review papers in this field are more time-sensitive, I have added the month of journal publication. Note that the original preprint may in some cases have been available online long before the published version.

**(2019-04) Deep learning: new computational modelling techniques for genomics** [[Nature Reviews Genetics paper](https://www.nature.com/articles/s41576-019-0122-6)]

This is a very nice conceptual review of how deep learning can be used in genomics. It explains how convolutional networks, recurrent networks, graph convolutional networks, autoencoders and GANs work. It also explains useful concepts like multi-modal learning, transfer learning, and model explainability.

**(2018-11) A primer on deep learning in genomics** [[Nature Genetics paper](https://www.nature.com/articles/s41588-018-0295-5)][[Colaboratory notebook with tutorial](https://colab.research.google.com/drive/17E4h5aAOioh5DiTo7MZg4hpL6Z_0FyWr)]

This review, which features yours truly as one of its co-authors, is billed as a 'primer' which means it tries to help genomics researchers get started with deep learning. We tried to accomplish this by highlighting many practical issues such as tooling (not only deep learning libraries but also GPU cloud platforms, model zoos and online courses), defining your deep learning problem, explainability and troubleshooting. We also made a tutorial on Colaboratory that shows how to set up and run a simple convolutional network model for learning binding motifs, and how to inspect the model's predictions after it has been trained.

**(2018-04) Opportunities And Obstacles For Deep Learning In Biology And Medicine** [[bioRxiv preprint](http://biorxiv.org/content/early/2017/05/28/142760)][[J Roy Soc interface paper](https://royalsocietypublishing.org/doi/10.1098/rsif.2017.0387)]

This impressive collaborative review was written completely in the open on [Github](https://github.com/greenelab/deep-review). It is focused on discussing how deep learning may be able to transform patient classification and treatment as well as fundamental biological research in the future, and what the main obstacles are that could prevent it from happening. A lot of interesting points are brought up here. Together with the review listed below, which has a more technical slant, you will get a good overview of how deep learning is used and can be used in biology and medicine.

**(2017-01) Deep learning for health informatics** [[open access paper](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7801947)]

An overview of several types of deep nets and their applications in translational bioinformatics, medical imaging, "pervasive sensing", medical data and public health.

**(2016-07) Deep learning for computational biology** [[open access paper](http://msb.embopress.org/content/12/7/878)]

This is a very nice review of deep learning applications in biology. It primarily deals with convolutional networks and explains well why and how they are used for sequence (and image) classification.

## General <a name="general"></a>

Papers on methods that are more widely applicable to biological or clinical data.

**Fast animal pose estimation using deep neural networks** [[Github](https://github.com/talmo/leap)][[bioRxiv preprint](
https://www.biorxiv.org/content/biorxiv/early/2018/05/25/331181.full.pdf)]

This paper describes generating confidence maps and pose from flies. The repository includes a graphical user interface for labeling body parts.

## Model repositories and resources <a name="repositories"></a>

**The Kipoi repository accelerates community exchange and reuse of predictive models for genomics** [[Github](https://github.com/kipoi/kipoiseq/)][[Website](https://kipoi.org/)][[Paper](https://www.nature.com/articles/s41587-019-0140-0)] 

Kipoi is a model zoo for genomics, installable by a simple pip install, which provides a consistent interface to hundreds of predictive models in genomics. Kipoi implements a standard set of data loaders for training and prediction of sequence models in deep learning.

**DragoNN** [[Github](https://github.com/kundajelab/dragonn)[[Website](https://kundajelab.github.io/dragonn/)]

DragoNN provides a toolkit for learning about modelling regulatory sequence with neural networks. It has tools for interpreting sequence models and web-based tutorials using Jupyter Notebooks for teaching interactive model manipulation and visualization.

## Chemoinformatics and drug discovery <a name="chemo"></a>

**Neural graph fingerprints** [[github](https://github.com/HIPS/neural-fingerprint)][[gitxiv](http://gitxiv.com/posts/DFtFytneou3SXLuSM/convolutional-networks-on-graphs-for-learning-molecular)]

A convolutional net that can learn features which are useful for predicting properties of novel molecules; “molecular fingerprints”. The net works on a graph where atoms are nodes and bonds are edges. Developed by the group of Ryan Adams, who used to co-host the very good [Talking Machines](http://www.thetalkingmachines.com/) podcast.

**Automatic chemical design using a data-driven continuous representation of molecules** [[github](https://github.com/aspuru-guzik-group/chemical_vae)][[preprint](https://arxiv.org/abs/1610.02415)]

Abstract starts: "We report a method to convert discrete representations of molecules to and from a multidimensional continuous representation. This model allows us to generate new molecules for efficient exploration and optimization through open-ended spaces of chemical compounds."

**Objective-Reinforced Generative Adversarial Networks (ORGAN)** [[github](https://github.com/gablg1/ORGAN)][[preprint](https://arxiv.org/abs/1705.10843)]

A method that combines generative models with reinforcement learning to direct the generative process towards some desired target, ORGAN is a generic method for discrete data but is in this case exemplified by a drug discovery use case.

**Molecular De-Novo Design through Deep Reinforcement Learning** [[github](https://github.com/MarcusOlivecrona/REINVENT)][[preprint](https://arxiv.org/abs/1704.07555)]

PyTorch sequence generation model that uses reinforcement learning. Nice widget showing training progress and molecules generated during training is shown on the Github page. Abstract starts: "This work introduces a method to tune a sequence-based generative model for molecular de novo design that through augmented episodic likelihood can learn to generate structures with certain specified desirable properties. We demonstrate how this model can execute a range of tasks such as generating analogues to a query structure and generating compounds predicted to be active against a biological target."

**One-shot learning models for drug discovery and DeepChem** [[github](https://github.com/deepchem/deepchem)][[Python library](http://deepchem.io/)][[paper](http://pubs.acs.org/doi/abs/10.1021/acscentsci.6b00367)]

DeepChem is a "... [P]ython library that aims to make the use of machine-learning in drug discovery straightforward and convenient" which checks a lot of boxes when it comes to advanced deep learning: one-shot learning, graph convolutional networks, learning from less data, and LSTM embeddings. According to the GitHub site, "DeepChem aims to provide a high quality open-source toolchain that democratizes the use of deep-learning in drug discovery, materials science, and quantum chemistry."

**The cornucopia of meaningful leads: Applying deep adversarial autoencoders for new molecule development in oncology** [[github](https://github.com/spoilt333/onco-aae)][[paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5355231/)]

Explores the use of generative adversarial networks (GAN) in generating new molecular leads for drug candidates. In analogy to generating images or video that "look like" they come from some specified distribution, perhaps with some conditioning like "show me a cat picture", the authors reason that novel drug-like molecular structures can be generated with cues about what kind of drug one wants. Here they explore a specific type of generative network, an adversarial autoencoder (AAE), and adapt it into what they call a "artificially-intelligent drug discovery engine."

**Deep learning enables rapid identification of potent DDR1 kinase inhibitors** [[github](https://github.com/insilicomedicine/gentrl)][paper](https://www.nature.com/articles/s41587-019-0224-x)] In this paper from InSilico Medicine, which came out to some fanfare in 2019, an approach called GENTRL (Generative Tensorial Reinforcement Learning) was used to do rapid discovery of small-molecule inhibitors towards an interesting target. Using this method, the authors were able to come up with a candidate molecule in just 21 days. The model uses an initial generative step with a variational autoencoder and a reinforcement learning procedure for exploring the chemical space. They use an interesting loss function based on Kohonen self-organizing maps. Tensor decomposition was used to encode the relationship between chemical structures and properties. 


## Biomarker discovery <a name="biomarker"></a>

**Deep biomarkers of human aging** [[online predictor](http://www.aging.ai/)][[paper](https://www.ncbi.nlm.nih.gov/pubmed/27191382)]

From the abstract: "One of the major impediments in human aging research is the absence of a comprehensive and actionable set of biomarkers that may be targeted and measured to track the effectiveness of therapeutic interventions. In this study, we designed a modular ensemble of 21 deep neural networks (DNNs) of varying depth, structure and optimization to predict human chronological age using a basic blood test. "

## Generic 'omics tools <a name="omics"></a>

**Continuous Distributed Representation of Biological Sequences for Deep Genomics and Deep Proteomics**[[github](https://github.com/ehsanasgari/Deep-Proteomics)][[paper](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0141287)]

The GitHub summary reads: "We introduce a new representation for biological sequences. Named bio-vectors (BioVec) to refer to biological sequences in general with protein-vectors (ProtVec) for proteins (amino-acid sequences) and gene-vectors (GeneVec) for gene sequences, this representation can be widely used in applications of deep learning in proteomics and genomics. Biovectors are basically n-gram character skip-gram wordvectors for biological sequences (DNA, RNA, and Protein). In this work, we have explored biophysical and biochemical meaning of this space. In addition, in variety of bioinformatics tasks we have shown the strength of such a sequence representation."

**pysster: Learning Sequence and Structure Motifs in DNA and RNA Sequences using Convolutional Neural Networks**[[github](https://github.com/budach/pysster)][[preprint](https://www.biorxiv.org/content/early/2017/12/06/230086)]

A toolbox for learning motifs from DNA/RNA sequence data using convolutional neural networks, this Tensorflow-based library supposedly runs on GPU out of the box and also does things like hyperparameter optimization and visualizations of what different network layers are learning.

### NLP inspired <a name='omics_nlp'></a>

**Genomic-ULMFiT: ULMFiT for Genomic Sequence Data** [[github](https://github.com/kheyer/Genomic-ULMFiT)]

This repo is an implementation of FastAI's ULMFiT language transfer learning model for genomics. ULMFiT is based on an AWD-LSTM model and has been shown to be very effective for solving various text classification tasks. Here, the repo's author has extended FastAI's classes with specific subclasses for DNA sequence data. The concept with ULMFiT is that you (1) learn a language model from a large body of text in an unsupervised way (ie you don't need any labels) by having the model guess the next word (or token); (2) take the language model from step (1) and fine-tune it on the (probably) smaller labeled data set that you want to do classification on, but still do the training without labels in this step (and try to predict the next word), (3) finally fine-tune on the final classification task, using the labels. In genomics, the large body of text in step (1) could be, for instance, the whole human genome, or some other subset of GenBank/Sequence Read Archive/... The author shows that this approach works quite well for a range of classification problems, like E. coli and human promoter classification, metagenomic classification, enhancer classification and mRNA/lincRNA classification. 

**Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences** [[preprint](https://www.biorxiv.org/content/10.1101/622803v1.full)]

In this work from Facebook's AI group, the BERT language model is used to train a language model on 86 billion amino acids across 250 million sequences. Like with ULMFiT (above), the idea is to use transfer learning: pre-training on a massive amount of data to teach a model something about the underlying logic of the language of DNA or proteins, in order to then be able to fine-tune the model for specific tasks. Unfortunately I haven't found any implementation for this yet.

## Proteomics <a name="proteomics"></a>

**Pcons2 – Improved Contact Predictions Using the Recognition of Protein Like Contact Patterns** [[web interface](http://c2.pcons.net/)]

Here, a “deep random forest” with five layers is used to improve predictions of which residues (amino acids) in a protein are physically interacting which each other. This is useful for predicting the overall structure of the protein (a very hard problem.)

**A Deep Learning Model for Predicting Tumor Suppressor Genes and Oncogenes from PDB Structure** [[github](https://github.com/tavanaei/Cancer-Suppressor-Gene-Deep-Learning)][[bioRxiv preprint](https://www.biorxiv.org/content/early/2017/10/22/177378)]

The authors use CNNs on feature maps extracted from protein 3D structures in the Protein Data Base (PDB) to predict oncogenes and tumor-suppressor genes.   

**Deep-RBPPred: Predicting RNA binding proteins in the proteome scale based on deep learning** [[code](http://www.rnabinding.com/Deep_RBPPred/Deep-RBPPred.html)][[bioRxiv preprint](https://www.biorxiv.org/content/early/2017/10/27/210153)] 

Predicts RNA-binding proteins using CNNs.

**EVOVAE: Vartiational autoencoding of Protein Sequences**[[code](https://github.com/samsinai/VAE_protein_function)][[arXiv preprint](https://arxiv.org/abs/1712.03346)]

From the abstract: "We present an embedding of natural protein sequences using a Variational Auto-Encoder and use it to predict how mutations affect protein function. We use this unsupervised approach to cluster natural variants and learn interactions between sets of positions within a protein. This approach generally performs better than baseline methods that consider no interactions within sequences, and in some cases better than the state-of-the-art approaches that use the inverse-Potts model. This generative model can be used to computationally guide exploration of protein sequence space and to better inform rational and automatic protein design."

**Protein Loop Modeling Using Deep Generative Adversarial Network**[[paper](https://ieeexplore.ieee.org/abstract/document/8372069/)][[website](https://zhaoyu.li/loop_modeling_gan.html)]

From the abstract: "Biology and medicine have a long-standing interest in computational structure prediction and modeling of proteins. There are often missing regions or regions that need to be remodeled in protein structures. The process of predicting particular missing regions in a protein structure is called loop modeling. In this paper, we propose a generative adversarial network (GAN) in deep learning for loop modeling using the idea of image inpainting. The generative network is to capture the context of the loop region and predict the missing area. The adversarial network is to make the prediction look real and provide gradients to the generative network. The proposed network was evaluated on a common benchmark for loop modeling. Experiments show that our method can successfully predict the loop region and has achieved better performance than the state-of-the-art tools. To our knowledge, this work represents the first attempt of using GAN for any bioinformatics studies."


## Metabolomics <a name="metabolomics"></a>

**Deep Learning Accurately Predicts Estrogen Receptor Status in Breast Cancer Metabolomics Data** [[code](http://pubs.acs.org/doi/suppl/10.1021/acs.jproteome.7b00595/suppl_file/pr7b00595_si_001.pdf)][[paper](http://pubs.acs.org/doi/full/10.1021/acs.jproteome.7b00595)]

Classification algorithms for metabolomics data with respect to estrogen receptor status are compared, and the best performing algorithm is an autoencoder-based feedforward network with parameters tuned using H2O's R interface.

### Generative models <a name='generative'></a>

In many cases, it can be useful to generate synthetic data that resembles real data in order to boost dataset sizes or avoid violating patient privacy. Here, some of these approaches are listed.

**Privacy-preserving generative deep neural networks support clinical data sharing** [[Github](https://github.com/greenelab/SPRINT_gan)][[bioRxiv preprint](https://www.biorxiv.org/content/early/2017/11/15/159756)]

This describes a clever idea where generative adversarial networks (GANs) are used to synthesize data that closely resembles actual data measured on study participants, but which cannot be traced back to a specific subject. The latter aspect, called differential privacy, is incorporated into the method by design and gives strong guarantees of the likelihood that a subject could be identified as a member of a trial.

**Creating artificial human genomes using generative models** [[preprint](https://www.biorxiv.org/content/biorxiv/early/2019/10/07/769091.full.pdf)]

The authors compare Restricted Boltzmann Machines (RBM) and Generative Adversarial Networks (GAN) as tools for creating synthetic human genomes.

## Genomics <a name="genomics"></a>

This category is divided into several subfields.

### Variant calling <a name='genomics_variant-calling'></a>

**DeepVariant** [[github](https://github.com/google/deepvariant)][[preprint](https://www.biorxiv.org/content/early/2016/12/21/092890)]

This preprint from Google originally came out in late 2016 but it got the most publicity about a year later when the code was made public and press releases started appearing. The Google researchers approached a well-studied problem, variant calling from DNA sequencing data (where the aim is to correctly identify variations from the reference genome in an individual's DNA, e.g. mutations or polymorphisms) using a counter-intuitive but clever approach. Instead of using the nucleotides in the sequenced DNA fragments directly (in the form of the symbols A, C, G, T), they first converted the sequences into images and then applied convolutional neural networks to these images (which represent "pile-ups" or DNA sequences; stacks of aligned sequences.) This turned out to be a very effective way to call variants as proven by both Google's own and independent benchmarks.

### Gene expression <a name='genomics_expression'></a>

In modeling gene expression, the inputs are typically numerical values (integers or floats) estimating how much RNA is produced from a DNA template in a particular cell type or condition.

**Gene Expression Convolutions Using Gene Interaction Graphs** [[github](https://github.com/mila-iqia/gene-graph-conv)] [[arxiv](https://github.com/mila-iqia/gene-graph-conv)]
They discuss how gene-gene interaction graphs (same pathway, protein-protein, co-expression, or research paper text association) can be used to impose a bias on a deep neural network model similar to the spatial bias imposed by convolutions on an image. They find this approach provides an advantage for particular tasks in a low data regime but is very dependent on the quality of the graph used. 

**ADAGE – Analysis using Denoising Autoencoders of Gene Expression** [[github](https://github.com/greenelab/adage)][[gitxiv](http://gitxiv.com/posts/M9Dnc8HbKvNgsSp5D/adage-analysis-using-denoising-autoencoders-of-gene)]

This is a Theano implementation of stacked denoising autoencoders for extracting relevant patterns from large sets of gene expression data, a kind of feature construction approach if you will. I have played around with this package quite a bit myself. The authors initially published a [conference paper](http://www.worldscientific.com/doi/abs/10.1142/9789814644730_0014) applying the model to a compendium of breast cancer (microarray) gene expression data, and more recently posted a paper on [bioRxiv](http://biorxiv.org/content/early/2015/11/05/030650) where they apply it to all available expression data (microarray and RNA-seq) on the pathogen Pseudomonas aeruginosa. (I understand that this manuscript will soon be published in a journal.)

**Learning structure in gene expression data using deep architectures** [[paper](http://biorxiv.org/content/early/2015/11/16/031906)]

This is also about using stacked denoising autoencoders for gene expression data, but there is no available implementation (as far as I could tell). Included here for the sake of completeness (or something.)

**Gene expression inference with deep learning** [[github](https://github.com/uci-cbcl/D-GEX)][[paper](http://biorxiv.org/content/early/2015/12/15/034421)]

This deals with a specific prediction task, namely to predict the expression of specified target genes from a panel of about 1,000 pre-selected “landmark genes”. As the authors explain, gene expression levels are often highly correlated and it may be a cost-effective strategy in some cases to use such panels and then computationally infer the expression of other genes. Based on Pylearn2/Theano.

**Learning a hierarchical representation of the yeast transcriptomic machinery using an autoencoder model** [[paper](http://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-015-0852-1)]

The authors use stacked autoencoders to learn biological features in yeast from thousands of microarrays. They analyze the hidden layer representations and show that these encode biological information in a hierarchical way, so that for instance transcription factors are represented in the first hidden layer.

**Boosting Gene Expression Clustering with System-Wide Biological Information: A Robust Autoencoder Approach** [[bioRxiv preprint](https://www.biorxiv.org/content/early/2017/11/05/214122)]

Uses a robust autoencoder (an autoencoder with an outlier filter) to cluster gene expression profiles. 

**Deep learning sequence-based ab initio prediction of variant effects on expression and disease risk** [[github](https://github.com/FunctionLab/ExPecto)][[paper](https://www.nature.com/articles/s41588-018-0160-6)]

The authors use a two-step model to predict the effect of genetic variants on gene expression. In the first step, the authors trained a convolutional neural network to model the 2002 epigenetic marks collected in ENCODE and ROADMAP consortium. In the second step, the authors trained a tissue-specific regularized linear model on the cis-regulatory region of the gene that is encoded by the first step convolutional neural network model. Then the effect of the variants on tissue-specific gene is calculated by the decrease in predicted gene expression through *in silico* mutagenesis.

### Predicting enhancers and regulatory regions <a name='genomics_enhancers'></a>

Here the inputs are typically “raw” DNA sequence, and convolutional networks (or layers) are often used to learn regularities within the sequence. Hat tip to [Melissa Gymrek](http://melissagymrek.com/science/2015/12/01/unlocking-noncoding-variation.html) for pointing out some of these.

**DanQ: a hybrid convolutional and recurrent deep neural network for quantifying the function of DNA sequences** [[github](https://github.com/uci-cbcl/DanQ)][[gitxiv](http://gitxiv.com/posts/aqrWwLoyg75jqNAYX/danq-a-hybrid-convolutional-and-recurrent-deep-neural)]

Made for predicting the function of non-protein coding DNA sequence. Uses a convolution layer to capture regulatory motifs (i e single DNA snippets that control the expression of genes, for instance), and a recurrent layer (of the LSTM type) to try to discover a “grammar” for how these single motifs work together. Based on Keras/Theano.

**Basset – learning the regulatory code of the accessible genome with deep convolutional neural networks** [[github](https://github.com/davek44/Basset)][[gitxiv](http://gitxiv.com/posts/fhET6G7gnBrGS8S9u/basset-learning-the-regulatory-code-of-the-accessible-genome)]

Based on Torch, this package focuses on predicting the accessibility (or “openness”) of the chromatin – the physical packaging of the genetic information (DNA+associated proteins). This can exist in more condensed or relaxed states in different cell types, which is partly influenced by the DNA sequence (not completely, because then it would not differ from cell to cell.)

**Basenji – Sequential regulatory activity prediction across chromosomes with convolutional neural networks** [[github1](https://www.github.com/calico/basenji)][[github2](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/gene_expression.py)][[biorxiv](https://www.biorxiv.org/content/early/2017/07/10/161851)]

A follow-up project to Basset, this Tensorflow-based model uses both standard and dilated convolutions to model regulatory signals and gene expression (in the form of CAGE tag density) in many different cell types. Notably, the underlying model has been brought into Google's Tensor2Tensor repository (see "github2" link above), which collects many models in image and speech recognition, machine translation, text classification etc. However, at the time of writing the Tensor2Tensor model seems not quite mature for easy use, so it is probably better to use the dedicated Basenji repo ("github1") for now. 

**DeepSEA – Predicting effects of noncoding variants with deep learning–based sequence model** [[web server](http://deepsea.princeton.edu/job/analysis/create/)][[paper](http://www.nature.com/nmeth/journal/v12/n10/full/nmeth.3547.html)]

Like the packages above, this one also models chromatin accessibility as well as the binding of certain proteins (transcription factors) to DNA and the presence of so-called histone marks that are associated with changes in accessibility. This piece of software seems to focus a bit more explicitly than the others on predicting how single-nucleotide mutations affect the chromatin structure. Published in a high-profile journal (Nature Methods).

**DeepBind – Predicting the sequence specificities of DNA- and RNA-binding proteins by deep learning** [[code](http://tools.genes.toronto.edu/deepbind/)][[paper](http://www.nature.com/nbt/journal/v33/n8/full/nbt.3300.html)]

This is from the group of Brendan Frey in Toronto, and the authors are also involved in the company Deep Genomics. DeepBind focuses on predicting the binding specificities of DNA-binding or RNA-binding proteins, based on experiments such as ChIP-seq, ChIP-chip, RIP-seq,  protein-binding microarrays, and HT-SELEX. Published in a high-profile journal (Nature Biotechnology.)

**DeeperBind - Enhancing Prediction of Sequence Specificities of DNA Binding Proteins** [[preprint](https://arxiv.org/pdf/1611.05777.pdf)]

This is an attempt to improve on DeepBind by adding a recurrent sequence learning module (LSTM) after the convolutional layer(s). In this way, the authors propose to capture a positional dimension that is lost in the pooling step in the original DeepBind design. They claim that benchmarking shows that this architecture leads to superior performance compared to previous work.

**DeepMotif - Visualizing Genomic Sequence Classifications** [[paper](https://arxiv.org/abs/1605.01133)]

This is also about learning and predicting binding specificities of proteins to certain DNA patterns or "motifs". However, this paper makes use of a combination of convolutional layers and [highway networks](https://arxiv.org/pdf/1505.00387v2.pdf), with more layers than the DeepBind network. The authors also show how a learned classifier can generate typical DNA motifs by input optimization; applying back-propagation with all the weights held constant in order to find an input pattern that maximally activates the appropriate output node in the network.

**Convolutional Neural Network Architectures for Predicting DNA-Protein Binding** [[code](http://cnn.csail.mit.edu/)][[paper](http://bioinformatics.oxfordjournals.org/content/32/12/i121.full)]

This work describes a systematic exploration of convolutional neural network (CNN) architectures for DNA-protein binding. It concludes that the convolutional kernels are very important for the success of the networks on motif-based tasks. Interestingly, the authors have provided a Dockerized implementation of DeepBind from the Frey lab (see above) and also provide EC2-laucher scripts and code for comparing different GPU enabled models programmed in Caffe.

**PEDLA: predicting enhancers with a deep learning-based algorithmic framework** [[code](https://github.com/wenjiegroup/PEDLA)][[paper](http://biorxiv.org/content/early/2016/01/07/036129)]

This package is for predicting enhancers (stretches of DNA that can enhance the expression of a gene under certain conditions or in a certain kind of cell, often working at a distance from the gene itself) based on heterogeneous data from (e.g.) the ENCODE project, using 1,114 features altogether.

**DEEP: a general computational framework for predicting enhancers** [[paper](http://nar.oxfordjournals.org/content/early/2014/11/05/nar.gku1058.full)][[code](http://cbrc.kaust.edu.sa/deep/)]

An ensemble prediction method for enhancers.

**Genome-Wide Prediction of cis-Regulatory Regions Using Supervised Deep Learning Methods** (and several other papers applying various kinds of deep networks to regulatory region prediction) [[code](https://github.com/yifeng-li/DECRES)] (one [[paper](http://biorxiv.org/content/early/2016/02/28/041616)] out of several)

Wyeth Wasserman’s group have made a kind of [toolkit](https://github.com/yifeng-li/DECRES) (based on the Theano tutorials) for applying different kinds of deep learning architectures to cis-regulatory element (DNA stretches that can modulate the expression of a nearby gene) prediction. They use a specific “feature selection layer” in their nets to restrict the number of features in the models. This is implemented as an additional sparse one-to-one linear layer between the input layer and the first hidden layer of a multi-layer perceptron.

**FIDDLE: An integrative deep learning framework for functional genomic data inference** [[paper](http://biorxiv.org/content/early/2016/10/17/081380)][[code](https://github.com/ueser/FIDDLE)][[Youtube talk](https://www.youtube.com/watch?v=pcLTUsOm5pc&feature=youtu.be&list=PLlMMtlgw6qNjROoMNTBQjAcdx53kV50cS&t=2411)]

The group predicted transcription start site and regulatory regions but claims this solution could be easily generalized and predict other features too. FIDDLE stands for Flexible Integration of Data with Deep LEarning. The idea (nicely explained by the author in the YouTube video above) is to model several genomic signals jointly using convolutional networks. This could be for example DNase-seq, ATAC-seq, ChIP-seq, TSS-seq, maybe RNA-seq signals (as in .wig files with one value per base in the genome).

**Deep Learning Of The Regulatory Grammar Of Yeast 5′ Untranslated Regions From 500,000 Random Sequences** [[paper](http://genome.cshlp.org/content/27/12/2015)][[code](http://genome.cshlp.org/content/suppl/2017/11/02/gr.224964.117.DC1/Supplemental_code.tar.gz)]

This is a CNN model that attempts to predict protein expression from the DNA sequence in a specific type of genomic region called 5' UTR (five-prime untranslated region). The model is built in Keras and a nice touch by the authors is that they optimized the parameters using hyperopt, which is also shown in one of the Jupyter notebooks that comes along with the paper. The results look promising and easily reproducible, judging from my own trial.

**Modeling Enhancer-Promoter Interactions with Attention-Based Neural Networks** [[bioRxiv preprint](https://www.biorxiv.org/content/early/2017/11/14/219667)][[code](https://github.com/wgmao/EPIANN)]

The concept of attention in (recurrent) neural networks has become quite popular recently, not least because it has been used to great effect in machine translation models. This paper proposes an attention-based model for getting at the interactions between enhancer sequences and promoter sequences.

**Predicting Transcription Factor Binding Sites with Convolutional Kernel Networks** [[bioRxiv preprint](https://www.biorxiv.org/content/early/2017/11/10/217257)][[code](https://gitlab.inria.fr/dchen/CKN-seq)]

This paper uses a hybrid of CNNs (to learn good representations) and kernel methods (to learn good prediction functions) to predict transcription factor binding sites.

**Predicting DNA accessibility in the pan-cancer tumor genome using RNA-seq, WGS, and deep learning** [[bioRxiv preprint](https://www.biorxiv.org/content/early/2017/12/05/229385)]

Like Basset (above) this paper shows how to predict DNA accessibility from sequence using CNNs, but it adds the possibility to leverage RNA sequencing data from different cell types as input. In this way implicit information related to cell type can be "transferred" to the accessibility prediction task.


### Non-coding RNA <a name='genomics_non-coding'></a>

**DeepLNC, a long non-coding RNA prediction tool using deep neural network** [[paper](http://link.springer.com/article/10.1007%2Fs13721-016-0129-2)] [[web server](http://bioserver.iiita.ac.in/deeplnc/)]

Identification of potential long non-coding RNA molecules from DNA sequence, based on k-mer profiles.

**A Deep Recurrent Neural Network Discovers Complex Biological Rules to Decipher RNA Protein-Coding Potential** [[github](https://github.com/hendrixlab/mRNN)][[paper](https://www.biorxiv.org/content/early/2017/11/13/200758.1)] 

From the abstract: *While traditional, feature-based methods for RNA classification are limited by current scientific knowledge, deep learning methods can independently discover complex biological rules in the data de novo. We trained a gated recurrent neural network (RNN) on human messenger RNA (mRNA) and long noncoding RNA (lncRNA) sequences. Our model, mRNA RNN (mRNN), surpasses state-of-the-art methods at predicting protein-coding potential.*

### Methylation <a name='genomics_methylation'></a>

**DeepCpG - Predicting DNA methylation in single cells**
[[paper](http://dx.doi.org/10.1186/s13059-017-1189-z)]
[[code](https://github.com/cangermueller/deepcpg)]
[[docs](http://deepcpg.readthedocs.io/en/latest/)]

DeepCpG is a deep neural network for predicting DNA methylation in multiple cells. DeepCpG has a modular architecture, consisting of a recurrent CpG module to account for correlations between CpG sites within and across cells, a convolutional DNA module to extract patterns from a wide DNA sequence window, and a Joint module that integrates the evidence from the CpG and DNA module to predict the methylation state of multiple cells for a target CpG site. DeepCpG yields accurate predictions, enables discovering DNA sequence motifs that are associated with DNA methylation states and cell-to-cell variability, and can be used for analyzing the effect of single-nucleotide mutations on DNA methylation. DeepCpG is implemented in Python and publicly available.

**Predicting DNA Methylation State of CpG Dinucleotide Using Genome Topological Features and Deep Networks** [[paper](http://www.nature.com/articles/srep19598)][[web server](http://dna.cs.usm.edu/deepmethyl/)]

This implementation uses a stacked autoencoder with a supervised layer on top of it to predict whether a certain type of genomic region called “CpG islands” (stretches with an overrepresentation of a sequence pattern where a C nucleotide is followed by a G) is methylated (a chemical modification to DNA that can modify its function, for instance methylation in the vicinity of a gene is often but not always related to the down-regulation or silencing of that gene.) This paper uses a network structure where the hidden layers in the autoencoder part have a much larger number of nodes than the input layer, so it would have been nice to read the authors’ thoughts on what the hidden layers represent.

### Single-cell applications <a name='genomics_single-cell'></a>

**DeepCpG - Predicting DNA methylation in single cells**
[[paper](http://dx.doi.org/10.1186/s13059-017-1189-z)]
[[code](https://github.com/cangermueller/deepcpg)]
[[docs](http://deepcpg.readthedocs.io/en/latest/)]

See above.

**CellCnn – Representation Learning for detection of disease-associated cell subsets**
[[code](https://github.com/eiriniar/CellCnn)][[paper](http://biorxiv.org/content/early/2016/03/31/046508)]

This is a convolutional network (Lasagne/Theano) based approach for “Representation Learning for detection of phenotype-associated cell subsets.” It is interesting because most neural network approaches for high-dimensional molecular measurements (such as those in the gene expression category above) have used autoencoders rather than convolutional nets.

**DeepCyTOF: Automated Cell Classification of Mass Cytometry Data by Deep Learning and Domain Adaptation**[[paper](http://biorxiv.org/content/biorxiv/early/2016/05/31/054411.full.pdf)]

Describes autoencoder approaches (stacked AE and multi-AE) to gating (assigning cells into discrete groups) with mass cytometry (CyTOF).

**Using Neural Networks To Improve Single-Cell RNA-Seq Data Analysis**[[preprint](http://biorxiv.org/content/early/2017/04/23/129759)]

Tests a variety of neural network architectures for obtaining a reduced representation of single-cell gene expression data. Introduces a database of tens of thousands of single-cell profiles which can be queried to infer a cell type or state based on this reduced representation.

**Removal of batch effects using distribution-matching residual networks**[[code](https://github.com/ushaham/BatchEffectRemoval)][[paper](https://academic.oup.com/bioinformatics/article-abstract/doi/10.1093/bioinformatics/btx196/3611270/Removal-of-Batch-Effects-using-Distribution)]

Most high-throughput assays in genomics, proteomics etc. are affected to some extent by systematic technical errors, so-called "batch effects". This paper uses a residual neural network to attenuate batch effects by trying to match the distributions of replicate experiments on e.g. single-cell RNA sequencing or mass cytometry. 

**Active deep learning reduces annotation burden in automatic cell segmentation** [[bioRxiv preprint](https://www.biorxiv.org/content/early/2017/11/01/211060)]

Active learning, a framework addressing how to select training examples in order to train a model most efficiently, is shown to significantly reduce the time required by experts to annotate cell segmentation images in high-throughput high-context microscopy. Training deep learning models on this type of application of course requires a lot of high-quality labeled data, but the time of the human experts that can provide the labels (perform annotation) is limited and expensive. 

**scVAE: Variational auto-encoders for single-cell gene expression data** [[code](https://github.com/scvae/scvae)][[preprint](https://www.biorxiv.org/content/10.1101/318295v2)]

This approach models single-cell gene expression data directly from counts without initial normalization, and performs clustering in the latent space. Since it is based on a variational autoencoder, it can also be used to generate synthetic single-cell data by sampling from the latent distribution.


### Population genetics <a name='genomics_pop'></a>

**Deep learning for population genetic inference** [[code](https://sourceforge.net/projects/evonet/)][[paper](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004845)]

**Diet networks: thin parameters for fat genomics** [[manuscript](http://openreview.net/pdf?id=Sk-oDY9ge)]

This weirdly-named paper addresses the frequently encountered problem in genomics where the number of features is much larger than the number of training examples. Here, it is addressed in the context of SNPs (single-nucleotide polymorphisms, genetic variations between individuals). The authors propose a new network parametrization that reduces the number of free parameters using a multi-task architecture which tries to learn a useful embedding of the input features.

### Systems biology<a name='sysbio'></a>

**Using deep learning to model the hierarchical structure and function of a cell** [[web server](http://d-cell.ucsd.edu)][[paper](https://www.nature.com/articles/nmeth.4627/)]

In this ambitious paper, the authors attempt to construct an interpretable neural network model (VNN; visible neural network) of a eukaryotic cell based on millions of genotype-phenotype associations. The network is built in a hierarchy with 12 levels, where each level is supposed to reflect a biologically meaningful level of organization. The resulting model can predict, for a given genetic perturbation, what the resulting phenotype is likely to be.

## Neuroscience <a name='neuro'></a>

There are potentially lots of implementations that could go here.

**Deep learning for neuroimaging: a validation study** [[paper](http://journal.frontiersin.org/article/10.3389/fnins.2014.00229/abstract)]

**SPINDLE: SPINtronic deep learning engine for large-scale neuromorphic computing** [[paper](http://dl.acm.org/citation.cfm?id=2627625)]
