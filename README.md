
# Lab_fall2021

* * *
### 08/24/2021


Three challenges in automatic mixing:

* objective evaluation, no one single way to approach a good mix
* different plugins used in practice
* interdependency between audio effects in the signal chain

Current output parameters:

1. Loudness difference between the vocal track and the backing track
2. dynamic range of the vocal track
3. Loudness difference between the vocal track and the backing track in each frequency band


To-Do:
1. define the question
2. define a set of output parameters
3. get dataset
4. define ground truth
5. the baseline system (voice power spectrum estimate?)



* * *

### 09/02/2021

To-Do:

1. More parameters
2. Find a source seperation system

* * *

### 09/09/2021

Last Week:

1. Came up with more parameters
2. Tried

To-Do:

1. More parameters
2. Find a source seperation system

* * *

### 09/16/2021

Last Week:

1. VGG
2. Finalized output parameters

To-Do:

1. Prepare groundtruth data
2. Find some low-level features
3. SVM regression or NN
4. Sliding window problem...

Low-level features => SVM
VGG embeddings => SVM
Source Separation embeddings => SVM


* * *

### 09/23/2021

Last Week:

1. Built the groundtruth data
2. Found more possible dataset
2. Some problems with SVM


To-Do:

1. try subsamples of each track, make SVM run
2. dual output/one model or single output/two models?
3. a few low level features
4. relative loudness against the mixture loudness?


* * *

### 9/30/2021


Last Week:

1. Changed the target ground truth from absolute loudness to relative loudness between the mixture loudness
2. Evaluated the computational cost
3. Own feature, comb filter?



To-Do:

1. results on subset of the data for SVD, using standard features
2. input should be normalized... and include mixture loudness
2. Normalized output from [0 to -20,30] to [0,1]
2. Larger dataset and subsample(medlyDB, MusDB)
3. 1dB step on only one song, to test the training accuracy(no data split, supposes to have very good results)


* * *

### 10/7/2021


Last Week:

1. No result from linear regression
2. Good result from XGBregressor with unsplit data

To-Do:
1. proper metric: absolute error
2. how different files have different results, maximum error of each file
3. listen, explain the ground truth and the peaks
4. some more features
5. new ideas

* * *

### 10/14/2021

last Week:

1. Bad results with proper data split
2. more features?
3. more data?


To-Do:

1. Break the chain?
2. Max error
3. dataset distribution of mix ratio
4. pipeline
5. take a look at extreme cases
6. a fake prediction from the mean value of the training set, the worst case


* * *

### 10/21/2021

last Week:

1. pipeline finshied, including ground truth generation (for MIR-1K dataset format only, but easy to modify), feature extraction, and machine learning models.
2. Created a mean value predictor, and the result is not bad...

Max error
dataset distribution of mix ratio
different normalization?
Break the chain?
