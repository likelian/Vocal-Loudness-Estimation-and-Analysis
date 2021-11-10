
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

1. pipeline finishied, including ground truth generation (for MIR-1K dataset format only, but easy to modify), feature extraction, and machine learning models.
2. Created a mean value predictor, and the result is not bad...
3. Max error
4. Under the same circumstance, z-score(StandardScaler) produces better results.
5. Under the same circumstance, chained the accompaniment before the vocal produces better results in all 4 metrics.
6. Unchained individual models are better than vox first, worse than acc first.
7. Double chaining gets the worse reuslts.
8. Computed the mean and std of the ground truth
9. Plot the histogram of the ground truth
10. Decreasing the training set size can improve the results of SVM, with fluctuation.



Best results for a small training set:

split before 1000 and after 1000
sub_X_train(1264, 22)
y_test(1000, 2)
StandardScaler
Mean value: [-3.1906353  -3.32109234]
Mean_value MAE_acc: 0.7575401288346398
Mean_value MAE_vox: 0.8486054807493958

Mean_value ME_acc: 4.2533477740951735
Mean_value ME_vox: 4.767215997357912

SVR training time: 0.19724297523498535

SVR_chained_acc_first MAE_acc: 0.6347617732628478
SVR_chained_acc_first MAE_vox: 0.6097516983387918

SVR_chained_acc_first ME_acc: 3.4882807540130503
SVR_chained_acc_first ME_vox: 4.5103523891910955


To-Do:

1. mean absolute error, max error and error histogram of each file
2. make it uniform distribution (data augmentation) (cut off at where we don’t hear anymore in the mix, or just too rare)
3. shuffle the results plot randomly, see why it is so smooth
4. two future directions: more parameter to extract, or large analysis

* * *

### 10/27/2021


Last week:


1. Shuffle the resulting plots(Learning is happening, but limited in the (-2, -4)dB range.
2. Individual error histogram
3. Total error histogram (data point level)
4. Total error histogram (file level)
5. averge error over the file level
6. More uniform distribution (kind of)


error_mean_value_average
vox_MAE        acc_MAE     vox_ME       acc_ME
[ 0.88644353  1.16208117  3.39344598 13.47482205]


error_SVR_average
vox_MAE        acc_MAE     vox_ME       acc_ME
[ 0.77124489  0.99434226  3.02321448 13.07296623]


Next week:

1. Verify if it is time dependent, compute the error again for the shuffled results, to eliminate any bug, and plot them together (a process of debug by looking at the result)
2. Improve the model, push the ME to 0
3. make the plot of the same scale resolution
4. Data augmentation

Extract neighbor features?


* * *

### 11/04/2021


Last week:

1. Shuffled reuslts
2. as acc distribution gets more uniform (towards the lower value), vox distribution move towards 0dB


Next Week:

1. Extending the dataset
    1. musdb_ dataset (easy one)
    2. multitrack datasets
2. more features?
3. plot histogram of x-axis vocal gain, y-axis mean average error over file
4. Decide the direction (after extending the dataset) (it’s ok to move forward with some issues, not lying), pick one option, then contact over teams immediatlly,
take some small steps
    1. improve (VGG?)
    2. frequency ratio, new direction
    3. study on many songs over history
5.




* * *

### 11/11/2021

Last week:

1. Get the MUSDB dataset and running, results are funny but not surprising
    1. Because the ground  truth distribution is different, it's hard to compare with the previous experiments, need to evalute on the same dataset
2. Do the truncation below -15dB
3. Scaling the input ground truth for training (-15, 0) to (0, 1) getting worse MAE, but it fits better for edge cases. Better ME result.
4. apply MuSDB trained model to MIR-1k, bad results


apply MuSDB trained model to MIR-1k

VGG?


* * *

MUSDB dataset, evaluated on 30 files and the subset of the rest as trainining set(around 36000 data points)
error_mean_value_average
vox_MAE        acc_MAE     vox_ME       acc_ME
[  1.10161774  68.81741637  10.02292489 150.19677982]


error_SVR_average
vox_MAE        acc_MAE     vox_ME       acc_ME
[  0.75647311  41.93878386  10.1346816  168.52597514]


From MUSDB to MIR-1k, evaluate on the first 30 files

error_mean_value_average
vox_MAE        acc_MAE     vox_ME       acc_ME
[ 1.10951481  4.92782653  3.88809626 18.30719372]

error_SVR_average
vox_MAE        acc_MAE     vox_ME       acc_ME
[ 1.39899926  4.28260407  4.63746981 19.10009475]


Cheating!!!!
From MUSDB to MIR-1k, evaluate on the first 30 files
Some MIR-1K samples are mixed with in the training set, including the test set. So, a lot better, yeah,,


error_mean_value_average
vox_MAE        acc_MAE     vox_ME       acc_ME
[0.87822128 2.68761465 3.42661131 7.21142468]
error_SVR_average
vox_MAE        acc_MAE     vox_ME       acc_ME
[1.13083139 1.89660596 3.15996962 7.43217401]

* * *
