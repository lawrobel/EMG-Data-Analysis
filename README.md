#  Classification of sEMG Hand Signals Using Machine Learning Methods
## What is this project about?
<p>In this project a framework is created for extracting out features from both the time and frequency domains of sEMG signals and classifying these signals with machine learning models such as random forest, SVM, and gradient boosting. Data visualization of the feature-extracted signals using the dimensionality-reduction techniques of t-SNE and PCA is also supported. These visualizations can be used to access how similar signals of a particular class are with respect to a selected number of features.</p>

### Supported Features

<ul><li><b>Autoregressive coefficients</b> </li></ul>
<ul><li><b>Wilson Amplitude</b></li></ul>
<ul><li><b>Root Mean Square</b></li></ul>
<ul><li><b>Variance</b></li></ul>
<ul><li><b>Waveform Length</b></li></ul>
<ul><li><b>Mean Absolute Value</b></li></ul>
<ul><li><b>Zero Crossing</b></li></ul>
<ul><li><b>Sign Slope Changes</b></li></ul>

### Supported Machine Learning Models

<ul><li><b>Linear and Quadratic Discriminant Analysis</b> </li></ul>
<ul><li><b>Support Vector Machine with RBF kernel</b></li></ul>
<ul><li><b>Random Forest and Extra Trees</b></li></ul>
<ul><li><b>Gradient Boosting</b></li></ul>

## Datasets description

<p> There are two datasets used in this notebook. For one dataset, two males and three females of the same age approximately (20 to 22-year-old) conducted the six grasps for 30 times each. The measured time is 6 sec. For the other dataset, one male subject conducted the six grasps for 100 times each for 3 consecutive days. The measured time is 5 sec. Subjects were asked to perform these varies grasp movements while two electrodes were placed on their forearms to read the sEMG signals. The performed grasps were as follows:</p>
 <ul><li><b>Spherical</b> for holding spherical tools </li></ul>
<ul><li><b>Tip</b> for holding small tools</li></ul>
<ul><li><b>Palmar</b> for grasping with palm facing the object</li></ul>
<ul><li><b>Lateral</b> for holding thin, flat objects, </li></ul>
<ul><li><b>Cylindrical</b> for holding cylindrical tools </li></ul>
<ul><li><b>Hook</b> for supporting a heavy load</li></ul>
<p>This data is from the UCI Machine Learning Repository and is provided by Christos Sapsanis, Anthony Tzes, and G. Georgoulas. </p>


