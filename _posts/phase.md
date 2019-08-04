---
layout: post
title: Tutorial to Phase Separation Tutorial
excert: Guide to use image processing and analysis to detect phase separation.
redirect_from: phase-tutorial.md
---

For the first few years of my PhD research, I've spent the bulk of my time studying ionic
liquids solvated in various organic solvents.  Ionic liquids are essentially salts (pairs of
cations and anions) that are liquid at ambient conditions.  For energy storage applications,
ionic liquids are attractive electrolytes due to their high electrochemical stability, which
allows for devices such as batteries and supercapacitors to be operated at higher voltages.
We can quadratically increase the energy density of a device through a voltage increase.  If
you think about this in the sense of an electric vehicle, higher energy density allows the
electric vehicle to travel for further distances between charges.

The issues with ionic liquids stem from their slow transport properties, namely diffusivity
and conductivity.  Since the charging rates of supercapacitors are limited by transport, lower
diffusivity and conductivity can negatively affect the power density of a device.  Going back
to the electric vehicle example, we can think about power density as how quickly it takes for
an electric vehicle to charge.

This is where the organic solvents come into play.  It has been shown through various
techniques that solvation of ionic liquids can improve the diffusivity of the ions, which can
lead to faster charging rates of a supercapacitor.  We have recently conducted a screening
study where over 400 molecular dynamics (MD)simulations were run of an ionic liquid solvated in
various solvents at a range of concentrations.

This is where the motivation for developing this image analysis package started.  When we
first started running the MD simulations for this study, we quickly realized that some of the
systems were phase-separated.  That is, the ionic liquid and solvent are in two distinct
phases from eachother.  To get a visual idea of what's going on, we can view two snapshots of
a well-mixed system and phase-separated system below.

![homo](/images/blog/phase/homo.png)
![hetero](/images/blog/phase/hetero.png)

If we were only looking at 5-10 simulations, detecting phase separation would be simple.  We
could just visualize inspect each system individually.  However when we have over 400 systems,
this quickly becomes a tedious process.

In the last couple years, I've been amazed at the
image detection problems people have been able to solve with machine learning algorithms.  I
wanted to tackle this problem using a convolutional neural network (CNN) or logistic
regression, however my initial attempts were unsuccessful.  

I reached out to several members about advice on how to solve this problem from the Nashville
Data Nerds meetup.  Rob Harrigan and Stephen Bailey were kind enough to offer some feedback,
first letting me know that those machine learning approaches were overkill for such a problem.
Rather, the solution they offered was to use a mix of image processing and analysis to count
the number of connected components in each image.

### Image Processing
The reason for using image processing for this problem is to manipulate the images in way that
makes it easiest to later count the connected components in the system.  For these steps, I
used the scikit-image package in Python (https://scikit-image.org).  Below I will outline
the steps taken to process these images.

#### Convolution
The first step in our image processing is to convolve the images.  If you look at the raw
images above, there are some shadowing effects present from the rendering.  The convolution
process will smooth out the shadowing from the rendered image so that the image contains
mostly red or blue, and as little greys and blacks as possible.  The function to convolve the
image is:

        from skimage.io import imread
        import numpy as np

        image = imread('image.png')
        kernel_size = 10
        kernel = np.ones((kernel_size, kernel_size)) / kernel_size
        blurred = gaussian_filter(iamge, sigma=0.8)
        convolved = _convolveImage(image-0.8*blurred, kernel)

The resulting image looks like this:
![homo](/images/blog/phase/homo.png)

Here is a good blog post explaining how image convolution works and how the kernel affects the
convolution: http://setosa.io/ev/image-kernels/

#### Dominant Color Through K-Means Clustering
To continue with our image processing, we first need to determine the dominant color in each
of our images.  Since we looked at ionic liquid solutions at a wide range of concentrations,
sometimes the dominant color is blue and other times the dominant color is red.  To accomplish
this, I chose to use K-Means clustering through scikit-learn.  The idea here is that k-means
clustering will be applied to the image in RGB format.  The location of the cluster will
correspond to a color via an RGB array, which will determine the dominant color.  This will
determine how we later threshold the image into a binary image. 

#### Otsu's Method
In order to count the connected components of the image, we need a binary image of our system.
That is, the colors of our image is either black or white.  This is where Otsu's Method of
thresholding comes in, which will convert a grayscale image to a binary image. Otsu's Method
is just one of many binarization algorithms, and you can learn more about it here: http://www.labbookpages.co.uk/software/imgProc/otsuThreshold.html.  In the code below, we will convert the convolved imaged to grayscale and then call Otsu's method function.

        from image_process import apply_otsu

        gray = convolved[:,:,0]
        im_bw = apply_otsu(gray, )

#### Connected Components Analysis
Once we have a binary image, we are ready to move onto the image analysis portion of the
process.  First we will perform connected components labeling with scikit-image.  In connected
components labeling, the image is scanned for each pixel, to identify regions that have
similar pixel values.  Remember that since we are looking at a binary image, the pixels will
either have a value of 1 or 0.  To read more about this process, I suggest reading this
article: https://homepages.inf.ed.ac.uk/rbf/HIPR2/label.htm.  The label function in
scikit-image will return the labeled image array as well as the number of labels that were
identified.  At this point we have counted the number of components in our image.  Depending
on the case, we were counting the number of ionic liquid clusters or the number of solvent
clusters in the system.

One issue that came up during the development of this process is dealing with small clusters.
These are most likely artifacts from preparing the image and should not be counted as a
cluster.  As a result, the `_cutoff_particles` function is used to remove these clusters from
the count of each image.  In this case, the criteria used is that if a component's area was
less than 300, it was removed from the count.

At this point, we can plot and display the image with the connected components labeled with
the 'regionprops' function in scikit-image.  You can see the results of the two original
images below.  For the first image, we see that scikit-image detected a single large cluster
of fluid.  From this, we can infer that this system is phase separated.  In the second image,
we see many more components that have been identified and whose areas are much lesser in
comparison to the cluster in the first image.  As a result, we can conclude that this system
is much more well mixed and is a homogenous solution.


#### Results and Conclusions
When I implemented this package, I looped through thousands of MD snapshots rather than just
two.  With a greater sample size, I was able to plot a distribution of cluster counts.  In the
figure below, we see that the images of the phase separated systems exhibit a distribution of
lower cluster counts, while the images of the homogeneous systems exhitibit a distribution of
higher cluster counts.  This is great, as it shows that we are able to differentiate between
these two types of systems.

With an image processing and analysis process like this, we can detect systems that are phase
separated for studies that have large amounts of simulations.  While this study is highly
relevant to my PhD research, I was motiviated to do this project mostly by my own curiousity
in image data science.  I'm happy with the end result, and was able to learn a lot in the
process.  If you are interested in using this package and are unsure how, feel free to contact
me!
