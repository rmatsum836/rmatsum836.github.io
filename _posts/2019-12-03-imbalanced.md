---
layout: post
title: Oversampling with MLB Statcast Data
category:
- Data Science
- Software
excerpt: Oversampling to balance your dataset
--- 
When I tell someone I'm a PhD student in chemical engineering, most of the time they have the
impression that I spend my days taking classes and studying topics like chemical plant processes.  As a result, they are
usually surprised to hear that the majority of my day-to-day job consists of writing Python code.
They are often even more confused whenever I say that I aspire to be a data scientist after
graduating.

With my formal education in math and statistics only coming from my engineering background, this means
that there are many data science and machine learning concepts that I need to learn on my own.
Personally, it's easier for me to learn new concepts while working on projects rather than trying to
learn from a textbook.  Recently I've been working on a project to predict baseball pitches from MLB
Statcast data.  Sports have always been an interest of mine, and I've been looking for a data science
project to work on that incorporates it.  Baseball is a great sport for these types of projects due to
the massive amount of data that is collected.  To put it in perspective, each MLB stadium has a radar
that is able to collect 30+ pieces of data for a single pitch in a game.  With 30 MLB teams playing
162 games each year, you can imagine how big this dataset is.

## A Brief Exploration of MLB Statcast Data

Predicting baseball pitch types is a multi-class classification problem.  Based on the various columns
in the data set, we need to be able to predict the class, or type that the pitch belongs to.  In the
future, I will go over more details of the data, machine learning models, etc.  Though today, I'm
going to just be focusing on the issue of baseball pitches being imbalanced.  There are many types of
pitches in baseball: four-seam fastballs, two-seam fastballs, curveballs, sliders, etc.  If you can
imagine, these pitches are thrown at different rates.  To prove this, we can take a look at a count of
all the pitches for a portion of the MLB Statcast dataset.  To query the data, we are using the
[Pybaseball](https://github.com/jldbc/pybaseball) package.  We query data from September 20, 2019 to
October 6, 2019 with the following command:
```
from pybaseball import statcast

start = '2019-09-20'
end = '2019-10-06'
data = statcast(start, end)
```

By running `np.shape(data)` we see that our dataset consists of 45,643 data points with 90 different
columns.  Not bad for two weeks worth of data.  To get the types of pitches in the dataset, we execute
the following line of code:
```
set(pitch_types)
{'CH', 'CU', 'EP', 'FC', 'FF', 'FS', 'FT', 'KC', 'SI', 'SL'}
```
There are 10 pitch types in total.  Below are the descriptions for each pitch:

- `CH`: changeup
- `CU`: curveball
- `EP`: ephus
- `FC`: cutter
- `FF`: four-seam
- `FS`: splitter
- `FT`: two-seam
- `KC`: knuckle-curve
- `SI`: sinker
- `SL`: slider
- `KN`: knuckle
- `FO`: forkball
- `PO`: pitchout
- `SC`: screwball

We can get the count of each pitch type in our dataset with the following command:
```
data.groupby('pitch_type').count()['index']

pitch_type
CH     5085
CU     4790
EP        3
FC     2641
FF    16055
FS      656
FT     3760
KC      800
SI     3148
SL     8696
Name: index, dtype: int64
```

The results here are expected.  The majority of pitches in the data set are four-seam fastballs, which
are considered to be the bread and butter pitch for the majority of pitchers.  The four-seam fastball
is followed by the slider, changeup, and curveball, which are breaking balls.  Pitchers usually have
some combination of these three pitches, but they are thrown less often than the four-seam fastball.
One pitch that is thrown very infrequently is the eephus, which is characterized by a high trajectory
and low velocity.  This is a pitch we may consider dropping from the data set due to its infrequency.

This brings us to an inherent problem with this dataset: imbalance.  We have a large amount for
four-seam fastballs, but not alot of data for pitches like sinkers and splitters.  As a result, we may
introduce a frequency bias into a machine learning algorithm, meaning that more emphasis is placed on
classification types that occure more frequently.  To resolve this issue we can use either the
technique of oversampling, where we generate more examples of the minority classes, or undersampling,
where we remove instances of the majority classes.  We can implement both methods using a package
called [imbalanced-learn](https://imbalanced-learn.readthedocs.io/en/stable/).

## imbalanced-learn

imbalanced-learn was started by Fernando Nogueira in 2014, and was designed to implement the
synthetic minority oversampling technique (SMOTE) algorithm.  What's great about imbalanced-learn is
that it was designed to be fully compatible with [scikit-learn](https://scikit-learn.org/stable/).
Today, imbalanced-learn contains many oversampling and undersampling algorithms in addition to
SMOTE.  However, I'll just be focusing on SMOTE for this post.  I found this [blog
post](http://rikunert.com/SMOTE_explained) to give a good explanation of SMOTE, but essentially this
algorithm creates new data points for the minority classes in between existing data points.  Below,
I give a demonstration of how SMOTE in imbalanced-learn can improve the performance of a machine
learning model.

## Using SMOTE to help predict MLB Pitch Types

To evaluate the performance of SMOTE, I built a Random Forest classification model with
scikit-learn.  At the time of writing this post, I have only done a small amount of feature
engineering.  I created `movement_x` and `movement_z` that quantify the horizontal and vertical
movements of a pitch, using the equations given by [Alan
Nathan](http://baseball.physics.illinois.edu/Movement.pdf).  Additionally I created `h_vs_v`, a
ratio of the horizontal and vertical pitch movement, `h_vs_speed`, a ratio of the horizontal
movement and pitch speed, and `v_vs_speed`, a ratio of the veritcal movement and pitch speed.  At
this time I also haven't done much exploration of various classification algorithms or
hyperparameter tuning.  As a result I'm not too concerned with the overall performance of the
model.  Right now I simply want to show how oversampling can improve model performance.

To start, I begin by querying the data from pybaseball.  This time however, I query data from May 1,
2018 to October 6, 2019.  After dropping and adding various features, the data set has x features
for this demonstration.  I've previously run into issues trying to overample with SMOTE with this
many features and a smaller data set, which is why I've queried a larger data set.  Now we have
1,334,045 data points to work with.

Besides the feature engineering explained above, there are a few more things done with the data
before feeding it into a model.  First, the pitch type strings are converted to integers with the
following code:

```
factor = pd.factorize(data['pitch_type'])
data['pitch_type'] = factor[0]
definitions = factor[1]

print(set(data.pitch_type))
print(definitions)

{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
Index(['FC', 'KC', 'SL', 'CH', 'FF', 'FT', 'CU', 'SI', 'FS', 'FO', 'KN', 'PO',
       'SC'],
      dtype='object')
```

Afterwards, the data is normalized using the `StandardScaler()` in scikit-learn:

```
scaler = sklearn.preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Now we build the random forest in scikit-learn:

```
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100,
                           oob_score=True,
                           random_state=0)
rf.fit(X_train, y_train)

rf_predictions = rf.predict(X_test)
```

The results can be seen in the confusion matrix below.  
We observe that the random forest model does
a good job at predicting the pitch types that have many data points: changeups, curveballs,
four-seam fastballs, and sliders.  
We can also take a look at the precision and recall for this model.  
Recall and precision are usually better metrics for imbalanced data and classification problems in general, as accuracy can usually lead to misleading results.
Recall is the defined as the number of true positives divded by the number of true positives and number of false negatives, which represents the ratio that true positives are correctly identified. 
Precision is defined as the number of true positives divided by the number of true positives and false positives, which represents the ratio that positive predictions are actually correct. 
The F1 score is essentially a weighted mean between recall and precision.  
To do calculate these metrics, we will import `metrics` from scikit-learn.

  <center><img style="margin: 0px 25px 20px 0px;" src="/images/blog/dec4/non_smote_confusion.png" width="800" height="800" /></center>
  <center><em> A confusion matrix of the Random Forest Classifier results with imbalanced data </em></center><br/>

```
from sklearn import metrics
print(metrics.classification_report(y_update, rf_update, target_names=sorted(set(y_update))))

              precision    recall  f1-score   support

          CH       0.91      0.95      0.93     50090
          CU       0.89      0.89      0.89     38776
          FC       0.85      0.72      0.78     28095
          FF       0.93      0.97      0.95    167880
          FO       0.00      0.00      0.00        25
          FS       0.88      0.62      0.73      6581
          FT       0.78      0.77      0.78     44518
          KC       0.90      0.72      0.80     11202
          KN       0.00      0.00      0.00         3
          PO       0.00      0.00      0.00        28
          SC       0.00      0.00      0.00        14
          SI       0.85      0.73      0.79     36646
          SL       0.87      0.93      0.90     75256

    accuracy                           0.89    459114
   macro avg       0.60      0.56      0.58    459114
weighted avg       0.89      0.89      0.89    459114
```

The model does a good job of predicting changeups, curveballs, four-seam fastballs, and sliders as indicated by their recall and precision
scores.  High recall means that the model is able to correctly identify a large proportion of actual positives for these pitches, while high
precision indicates that a high proportion of the predicted pitch types were actually correct.  Also identified by the confusion matrix,
there are pitchtypes such as cutters, two-seam fastballs, knucklecurves, and sinkers that the model does a decent job of predicting.  The
precision and recall scores for these pitches are between 0.6 and 0.9.  However, there are a number of pitches that have precision and
recall scores of zero: forkballs, knuckleballs, pitchouts, pitchouts, and screwballs.  These were pitches with little data, so hopefully
SMOTE can imrpove the predictions.

Now let's use SMOTE in imbalanced-learn to see if we can improve the performance of our model through
oversampling.  To use SMOTE oversampling we use the `SMOTE()` function within imbalanced-learn:

```
from imblearn.over_sampling import SMOTE

X_resampled, y_resampled = SMOTE(sampling_strategy='not majority', random_state=85).fit_resample(X_train,y_train)
```

Here, we set the `sampling_strategy` to `not majority`, which will resample all of the classes
except for the majority class.  In this case, the majority class is the four-seam fastball.  The
collections package in Python can be imported to view the count of each pitch type in `y_resampled`:

```
import collections

collections.Counter(y_resampled)

Counter({0: 311899,
         1: 311899,
         2: 311899,
         3: 311899,
         4: 311899,
         5: 311899,
         6: 311899,
         7: 311899,
         8: 311899,
         9: 311899,
         10: 311899,
         11: 311899,
         12: 311899})
```

Nice! SMOTE generated synthetic data points so that now each pitch type is equal with each other.
Now let's see if this improves the performance of the random forest classifier.  Let's run the random forest classifier again, now
with the resampled data from SMOTE.  The confusion matrix from the result are also shown down below.

```
# Run Random Forest Model
rf = RandomForestClassifier(n_estimators=100,
                           oob_score=True,
                           random_state=0)

rf.fit(X_resampled, y_resampled)

rf_predictions = rf.predict(X_test)
```

  <center><img style="margin: 0px 25px 20px 0px;" src="/images/blog/dec4/smote_confusion.png" width="800" height="800" /></center>
  <center><em> A confusion matrix of the Random Forest Classifier results with SMOTE</em></center><br/>

From the confusion matrix, we see mixed results.  Pitches that already had a sufficient number of data points
like cutters, knucklecurves, changeups, and four-seam fastballs are predicted with roughly the same accuracy.  However
pitches like sliders, two-seam fastballs, curveballs, and sinkers are predicted with much better accuracy.
There are still pitch types that aren't being predicted well at all, such as pitchouts and screwballs,
which is likely due to the small amount of data for these pitch types to begin with.  In fact, I originally didn't come across pitchouts when analyzing a smaller size of data, so this might be an inconsistent pitch type.  
Similar to what I did with the imbalanced data, I generated a list of precision, recall, and f1-scores.

```
print(metrics.classification_report(y_update, rf_update, target_names=sorted(set(y_update))))

          precision    recall  f1-score   support

          CH       0.93      0.93      0.93     50090
          CU       0.90      0.91      0.90     38776
          FC       0.77      0.83      0.80     28095
          FF       0.96      0.93      0.95    167880
          FO       0.31      0.16      0.21        25
          FS       0.75      0.81      0.78      6581
          FT       0.76      0.82      0.79     44518
          KC       0.83      0.84      0.83     11202
          KN       1.00      0.33      0.50         3
          PO       0.16      0.21      0.18        28
          SC       0.50      0.07      0.12        14
          SI       0.82      0.80      0.81     36646
          SL       0.91      0.89      0.90     75256

    accuracy                           0.89    459114
   macro avg       0.74      0.66      0.67    459114
weighted avg       0.89      0.89      0.89    459114
```

For the pitches that were already predicted well, SMOTE doesn't appear to have much of an effect.  This is expected, as there was already a
sufficient amount of data for these pitch types.  However, a decent amount of improvement is observed for the pitches that had zero recall
and precision.  For example, forkballs now have a precision of 0.31 and a recall of 0.16.  While these scores are still not good, they are
still quite an improvement.  Most likely, further improvement of the model will come from additional feature engineering, model exploration,
hyperparameter tuning and such.  

Overall, imbalanced classes is something to be mindful of when working on classification problems.  Oversampling and the use of imbalanced-learn can help improve the performance of imbalanced clases.
In addition to SMOTE, there are a variety of oversampling techniques contained within this package.  Though I didn't cover it, there are
also numerous undersampling techniques as well.  One other disclaimer I want to make is that I still consider myself to be a beginner when
it comes to data science and machine learning.  And as a result, there may have been design choices I made in this model that are incorrect
or was not the best implementation.  If you happen to catch anything like this, please feel free to reach out to me and let me know.
Thanks!

** At the time of this post, I am still working on cleaning up the Jupyter notebook used to build and run this model.  Once I do so, I will
publish the code online.
