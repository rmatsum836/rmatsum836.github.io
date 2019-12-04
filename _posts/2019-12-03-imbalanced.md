---
layout: post
title: Oversampling with MLB Statcast Data
category:
- Data Science
- Software
excerpt: Oversampling to balance your dataset
---

When I tell someone I'm a PhD student in chemical engineering, most of the time they have the
impression that I'm studying chemical plant processes and such in great detail.  As a result, they are
usually surprised to hear that the majority of my day-to-day job consists of writing Python code.
They are usually even more confused whenever I say that I aspire to be a data scientist after
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
- `CU`:  curveball
- `EP`: ephus
- `FC`: cutter
- `FF`: four-seam
- `FS`: splitter
- `FT`: two-seam
- `KC`: knuckle-curve
- `SI`: sinker
- `SL`: slider

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
