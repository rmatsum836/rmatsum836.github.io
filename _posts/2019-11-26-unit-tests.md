---
layout: post
title: My Experience With Unit Tests 
category:
- Software
excerpt: Why unit testing is an integral part of your next software package
---

Last week I spent most of my time developing new features in 
[mBuild](https://github.com/mosdef-hub/mbuild)
.  Doing this, there were a few times where the existing unit tests in these packages prevented me from pushing code with breaking changes to GitHub.  In essence, unit tests saved me from looking (really) dumb to my peers.
Before I continue, let's first go over what a unit test is.

  <center><img style="margin: 0px 25px 20px 0px;" src="/images/blog/nov26/seinfeld.jpg" width="300" height="300" /></center>
  <center><em> What's the deal with unit tests?! </em></center>

### What is a Unit Test

According to
[softwaretestingfundamentals.com](http://softwaretestingfundamentals.com/unit-testing/), 
"Unit Testing is a level of software testing where individivual units/components of
a software ares tested."  The reason for unit testing is to ensure that each piece of code
works as intended.  In most software packages, there are usually many
moving parts of the code in the form of various classes and functions.  Even in
mBuild and foyer which are relatively small packages, the codebase is diverse and large
enough to make unit testing a necessary component.

### How Unit Testing Helped Me

This week I specifically wanted to
make an addition to our LAMMPS writer in mBuild.  This writer
is a function that writes out the information contained in a ParmEd
structure to a file that can be used to run a simulation in LAMMPS.  In particular
there were two changes I wanted to implement: (1) Refactor the function by moving
parts of the code into several helper functions and (2) Add the functionality to
write out the information in Lennard Jones (LJ) units.  Currently the function only
writes out system information in the `real` set of units, which
requires few unit conversions from the set of units mBuild operates in.  On the
other hand, converting to LJ units is more involved and requires that we
keep track of multiple unit conversion methods.  To do so, I added in a set of
conversion factors into the code, which are determined based on the `unit_style`
defined by the user.

When I finished making these additions to the code, I ran the unit tests for the
[LAMMPS
writer](https://github.com/mosdef-hub/mbuild/blob/master/mbuild/tests/test_lammpsdata.py).
If you look at the code, you'll see that the tests ensure that a variety of structures
are properly written out to LAMMPS input files of varying formats.  When I ran
this set of
tests, I was shocked to find that the majority of them failed.  However this was
beneficial because these tests pointed me to the exact line in the function where
the error occurred.  In this case, I was trying to call an attribute of a class that
didn't actually exist.  Making this change was an easy fix, and afterwards I ran
the tests again.  Turns out there was another error!  This time, the error occurred
when I was multipling and dividing the conversion factors to convert the units of
the parameters.  Specifically, several paramter sets were contained in Python Lists,
and I caused an error when I tried to multiply and divide these Lists with the
conversion factors, which are integers.  To fix this, I simply converted these lists
to numpy arrays.  Afterwards all of the tests passed which ensured that my additions
weren't breaking the existing code.  Next I wrote new unit tests to ensure that the 
various unit conversions to LJ units are correct.

This experience reveals several benefits of unit testing.  For one, unit testing allowed me
the confidence to refactor and make additions without too much fear of breaking things.  This
is because when things actually did break, the unit tests were able to quickly detect these
errors introduced by the changes and I was able to easily fix them. This leads into the next
benefit of unit testing:  
writing unit tests saves time during development.  Sure it takes time to write unit
tests.  But once they are written, they make error detection easier and faster to fix in the
future.  Last but not least, its much less costly to fix errors during unit testing, rather
than trying to fix them after a new software version is released.  In my situation,
running the unit tests and having them fail wasn't a big deal.  Rather, had I pushed and merged
these changes into the master branch of the code, I would have introduced breaking changes
the workflows for various users of the software.  Not to mention, this would have been an
embarassing mistake on my part.  

### How to write a unit test in Python

I use [pytest](https://docs.pytest.org/en/latest/) to write all of my unit tests in Python.
There are already many great blogs on pytest and what make it great, which I will share below.
What makes pytest great in my opinion though is the ability to easily write tests just with
Python functions and `assert` statements.  To provide a quick example, I will use the
[calculator](https://github.com/mattwthompson/calculator) package created by Matt Thompson.
This package contains several functions that perform simple mathematic operations for the
purpose of showing how unit testing with Pytest works.  Let's take a look at `test_add()`
below:

```
def test_add():
    coolcalc = CoolCalc()
    a = 1
    b = 1
    mysum = coolcalc.add_a_b(a, b)
    assert mysum == 2
```

This test is checking to ensure that the method `add_a` is working as intended.  `add_a` is
simply a method that adds two numbers, `a` and `b`, together and returns the sum.  The test
will pass if the method returns a sum of 2.

A couple more things to note.  Tests in Pytest are represented as functions with a naming convention
that begins with `test_`.  Any function with this naming convention will be automatically run
when Pytest is called.  Next, tests are usually written with the `assert` statement in Python.
`assert` checks that the conditional statement following is True.

### Unit Testing Resources

Here are several posts that go over unit testing in greater detail:
- [https://pythontesting.net/framework/pytest/pytest-introduction/](https://pythontesting.net/framework/pytest/pytest-introduction/)
- [https://www.guru99.com/pytest-tutorial.html](https://www.guru99.com/pytest-tutorial.html)
- [https://changhsinlee.com/pytest-intro/](https://changhsinlee.com/pytest-intro/)

### Conclusions

In conclusion, the existing unit tests within mBuild saved me alot of time and embarassment
last week.  If you are developing a new software package, I highly recommend building up a
solid collection of unit tests.  One last point to mention is that unit testing (and testing
software in general) is a concept
and skill that many companies look for when hiring software engineers, data scientists, etc.
Understanding how to write a good unit test will make you a more qualified candidate for these
positions.
