# AlphaGoReplication

A replication of DeepMind's 2016 Nature publication, "Mastering the game of Go with deep neural networks and tree search," details of which can be found [on their website](http://deepmind.com/alpha-go.html).

[![Build Status](https://travis-ci.org/Rochester-NRT/AlphaGo.svg?branch=develop)](https://travis-ci.org/Rochester-NRT/AlphaGo)

# Current project status

_This is not yet a full implementation of AlphaGo_. Development is being carried out on the `develop` branch.

We are still in early development stages. We hope to have a functional training pipeline complete by mid March, and following that we will focus on optimizations.

# Installation instructions

Using a [virtual environment](http://docs.python-guide.org/en/latest/dev/virtualenvs/) is recommended. If you have python 2.7 and `pip` installed, you can install all of the project dependencies with

	pip install -r requirements.txt

To verify that this worked, try running the tests

	python -m unittest discover
