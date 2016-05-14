# RocAlphaGo

(Previously known just as "AlphaGo," renamed to clarify that we are not affiliated with DeepMind)

This project is a student-led replication/reference implementation of DeepMind's 2016 Nature publication, "Mastering the game of Go with deep neural networks and tree search," details of which can be found [on their website](http://deepmind.com/alpha-go.html). This implementation uses Python and Keras - a decision to prioritize code clarity, at least in the early stages.

[![Build Status](https://travis-ci.org/Rochester-NRT/RocAlphaGo.svg?branch=develop)](https://travis-ci.org/Rochester-NRT/RocAlphaGo)
[![Gitter](https://badges.gitter.im/Rochester-NRT/RocAlphaGo.svg)](https://gitter.im/Rochester-NRT/RocAlphaGo?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

# Documentation

See the [project wiki](https://github.com/Rochester-NRT/RocAlphaGo/wiki).

MicrosoftGeekHub Build Badge:

[![Build Status](https://travis-ci.org/Rochester-NRT/AlphaGo.svg?branch=develop)](https://travis-ci.org/MicrosoftGeekHub/AlphaGo)
[![Gitter](https://badges.gitter.im/MicrosoftGeekHub/AlphaGo.svg)](https://gitter.im/MicrosoftGeekHub?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

# Current project status

_This is not yet a full implementation of AlphaGo_. Development is being carried out on the `develop` branch.

Selected data (i.e. trained models) are released in our [data repository](http://github.com/Rochester-NRT/RocAlphaGo.data).

This project has primarily focused on the neural network training aspect of DeepMind's AlphaGo. We also have a simple single-threaded implementation of their tree search algorithm, though it is not fast enough to be competitive yet.

See the wiki page on the [training pipeline](https://github.com/Rochester-NRT/RocAlphaGo/wiki/Neural-Networks-and-Training) for information on how to run the training commands.

# How to contribute

See the ['Contributing'](CONTRIBUTING.md) document and join the [Gitter chat](https://gitter.im/Rochester-NRT/RocAlphaGo).
