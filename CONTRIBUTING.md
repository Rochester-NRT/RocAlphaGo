## Git guide for this project

Get familiar with git's collaboration model - there are 
[plenty](http://rogerdudler.github.io/git-guide/) 
[of](https://guides.github.com/introduction/flow/) 
[resources](https://www.atlassian.com/git/tutorials/syncing) 
for this!

Fork this repository, and push all your changes to your copy. Make sure your branch is up to date with the central repository before making a pull request. [Git-scm](https://git-scm.com/book/en/v2/Distributed-Git-Distributed-Workflows#Integration-Manager-Workflow) describes this model well.

Follow these guidelines in particular:

1. keep `upstream/master` functional
1. write useful commit messages
1. `commit --amend` or `rebase` to avoid publishing a series of "oops" commits (better done on your own branch, not `master`) ([read this](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History))
1. ..but don't modify published history
1. prefer `rebase master` to `merge master`, again for the sake of keeping histories clean. Don't do this if you're not totally comfortable with how `rebase` works.
1. track issues via GitHub's tools

## Coding guide

We are using Python 2.7. I recommend using `virtualenv` to set up an environment; [requirements.txt](requirements.txt) contains the necessary python modules. Beyond that, follow these guidelines:

1. remember that ["code is read more often than it is written"](https://www.python.org/dev/peps/pep-0008)
1. avoid premature optimization. instead, be pedantic and clear with code and we will make targeted optimizations later using a profiler
1. write [tests](https://docs.python.org/2/library/unittest.html). These are scripts that essentially try to break your own code and make sure your classes and functions can handle what is thrown at them
1. [document](http://epydoc.sourceforge.net/docstrings.html) every class and function, comment liberally