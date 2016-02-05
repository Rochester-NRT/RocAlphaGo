## Git guide for this project

Get familiar with git's collaboration model - there are 
[plenty](http://rogerdudler.github.io/git-guide/) 
[of](https://guides.github.com/introduction/flow/) 
[resources](https://www.atlassian.com/git/tutorials/syncing) 
for this!

Fork this repository, and push all your changes to your copy. Make sure your branch is up to date with the central repository before making a pull request. [Git-scm](https://git-scm.com/book/en/v2/Distributed-Git-Distributed-Workflows#Integration-Manager-Workflow) describes this model well.

Follow these guidelines in particular:

1. write useful commit messages
1. `commit --amend` or `rebase` to avoid publishing a series of "oops" commits ([read this](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History))
1. ..but don't modify published history
1. prefer `rebase upstream/master` to `merge upstream/master`, again for the sake of keeping histories clean