# Contributing

<!-- SPDX-License-Identifier: (Apache-2.0) -->

Feedback and contributions are very welcome!

Following sections provide help on how to make contributions:

## General information

For specific proposals, please provide them as
[pull requests](https://github.com/inferenceengine/shadernn/pulls)
or
[issues](https://github.com/inferenceengine/shadernn/issues)
via our
[GitHub site](https://github.com/inferenceengine/shadernn).


The "docs/" directory has information you may find helpful, for example:

-   [Getting-Started.md](docs/Getting-Started.md) provides a quick start guide
-   [Implement-Model-Processor.md](docs/Developer-Guide/Implement-Model-Processor.md) provides details for implement a processor for a new model
-   [Load-and-run-model.md](docs/Developer-Guide/Load-and-run-model.md) provides details for how to load and run a model
-   [Validate-Results.md](docs/Developer-Guide/Validate-Results.md) provides details for how to validate the results
-   [Benchmarking.md](docs/Developer-Guide/Benchmarking.md) provides details for how to do benchmarking tests

See [CODE OF CONDUCT](./CODE_OF_CONDUCT.md) for our code of conduct;

### Pull requests and different branches recommended

Pull requests are preferred, since they are specific.
For more about how to create a pull request, see
<https://help.github.com/articles/using-pull-requests/>.

We recommend creating different branches for different (logical)
changes, and creating a pull request when you're done into the main branch.
See the GitHub documentation on
[creating branches](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/)
and
[using pull requests](https://help.github.com/articles/using-pull-requests/).

### How we handle proposals
We use GitHub to track proposed changes via its
[issue tracker](https://github.com/inferenceengine/shadernn/issues) and
[pull requests](https://github.com/inferenceengine/shadernn/pulls).
Specific changes are proposed using those mechanisms.
Issues are assigned to an individual, who works it and then marks it complete.
If there are questions or objections, the conversation area of that
issue or pull request is used to resolve it.

### Two-person review

Our policy is that at least 50% of all proposed modifications will be reviewed
before release by a person other than the author,
to determine if it is a worthwhile modification and free of known issues
which would argue against its inclusion
(per the Gold requirement two_person_review).

We achieve this by splitting proposals into two kinds:

1. Low-risk modifications.  These modifications are being proposed by
   people authorized to commit directly, pass all tests, and are unlikely
   to have problems.  These include documentation/text updates
   (other than changes to the criteria) and/or updates to existing gems
   (especially minor updates) where no risk (such as a security risk)
   have been identified.  The project lead can decide that any particular
   modification is low-risk.
2. Other modifications.  These other modifications need to be
   reviewed by someone else or the project lead can decide to accept
   the modification.  Typically this is done by creating a branch and a
   pull request so that it can be reviewed before accepting it.

### Developer Certificate of Origin (DCO)

All contributions (including pull requests) must agree to
the [Developer Certificate of Origin (DCO) version 1.1](docs/dco.txt).
This is exactly the same one created and used by the Linux kernel developers
and posted on <http://developercertificate.org/>.
This is a developer's certification that he or she has the right to
submit the patch for inclusion into the project.

Simply submitting a contribution implies this agreement, however,
please include a "Signed-off-by" tag in every patch
(this tag is a conventional way to confirm that you agree to the DCO).
You can do this with <tt>git commit --signoff</tt> (the <tt>-s</tt> flag
is a synonym for <tt>--signoff</tt>).

Another way to do this is to write the following at the end of the commit
message, on a line by itself separated by a blank line from the body of
the commit:

````
Signed-off-by: YOUR NAME <YOUR.EMAIL@EXAMPLE.COM>
````

You can signoff by default in this project by creating a file
(say "git-template") that contains
some blank lines and the signed-off-by text above;
then configure git to use that as a commit template.  For example:

````sh
git config commit.template ~/shadernn/git-template
````

It's not practical to fix old contributions in git, so if one is forgotten,
do not try to fix them.  We presume that if someone sometimes used a DCO,
a commit without a DCO is an accident and the DCO still applies.

### License (Apache License 2.0)

All (new) contributed material must be released
under the [Apache License 2.0](./LICENSE).

### No trailing whitespace

Please do not use or include trailing whitespace
(spaces or tabs at the end of a line).
Since they are often not visible, they can cause silent problems
and misleading unexpected changes.
For example, some editors (e.g., Atom) quietly delete them by default.

## <span id="how_to_report_vulnerabilities">Vulnerability reporting (security issues)</a>

Please privately report vulnerabilities you find, so we can fix them!

See [SECURITY.md](./SECURITY.md) for information on how to privately report vulnerabilities.

## Documentation changes

Most of the documentation is in "markdown" format.
All markdown files use the .md filename extension.

Where reasonable, limit yourself to Markdown
that will be accepted by different markdown processors
(e.g., what is specified by CommonMark or the original Markdown)
In practice we use
the version of Markdown implemented by GitHub when it renders .md files,
and you can use its extensions
(in particular, mark code snippets with the programming language used).
This version of markdown is sometimes called
[GitHub-flavored markdown](https://help.github.com/articles/github-flavored-markdown/).
In particular, blank lines separate paragraphs; newlines inside a paragraph
do *not* force a line break.
Beware - this is *not*
the same markdown algorithm used by GitHub when it renders
issue or pull comments; in those cases
[newlines in paragraph-like content are considered as real line breaks](https://help.github.com/articles/writing-on-github/);
unfortunately this other algorithm is *also* called
GitHub rendered markdown.
(Yes, it'd be better if there were standard different names
for different things.)

The style is basically that enforced by the "markdownlint" tool.
Don't use tab characters, avoid "bare" URLs (in a hypertext link, the
link text and URL should be on the same line), and try to limit
lines to 80 characters (but ignore the 80-character limit if that would
create bare URLs).
Using the "rake markdownlint" or "rake" command
(described below) implemented in the development
environment can detect some problems in the markdown.
That said, if you don't know how to install the development environment,
don't worry - we'd rather have your proposals, even if you don't know how to
check them that way.

Do not use trailing two spaces for line breaks, since these cannot be
seen and may be silently removed by some tools.
Instead, use <tt>&lt;br&nbsp;/&gt;</tt> (an HTML break).


