# NPU-friendly LiSenNet — ICASSP paper

**Main document: `main.tex`** (on Overleaf, if a different file gets picked, set
it via Menu → Main document). Compiler: pdfLaTeX; bibliography: BibTeX.

| file | what it is |
| --- | --- |
| `main.tex` | the paper |
| `refs.bib` | bibliography (uncertain fields flagged `TODO verify`) |
| `spconf.sty`, `IEEEbib.bst` | the ICASSP author kit — vendored, do not edit |
| `figures/architecture.pdf` | Fig. 1, pre-built (this is what `main.tex` includes) |
| `figures/architecture.tex` | standalone TikZ source of Fig. 1 |

Red `[TODO: …]` markers flag what is still open: the author block, the
relu6-deep on-board rows of Table 2, the hybrid-decoder on-board latency, and
Fig. 2.

## Building locally

```bash
make          # main.pdf (rebuilds figures/architecture.pdf if its source changed)
make clean
```

Fig. 1 alone: `cd figures && pdflatex architecture.tex`. On Overleaf it is *not*
recompiled — the built PDF is what ships; edit the figure locally (or by
temporarily setting `figures/architecture.tex` as the main document) and
re-upload the PDF.

## Overleaf

The paper lives on the local-only `paper` branch and is **not** pushed to
`origin`. Two ways to get it into Overleaf:

**1. Upload a zip** (works on any plan). Zip the contents of this directory
*without* the build artifacts, then in Overleaf: New Project → Upload Project.

```bash
zip -r /tmp/paper.zip main.tex refs.bib spconf.sty IEEEbib.bst README.md \
    figures/architecture.pdf figures/architecture.tex
```

**2. Git bridge** (needs an Overleaf plan with Git integration). The bridge can
only push to a project that already exists, so create a blank project first,
copy its URL from Menu → Git, and generate a token under Account Settings → Git
integration. Then:

```bash
echo '<token>' > ~/.overleaf_token && chmod 600 ~/.overleaf_token
./push_to_overleaf.sh https://git.overleaf.com/<project-id> "Update paper"
```

The script copies only this directory's files into the Overleaf project's own
git history (it never pushes `eco8-neaixt` history anywhere), and feeds the
token through `GIT_ASKPASS`, so it never lands in a URL, in git config, or in
shell history.
