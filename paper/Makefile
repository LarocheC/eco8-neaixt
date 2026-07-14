# Build the ICASSP paper. `make` -> main.pdf
main.pdf: main.tex refs.bib spconf.sty IEEEbib.bst figures/architecture.pdf
	pdflatex -interaction=nonstopmode main
	bibtex main
	pdflatex -interaction=nonstopmode main
	pdflatex -interaction=nonstopmode main

figures/%.pdf: figures/%.tex
	cd figures && pdflatex -interaction=nonstopmode $*.tex

.PHONY: clean
clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf \
	      figures/*.aux figures/*.log
