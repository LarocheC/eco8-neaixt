# Build the ICASSP paper. `make` -> main.pdf
main.pdf: main.tex refs.bib spconf.sty IEEEbib.bst
	pdflatex -interaction=nonstopmode main
	bibtex main
	pdflatex -interaction=nonstopmode main
	pdflatex -interaction=nonstopmode main

.PHONY: clean
clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf
