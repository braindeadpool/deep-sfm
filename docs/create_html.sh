pandoc --filter pandoc-citeproc --bibliography=references.bib --variable classoption=twocolumn --variable papersize=a4pape -s report.md -t html -o index.html

