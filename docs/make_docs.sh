#!/bin/bash
sphinx-apidoc -o . ..
make html
open -a "Google Chrome" _build/html/index.html