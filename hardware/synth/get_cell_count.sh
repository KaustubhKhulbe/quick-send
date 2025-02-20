#/bin/bash

set -e

sed -nr 's/Number of cells: +?([0-9]+?)$$/\1/p' reports/area.rpt
