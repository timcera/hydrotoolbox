
# boughton chapman cm eckhardt ewma fixed furey lh local slide ukih willems


echo "chapman"
time hydrotoolbox baseflow_sep chapman --print_input < example.csv > example_chapman.csv
plottoolbox time --start_date 2018-01-01 --ofilename chapman.png < example_chapman.csv

echo "cm"
time hydrotoolbox baseflow_sep cm --print_input < example.csv > example_cm.csv
plottoolbox time --start_date 2018-01-01 --ofilename cm.png < example_cm.csv

echo "eckhardt"
time hydrotoolbox baseflow_sep eckhardt --print_input < example.csv > example_eckhardt.csv
plottoolbox time --start_date 2018-01-01 --ofilename eckhardt.png < example_eckhardt.csv

echo "ihacres"
time hydrotoolbox baseflow_sep ihacres 0.9 0.9 0.9 --print_input < example.csv > example_ihacres.csv
plottoolbox time --start_date 2018-01-01 --ofilename ihacres.png < example_ihacres.csv

echo "ewma"
time hydrotoolbox baseflow_sep ewma --print_input < example.csv > example_ewma.csv
plottoolbox time --start_date 2018-01-01 --ofilename ewma.png < example_ewma.csv

echo "usgs_hysep_fixed"
time hydrotoolbox baseflow_sep usgs_hysep_fixed --area 59.1 --print_input < example.csv > example_fixed.csv
plottoolbox time --start_date 2018-01-01 --ofilename fixed.png < example_fixed.csv

echo "furey"
time hydrotoolbox baseflow_sep furey --print_input < example.csv > example_furey.csv
plottoolbox time --start_date 2018-01-01 --ofilename furey.png < example_furey.csv

echo "usgs_hysep_local"
time hydrotoolbox baseflow_sep usgs_hysep_local --area 60 --print_input < example.csv > example_local.csv
plottoolbox time --start_date 2018-01-01 --ofilename local.png < example_local.csv

echo "usgs_hysep_slide"
time hydrotoolbox baseflow_sep usgs_hysep_slide --print_input < example.csv > example_slide.csv
plottoolbox time --start_date 2018-01-01 --ofilename slide.png < example_slide.csv

echo "ukih"
time hydrotoolbox baseflow_sep ukih --print_input < example.csv > example_ukih.csv
plottoolbox time --start_date 2018-01-01 --ofilename ukih.png < example_ukih.csv

echo "willems"
time hydrotoolbox baseflow_sep willems --print_input < example.csv > example_willems.csv
plottoolbox time --start_date 2018-01-01 --ofilename willems.png < example_willems.csv

echo "lh"
time hydrotoolbox baseflow_sep lh --print_input < example.csv > example_lh.csv
plottoolbox time --start_date 2018-01-01 --ofilename lh.png < example_lh.csv

echo "five_day"
time hydrotoolbox baseflow_sep five_day --print_input < example.csv > example_five_day.csv
plottoolbox time --start_date 2018-01-01 --ofilename five_day.png < example_five_day.csv

echo "strict"
time hydrotoolbox baseflow_sep strict --print_input < example.csv > example_strict.csv
plottoolbox time --start_date 2018-01-01 --ofilename strict.png < example_strict.csv

echo "boughton"
time hydrotoolbox baseflow_sep boughton --print_input < example.csv > example_boughton.csv
plottoolbox time --start_date 2018-01-01 --ofilename boughton.png < example_boughton.csv
