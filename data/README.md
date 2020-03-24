Please change line 214 and line 224 to your path before you run extract_rosbag_to_npy.py

use following code to run the script
python extract_rosbag_to_npy.py --bag indoor_flying1.bag --prefix indoor_flying1 --start_time 4.0 --max_aug 6 --n_skip 1 --output_folder [your location of the output folder]

and the start time are:
indoor_flying1---4.0
indoor_flying2---9.0
indoor_flying3---7.0
indoor_flying4---6.0

outdoor_day1---3.0
outdoor_day2---45.0

outdoor_night1---0.0
outdoor_night2---0.0
outdoor_night3---0.0