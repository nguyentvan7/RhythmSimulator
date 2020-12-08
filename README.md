# Rhythmic Pixel Simulator

(Insert blurb about rhythmic pixel regions here)

(Insert blurb about simulator purpose here)

(Insert citation for RP here)

## Getting Started

### Dependencies
Required libraries:  
- [fmt](https://github.com/fmtlib/fmt)
- [OpenCV](https://opencv.org/releases/)
- GCC version 6 or above

### Building
Simply run `make` in the root directory 

### Usage
`./rhythm-folder -o <OUTPUT_FOLDER> -r <REGION_FOLDER> [OPTIONS]`  
where `<OUTPUT_FOLDER>` specifies the desired folder to output statistics, memory traces and encoded/decoded images, and  
where `<REGION_FOLDER>` specifies the folder containing csv files specifying the regions per frame.
Options include:
- `-h, --help` for help
- `-i, --input <INPUT_FOLDER>` to specify the input folder. All images will be read in this folder.
- `-n, --nooutput <WIDTH> <HEIGHT>` to specify there should be no image output. No input image will be required in this case. `<WIDTH>` and `<HEIGHT>` describe the frame size of the intended input images.
- `-t, --notrace` to specify there should be no trace output.

### Examples  
- No image input/output, only memory trace generation. Output will be at `examples/output`.  
`./rhythm_folder -n 640 480 -r examples/csv -o examples/output`

- Full image input with encoded and decoded output, along with memory trace generation. Output will be at `examples/output2`.  
`./rhythm_folder -i examples/input2 -r examples/csv2 -o examples/output2`

- More examples to come soon
