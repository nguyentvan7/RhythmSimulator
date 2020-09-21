#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <cstdint>
#include <vector>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <experimental/filesystem>
#include <locale>
#include <chrono>

#define MEGABYTE 8000000.0
typedef cv::Point3_<uint8_t> Pixel;
Pixel zero(0, 0, 0); 
typedef struct Region{
    uint32_t x0;
    uint32_t x1;
    uint32_t y0;
    uint32_t y1;
    uint8_t stride;
	bool skip;
} region;
namespace fs = std::experimental::filesystem;

void print_usage();
std::tuple <int, int> get_px_index(uint32_t width, std::vector<uint8_t> rowmask, uint32_t row_offset, uint col);

bool region_sort(region a, region b) { return a.x0 < b.x0; }

int main(int argc, char *argv[]) {
	// Argument handling.
	std::string input_folder_name;
	std::string output_folder_name;
	std::string region_folder_name;
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
			print_usage();
			return 0;
		}
		else {
			if (strcmp(argv[i], "--input") == 0 || strcmp(argv[i], "-i") == 0) {
				i++;
				input_folder_name = argv[i];
			}
			else if (strcmp(argv[i], "--output") == 0 || strcmp(argv[i], "-o") == 0) {
				i++;
				output_folder_name = argv[i];
			}
			else if (strcmp(argv[i], "--region") == 0 || strcmp(argv[i], "-r") == 0) {
				i++;
				region_folder_name = argv[i];
			}
			else {
				std::cout << "Invalid argument. Use -h or --help flag to print usage.";
				return 0;
			}
		}
	}

	// Checking paths.
	// Check input folder.
	if (!fs::exists(input_folder_name)) {
		std::cout << "Input folder does not exist.";
		return 0;
	}
	if (!fs::is_directory(input_folder_name)) {
		std::cout << "Input is not a folder.";
		return 0;
	}
	// Create output folder.
	if (!fs::exists(output_folder_name)) {
		fs::create_directory(output_folder_name);
	}
	if (!fs::exists(output_folder_name + "/encoded")) {
		fs::create_directory(output_folder_name + "/encoded");
	}
	if (!fs::exists(output_folder_name + "/stats")) {
		fs::create_directory(output_folder_name + "/stats");
	}
	// Check region folder.
	if (!fs::exists(region_folder_name)) {
		std::cout << "Region folder does not exist.";
		return 0;
	}
	if (!fs::is_directory(region_folder_name)) {
		std::cout << "Region is not a folder.";
		return 0;
	}

	// Reading image and region file names.
	int filecount = 0;
	std::vector<std::string> imagelist;
	std::vector<std::string> regionlist;
	for (auto& p: fs::directory_iterator(input_folder_name)) {
		filecount++;
		imagelist.push_back(p.path());
	}
	for (auto& p: fs::directory_iterator(region_folder_name)) {
		regionlist.push_back(p.path());
	}
	std::sort(imagelist.begin(), imagelist.end());
	std::sort(regionlist.begin(), regionlist.end());

	// Storing previous data.
	std::vector<std::vector<std::vector<uint8_t>>> bitmasks;
	std::vector<cv::Mat> encoded_images;
	std::vector<std::vector<int>> row_offsets;

	// For statistics.
	uint64_t total_write_pixel_touches = 0;
	uint64_t total_write_bitmask_touches = 0;
	uint64_t total_write_row_offset_touches = 0;
	uint64_t total_write_bits = 0;
	uint64_t total_read_pixel_touches = 0;
	uint64_t total_read_bitmask_touches = 0;
	uint64_t total_read_row_offset_touches = 0;
	uint64_t total_read_bits = 0;
	int images = 0;

	std::cout << "Starting processing on " << filecount << " frames." << std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	
	for (int file = 0; file < filecount; file++) {
		std::string image_path = imagelist[file];
		std::string region_path = regionlist[file];
		uint32_t write_pixel_touches = 0;
		uint32_t write_bitmask_touches = 0;
		uint32_t write_row_offset_touches = 0;
		uint32_t read_pixel_touches = 0;
		uint32_t read_bitmask_touches = 0;
		uint32_t read_row_offset_touches = 0;

		std::vector<Pixel> encoded_pixels;
		std::vector<std::vector<uint8_t>> bitmask;
		std::vector<int> row_offset;

		// Read input image.
		cv::Mat input_image = cv::imread(image_path);
		int height = input_image.rows;
		int width = input_image.cols;
		int channels = input_image.channels();
		
		// Read region file.
		std::ifstream region_file(region_path);
		std::vector<region> regions;
		std::string line;
		// Iterate over all regions.
		while (std::getline(region_file, line)) {
			std::stringstream ss(line);
			std::string val;
			region r;
			std::getline(ss, val, ',');
			r.x0 = std::stoi(val);
			std::getline(ss, val, ',');
			r.y0 = height-std::stoi(val);
			std::getline(ss, val, ',');
			r.x1 = std::stoi(val);
			std::getline(ss, val, ',');
			r.y1 = height-std::stoi(val);
			std::getline(ss, val, ',');
			r.stride = std::stoi(val)-1;
			std::getline(ss, val, ',');
			r.skip = std::stoi(val)-1;
			regions.push_back(r);
		}
		std::sort(regions.begin(), regions.end(), region_sort);
		// Full frame capture.
		if (regions.empty()) {
			region r = {0, (uint32_t)width, 0, (uint32_t)height, 0, 0};
			regions.push_back(r);
		}

		// Iterate over image.
		for (int row = 0; row < height; row++) {
			std::vector<uint8_t> rowmask;
			row_offset.push_back(write_pixel_touches);
			write_row_offset_touches++;
			for (int col = 0; col < width/2; col++) {
				uint8_t pixelmask = 0b11;
				for (int reg = 0; reg < regions.size(); reg++) {
					// Small optimization.
					if (col*2 < regions[reg].x0) {
						break;
					}
					// Check if the pixel is within the region.
					if (regions[reg].x0 <= col*2 && col*2 <= regions[reg].x1 && regions[reg].y0 <= row && row <= regions[reg].y1) {
						if (pixelmask != 0b00) {
							if ((col*2 & regions[reg].stride) > 0) {
								pixelmask |= 0b10;
							}
							else {
								pixelmask &= 0b01;
							}
							if (regions[reg].skip) {
								pixelmask |= 0b01;
							}
							else {
								pixelmask &= 0b10;
							}
							// Prioritize stride, if skip and stride are active.
							if (pixelmask == 0b11) {
								pixelmask = 0b10;
							}
						}
						// 0b00 indicates regional pixel, exit loop now.
						if (pixelmask == 0b00) {
							write_pixel_touches += 2;
							encoded_pixels.push_back(input_image.at<Pixel>(row, col*2));
							encoded_pixels.push_back(input_image.at<Pixel>(row, col*2+1));
							break;
						}
					}
				} // Region loop
				rowmask.push_back(pixelmask);
				write_bitmask_touches++;
			} // Col loop
			bitmask.push_back(rowmask);
		} // Row loop

		// Fill last row of encoded image.
		while ((write_pixel_touches % width) != 0) {
			encoded_pixels.push_back(zero);
			encoded_pixels.push_back(zero);
			write_pixel_touches += 2;
		}

		// Convert vector to image.
		cv::Mat encoded_image = cv::Mat(encoded_pixels, true).reshape(channels, write_pixel_touches / width);
		encoded_images.insert(encoded_images.begin(), encoded_image);
		bitmasks.insert(bitmasks.begin(), bitmask);
		row_offsets.insert(row_offsets.begin(), row_offset);
		if (bitmasks.size() > 4) {
			encoded_images.pop_back();
			bitmasks.pop_back();
			row_offsets.pop_back();
		}
		
		// Decode image.
		cv::Mat output_image(height, width, input_image.type());
		for (int row = 0; row < height; row++) {
			// Assume we cache row_offset for each of the frames at each row.
			read_row_offset_touches += row_offsets.size();
			// Assume we cache the bitmask for the entire row for each of the 4 frames at each row.
			read_bitmask_touches += bitmasks.size()*width/2;
			for (int col = 0; col < width/2; col++) {
				uint8_t pixelmask = bitmask[row][col];
				if (pixelmask == 0b01) {
					// Skipped pixel, check previous frames, most recent first, but skipping current.
					for (int fr = 1; fr < bitmasks.size(); fr++) {
						if (bitmasks[fr][row][col] == 0b00 || bitmasks[fr][row][col] == 0b10) {
							std::tuple<int, int> rc = get_px_index(width, bitmasks[fr][row], row_offsets[fr][row], col);
							int r = std::get<0>(rc);
							int c = std::get<1>(rc);
							output_image.at<Pixel>(row, col*2) = encoded_images[fr].at<Pixel>(r, c);
							output_image.at<Pixel>(row, col*2+1) = encoded_images[fr].at<Pixel>(r, c+1);
							read_pixel_touches += 2;
							// Only want to grab pixels once.
							break;
						}
					}
				}
				else if (pixelmask == 0b00 || pixelmask == 0b10) {
					// Regional or strided pixel.
					std::tuple<int, int> rc = get_px_index(width, bitmask[row], row_offset[row], col);
					int r = std::get<0>(rc);
					int c = std::get<1>(rc);
					output_image.at<Pixel>(row, col*2) = encoded_image.at<Pixel>(r, c);
					output_image.at<Pixel>(row, col*2+1) = encoded_image.at<Pixel>(r, c+1);
					read_pixel_touches += 2;
				}
				else if (pixelmask == 0b11) {
					output_image.at<Pixel>(row, col*2) = zero;
					output_image.at<Pixel>(row, col*2+1) = zero;
				}
			}
		}

		// Save images.
		std::string output_image_name = image_path.substr(input_folder_name.length()+1, image_path.length()-input_folder_name.length()-1);
		std::string output_stats_name = region_path.substr(region_folder_name.length()+1, region_path.length()-region_folder_name.length()-1);
		cv::imwrite(output_folder_name + "/" + output_image_name, output_image);
		cv::imwrite(output_folder_name + "/encoded/" + output_image_name, encoded_image);
		images++;
		std::cout << "======= " << (int)images/filecount*100 << "% done (" << images << "/" << filecount << ") =======" << std::endl;
		
		// Calculate statistics.
		uint32_t write_current_bits = write_pixel_touches*24 + write_bitmask_touches*2 + write_row_offset_touches*24;
		uint32_t read_current_bits = read_pixel_touches*24 + read_bitmask_touches*2 + read_row_offset_touches*24;
		std::cout.imbue(std::locale(""));
		std::cout << std::setprecision(2) << write_current_bits/MEGABYTE << " MB estimated written with " << write_pixel_touches << " pixel touches, " << write_bitmask_touches << " bitmask touches, and " << write_row_offset_touches << " row offset touches for this frame." << std::endl;
		std::cout << std::setprecision(2) << read_current_bits/MEGABYTE << " MB estimated read with " << read_pixel_touches << " pixel touches, " << read_bitmask_touches << " bitmask touches, and " << read_row_offset_touches << " row offset touches for this frame." << std::endl;
		
		// Write to csv.
		std::ofstream csvfile(output_folder_name + "/stats/" + output_stats_name);
		csvfile << write_current_bits/MEGABYTE;
		csvfile << write_pixel_touches;
		csvfile << write_bitmask_touches;
		csvfile << write_row_offset_touches;
		csvfile << std::endl;
		csvfile << read_current_bits/MEGABYTE;
		csvfile << read_pixel_touches;
		csvfile << read_bitmask_touches;
		csvfile << read_row_offset_touches;
		csvfile << std::endl;
		csvfile << (read_current_bits+write_current_bits)/MEGABYTE;
		csvfile << read_pixel_touches+write_pixel_touches;
		csvfile << read_bitmask_touches+write_bitmask_touches;
		csvfile << read_row_offset_touches+write_row_offset_touches;
		csvfile.close();
		
		// Total stat calculation.
		total_write_bits += write_current_bits;
		total_write_pixel_touches += write_pixel_touches;
		total_write_bitmask_touches += write_bitmask_touches;
		total_write_row_offset_touches += write_row_offset_touches;
		total_read_bits += read_current_bits;
		total_read_pixel_touches += read_pixel_touches;
		total_read_bitmask_touches += read_bitmask_touches;
		total_read_row_offset_touches += read_row_offset_touches;
		
	} // File loop
	
	// Write total statistics.
	std::ofstream csvfile(output_folder_name + "/stats/total.csv");
	csvfile << total_write_bits/MEGABYTE;
	csvfile << total_write_pixel_touches;
	csvfile << total_write_bitmask_touches;
	csvfile << total_write_row_offset_touches;
	csvfile << std::endl;
	csvfile << total_read_bits/MEGABYTE;
	csvfile << total_read_pixel_touches;
	csvfile << total_read_bitmask_touches;
	csvfile << total_read_row_offset_touches;
	csvfile << std::endl;
	csvfile << (total_read_bits+total_write_bits)/MEGABYTE;
	csvfile << total_read_pixel_touches+total_write_pixel_touches;
	csvfile << total_read_bitmask_touches+total_write_bitmask_touches;
	csvfile << total_read_row_offset_touches+total_write_row_offset_touches;
	csvfile.close();

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
	std::cout << "==================================" << std::endl;
	std::cout << std::setprecision(2) << total_write_bits/MEGABYTE << " MB estimated written with " << total_write_pixel_touches << " pixel touches, " << total_write_bitmask_touches << " bitmask touches, and " << total_write_row_offset_touches << " row offset touches in total." << std::endl << std::endl;
	std::cout << std::setprecision(2) << total_read_bits/MEGABYTE << " MB estimated read with " << total_read_pixel_touches << " pixel touches, " << total_read_bitmask_touches << " bitmask touches, and " << total_read_row_offset_touches << " row offset touches in total." << std::endl;
	std::cout << "Completed in " << duration.count() << " seconds! Check ./" << output_folder_name << " for decoded/encoded frames and statistics";
	return 0;
} // Main function

// Get pixel indices from encoded frame.
std::tuple <int, int> get_px_index(uint32_t width, std::vector<uint8_t> rowmask, uint32_t row_offset, uint col) {
	int count = 0;
	for (int i = 0; i < col; i++) {
		if (rowmask[i] == 0b00) {
			// Count regional pixels on the way to the specified index.
			count += 2;
		}
	}

	int regional_px_cnt = row_offset + count;
	int r = regional_px_cnt / width;
	int c = regional_px_cnt % width;

	return std::make_tuple(r, c);
}

void print_usage() {
	std::cout << "Usage: rhythm-folder OPTIONS" << std::endl
			  << "Simulates rhythm on the given input image and saves to given output image." << std::endl
			  << "-h, --help\t\tPrints this dialog and exits." << std::endl
			  << "-i, --input\t\tSpecifies the input folder, all images will be read in this folder." << std::endl
			  << "-o, --output\tSpecifies the output folder. The folder will be created if it doesn't exist." << std::endl
			  << "-r, --region\t\tSpecifies the folder with regions for each frame in a csv.";
}
