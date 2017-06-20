#include <iostream>
#include <string>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/detail/camera.hpp>
#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/stitching/detail/util.hpp>
#include <opencv2/stitching/detail/warpers.hpp>
#include <opencv2/stitching/warpers.hpp>

using namespace std;
using namespace cv;
using namespace cv::detail;

int main(int argc, char* argv[]) {
	double scale = 1;

	float blend_strength = 5;

	bool try_gpu = true;

	float match_threshold = 0.3f;
	float adjuster_threshold = 1.f;

	int features_type = 1; 		/* 	0:surf
									1:orb 
								*/
	int adjustment_type = 1; 	/* 	0:reproj
									1:ray 
								*/
	int wave_corr_type = 1; 	/* 	0:none
									1:horizontal
									2:vertical 
								*/
	int warp_type = 2; 			/* 	0:rectilinear
									1:cylindrical
									2:spherical
									3:stereographic
									4:panini
								*/
	int exposure_comp_type = 1;	/*  0:none
									1:gain
									2:gainblocks
								*/
	int seam_type = 2; 			/* 	0:none
									1:voronoi
									2:gc_color
									3:gc_colorgrad
									4:dp_color
									5:dp_colorgrad
								*/
	int blend_type = 2; 		/* 	0:none
									1:feather
									2:multiband
								*/

	String output_name = "panorama_result_perso.jpg";
	double start_time = getTickCount();

	// 1 - Get input image names

	vector<String> image_names;

	for (size_t i = 1; i < argc; i++)
		image_names.push_back(argv[i]);

	int image_count = static_cast<int>(image_names.size());

	if (image_count < 2) {
		cout << "Need more images" << endl;
		return -1;
	}

	// 2 - Resize

	cout << "Finding features..." << endl;
	double t = getTickCount();

	Ptr<FeaturesFinder> finder; // http://docs.opencv.org/2.4.8/modules/stitching/doc/matching.html#detail-featuresfinder-collectgarbage

	switch (features_type) {
		case 0 :
			finder = makePtr<SurfFeaturesFinder>();
			break;
		case 1 :
			finder = makePtr<OrbFeaturesFinder>();
			break;
		default :
			return -1;
	}

	Mat image_orig, image_resized;
	vector<ImageFeatures> features(image_count); // http://docs.opencv.org/3.1.0/d4/db5/structcv_1_1detail_1_1ImageFeatures.html
	vector<Mat> images(image_count);
	vector<Size> image_sizes(image_count);

	for (size_t i = 0; i < image_count; i++) {
		image_orig = imread(image_names[i]);
		image_sizes[i] = image_orig.size();

		if (image_orig.empty()) {
			cout << "Can't open image " << image_names[i] << endl;
			return -1;
		}

		resize(image_orig, image_resized, Size(), scale, scale);
		images[i] = image_resized.clone();
		(*finder)(image_resized, features[i]);
		features[i].img_idx = i;
	}

	finder->collectGarbage();
	image_orig.release();
	image_resized.release();

	cout << "Finding features, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec" << endl;

	// 3 - Match features

	cout << "Pairwise matching..." << endl;
	t = getTickCount();

	vector<MatchesInfo> matches; // http://docs.opencv.org/3.1.0/d2/d9a/structcv_1_1detail_1_1MatchesInfo.html
	BestOf2NearestMatcher matcher(try_gpu, match_threshold); // http://docs.opencv.org/2.4.8/modules/stitching/doc/matching.html
	matcher(features, matches);
	matcher.collectGarbage();

	cout << "Pairwise matching, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec" << endl;

	// 4 - Biggest component and find homographies

	cout << "Finding biggest component and homographies..." << endl;
	t = getTickCount();
	
	vector<int> indices = leaveBiggestComponent(features, matches, match_threshold); // http://docs.opencv.org/trunk/d7/d74/group__stitching__rotation.html#ga855d2fccbcfc3b3477b34d415be5e786

	vector<Mat> images_temp;
	vector<String> image_names_temp;
	vector<Size> image_sizes_temp;
	vector<ImageFeatures> features_temp;
	vector<MatchesInfo> matches_temp;

	for (size_t i = 0; i < indices.size(); i++) {
		images_temp.push_back(images[indices[i]]);
		image_names_temp.push_back(image_names[indices[i]]);
		image_sizes_temp.push_back(image_sizes[indices[i]]);/*
		features_temp.push_back(features[indices[i]]);
		matches_temp.push_back(matches[indices[i]]);*/
	}
	/*
	images.clear();
	image_names.clear();
	image_sizes.clear();
	features.clear();
	matches.clear();
	*/

	images = images_temp;
	image_names = image_names_temp;
	image_sizes = image_sizes_temp;/*
	features = features_temp;
	matches = matches_temp;*/
	image_count = images.size();

	HomographyBasedEstimator estimator; // http://docs.opencv.org/3.1.0/db/d3e/classcv_1_1detail_1_1HomographyBasedEstimator.html
	vector<CameraParams> cameras; // http://docs.opencv.org/3.1.0/d4/d0a/structcv_1_1detail_1_1CameraParams.html

	if (!estimator(features, matches, cameras))
		return -1;

	for (size_t i = 0; i < image_count; i++) { // WHY
		Mat rotation;
		cameras[i].R.convertTo(rotation, CV_32F);
		cameras[i].R = rotation;
	}

	cout << "Finding biggest component and homographies, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec" << endl;

	// 5 - Bundle adjustment

	cout << "Bundle adjustment..." << endl;
	t = getTickCount();

	Ptr<BundleAdjusterBase> bundle_adjuster; // http://docs.opencv.org/trunk/d5/d56/classcv_1_1detail_1_1BundleAdjusterBase.html

	switch (adjustment_type) {
		case 0 :
			bundle_adjuster = makePtr<BundleAdjusterReproj>();
			break;
		case 1 :
			bundle_adjuster = makePtr<BundleAdjusterRay>();
			break;
		default :
			return -1;
	}

	bundle_adjuster->setConfThresh(adjuster_threshold);
for (int i = 0; i < cameras.size(); i++)
	cout << cameras[i].R << endl;

	if (!(*bundle_adjuster)(features, matches, cameras)) { // study the adjuster, find if we can make it better/faster
		return -1;
	}

	vector<double> focal_lengths;

	for (size_t i = 0; i < image_count; i++)
		focal_lengths.push_back(cameras[i].focal);

	sort(focal_lengths.begin(), focal_lengths.end());
	float global_focal;

	if (image_count % 2)
		global_focal = static_cast<float>(focal_lengths[image_count / 2]);
	else
		global_focal = static_cast<float>(focal_lengths[image_count / 2 - 1] + focal_lengths[image_count / 2 + 1]) * .5f;

	cout << "Bundle adjustment, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec" << endl;

	// 6 - Wave correlation

	cout << "Wave correlation..." << endl;
	t = getTickCount();

	if (wave_corr_type) {
		WaveCorrectKind wave_corr_mode;

		switch (wave_corr_type) {
			case 1 :
				wave_corr_mode = WAVE_CORRECT_HORIZ;
				break;
			case 2 :
				wave_corr_mode = WAVE_CORRECT_VERT;
				break;
			default :
				return -1;
		}

		vector<Mat> rotations;

		for (size_t i = 0; i < image_count; i++)
			rotations.push_back(cameras[i].R.clone());

		waveCorrect(rotations, wave_corr_mode);

		for (size_t i = 0; i < image_count; i++)
			cameras[i].R = rotations[i];
	}

	cout << "Wave correlation, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec" << endl;

	// 7 - Warp images

	cout << "Warping images..." << endl;
	t = getTickCount();

	vector<Point> corners(image_count);
	vector<UMat> masks(image_count);
	vector<UMat> warped_masks(image_count);
	vector<UMat> warped_images(image_count);
	vector<Size> warped_sizes(image_count);

	for (size_t i = 0; i < image_count; i++) {
		masks[i].create(image_sizes[i], CV_8U);
		masks[i].setTo(Scalar::all(255));
	}

	Ptr<WarperCreator> warper_creator;

	switch (warp_type) {
		case 0 :
			warper_creator = makePtr<cv::CompressedRectilinearWarper>(2.0f, 1.0f);
			break;
		case 1 :
			warper_creator = makePtr<cv::CylindricalWarper>();
			break;
		case 2 :
			warper_creator = makePtr<cv::SphericalWarper>();
			break;
		case 3 :
			warper_creator = makePtr<cv::StereographicWarper>();
			break;
		case 4 :
			warper_creator = makePtr<cv::PaniniWarper>(2.0f, 1.0f);
			break;
		default :
			return -1;
	}

	Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(global_focal * scale));
	vector<UMat> warped_images_aux(image_count);

	for (size_t j = 0; j < image_count; j++) {
		size_t i = indices[j];
		Mat_<float> K;
		cameras[i].K().convertTo(K, CV_32F);
		float temp = (float) scale;
		K(0, 0) *= temp;
		K(0, 2) *= temp;
		K(1, 1) *= temp;
		K(1, 2) *= temp;
		corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, warped_images[i]); // explain constants
		warped_sizes[i] = warped_images[i].size();
		warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, warped_masks[i]);
		warped_images[i].convertTo(warped_images_aux[i], CV_32F); // conversion for the seam finder
	}

	images.clear();
	masks.clear();

	cout << "Warping images, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec" << endl;

	// 8 - Fix exposure

	cout << "Fixing exposure..." << endl;
	t = getTickCount();

	int exposure_comp_mode;

	switch (exposure_comp_type) {
		case 0 :
			exposure_comp_mode = ExposureCompensator::NO;
			break;
		case 1 :
			exposure_comp_mode = ExposureCompensator::GAIN;
			break;
		case 2 :
			exposure_comp_mode = ExposureCompensator::GAIN_BLOCKS;
			break;
		default :
			return -1;
	}

	Ptr<ExposureCompensator> exposure_compensator = ExposureCompensator::createDefault(exposure_comp_mode);
	
	exposure_compensator->feed(corners, warped_images, warped_masks);

	warped_images.clear();

	cout << "Fixing exposure, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec" << endl;

	// 9 - Seam finding

	cout << "Seam finding..." << endl;
	t = getTickCount();

	Ptr<SeamFinder> seam_finder;

	switch (seam_type) {
		case 0 :
			seam_finder = makePtr<NoSeamFinder>();
			break;
		case 1 :
			seam_finder = makePtr<VoronoiSeamFinder>();
			break;
		case 2 :
			seam_finder = makePtr<GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);
			break;
		case 3 :
			seam_finder = makePtr<GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
			break;
		case 4 :
			seam_finder = makePtr<DpSeamFinder>(DpSeamFinder::COLOR);
			break;
		case 5 :
			seam_finder = makePtr<DpSeamFinder>(DpSeamFinder::COLOR_GRAD);
			break;
		default :
			return -1;
	}

	seam_finder->find(warped_images_aux, corners, warped_masks);

	warped_images_aux.clear();
	
	cout << "Seam finding, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec" << endl;

	// 10 - Blending

	cout << "Blending..." << endl;
	t = getTickCount();

	Ptr<Blender> blender;
	Size roi = resultRoi(corners, warped_sizes).size();
	float blend_width = sqrt(static_cast<float>(roi.area())) * blend_strength / 100.f;

	if (blend_width < 1)
		blend_width = 1;

	switch (blend_type) {
		case 0 :
			blender = Blender::createDefault(Blender::NO, try_gpu);
			break;
		case 1 : {
			blender = Blender::createDefault(Blender::FEATHER, try_gpu);
			FeatherBlender* fb = dynamic_cast<FeatherBlender*>(blender.get());
			fb->setSharpness(1.f / blend_width);
			break;
		}
		case 2 : {
			blender = Blender::createDefault(Blender::MULTI_BAND, try_gpu);
			MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get()); // why dynamic cast
			mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.0)) - 1));
			break;
		}
		default :
			return -1;
	}

	blender->prepare(corners, warped_sizes);
	
	cout << "Blending, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec" << endl;

	// 11 - Compositing

	cout << "Compositing..." << endl;
	t = getTickCount();

	Mat image_warp, image_warp_simple, mask, mask_dil, mask_cut, mask_warp;

	for (size_t i = 0; i < image_count; i++) {
		image_orig = imread(image_names[indices[i]]);
		resize(image_orig, image_resized, Size(), scale, scale);
		Mat K;
		cameras[i].K().convertTo(K, CV_32F);

		warper->warp(image_resized, K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, image_warp);

		mask.create(image_sizes[i], CV_8U);
		mask.setTo(Scalar::all(255));
		warper->warp(mask, K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, mask_warp);

		exposure_compensator->apply(i, corners[i], image_warp, mask_warp);
		image_warp.convertTo(image_warp_simple, CV_16S);
		dilate(warped_masks[i], mask_dil, Mat());
		resize(mask_dil, mask_cut, mask_warp.size());
		mask_warp = mask_warp & mask_cut;
	
		blender->feed(image_warp_simple, mask_warp, corners[i]);
	}

	Mat output, mask_output;
	blender->blend(output, mask_output);
	imwrite(output_name, output);

	cout << "Compositing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec" << endl;

	image_orig.release();
	image_resized.release();
	cout << "Finished, total time: " << ((getTickCount() - start_time) / getTickFrequency()) << " sec" << endl;
	
	return 0;
}