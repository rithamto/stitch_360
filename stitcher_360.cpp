#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/stitching/detail/warpers.hpp>
#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/detail/util.hpp>
#include <vector>
#include <string>
#include <android/log.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define TAG "OpenCVStitcher"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

extern "C" {

/**
 * Advanced Metadata-Driven 360 Stitcheing.
 * Uses Yaw/Pitch from IMU to force-place images into a spherical panorama.
 */
__attribute__((visibility("default"))) __attribute__((used))
const char* stitch_360_native(
    const char** image_paths,
    double* yaws,
    double* pitches,
    double* rolls,
    int num_images,
    const char* output_path,
    double focal_length,
    bool try_use_gpu
) {
    try {
        if (num_images < 1) return "Error: No images provided";

        LOGD("Starting Advanced Metadata Stitching for %d images (f=%f)", num_images, focal_length);

        // 1. Prepare vectors
        std::vector<cv::Mat> valid_images_warped;
        std::vector<cv::Mat> valid_masks_warped;
        std::vector<cv::Point> valid_corners;

        // Use CPU warper for stability on mobile
        cv::Ptr<cv::detail::SphericalWarper> warper = cv::makePtr<cv::detail::SphericalWarper>((float)focal_length);

        // Generic Distortion Coefficients (Adjustable/Future: pass via FFI)
        // [k1, k2, p1, p2, k3]
        cv::Mat distCoeffs = (cv::Mat_<float>(5,1) << -0.1f, 0.02f, 0, 0, 0);

        for (int i = 0; i < num_images; ++i) {
            cv::Mat img = cv::imread(image_paths[i]);
            if (img.empty()) {
                LOGE("Skipping empty image: %s", image_paths[i]);
                continue;
            }

            // A. Apply Lens Distortion Correction (Undistort)
            cv::Mat cameraMatrix = (cv::Mat_<float>(3,3) << 
                focal_length, 0, img.cols/2.0f, 
                0, focal_length, img.rows/2.0f, 
                0, 0, 1);
            
            cv::Mat undistorted;
            cv::undistort(img, undistorted, cameraMatrix, distCoeffs);
            img = undistorted;

            // B. Downsample for memory efficiency (Crucial for mobile stability)
            if (img.cols > 800) {
                double scale = 800.0 / img.cols;
                cv::resize(img, img, cv::Size(), scale, scale);
                // Correct camera matrix for scaled image
                cameraMatrix *= scale;
                cameraMatrix.at<float>(2,2) = 1.0f;
            }

            // C. Create 3D Rotation Matrix from Yaw/Pitch/Roll
            double y = yaws[i] * M_PI / 180.0;
            double p = pitches[i] * M_PI / 180.0;
            double r = rolls[i] * M_PI / 180.0;

            // Rotation around Y (Yaw)
            cv::Mat Ry = (cv::Mat_<float>(3,3) << 
                cos(y), 0, sin(y),
                0, 1, 0,
                -sin(y), 0, cos(y));
            
            // Rotation around X (Pitch)
            cv::Mat Rp = (cv::Mat_<float>(3,3) << 
                1, 0, 0,
                0, cos(p), -sin(p),
                0, sin(p), cos(p));
            
            // Rotation around Z (Roll)
            cv::Mat Rr = (cv::Mat_<float>(3,3) << 
                cos(r), -sin(r), 0,
                sin(r), cos(r), 0,
                0, 0, 1);
            
            // Final Combined Rotation
            cv::Mat Rotation = Ry * Rp * Rr;

            // D. Warping
            cv::Mat warped_img;
            cv::Point corner = warper->warp(img, cameraMatrix, Rotation, cv::INTER_LINEAR, cv::BORDER_REFLECT, warped_img);
            
            cv::Mat warped_mask;
            warper->warp(cv::Mat(img.size(), CV_8U, cv::Scalar(255)), cameraMatrix, Rotation, cv::INTER_NEAREST, cv::BORDER_CONSTANT, warped_mask);

            valid_images_warped.push_back(warped_img);
            valid_masks_warped.push_back(warped_mask);
            valid_corners.push_back(corner);
            
            // Release original image to free memory immediately
            img.release();
            LOGD("Warped image %d/%d", i+1, num_images);
        }

        if (valid_images_warped.empty()) return "Error: No valid images could be loaded/warped";

        // 2. Exposure Compensation (Sync brightness across frames)
        LOGD("Applying Exposure Compensation...");
        cv::Ptr<cv::detail::ExposureCompensator> compensator = cv::detail::ExposureCompensator::createDefault(cv::detail::ExposureCompensator::GAIN);
        
        std::vector<cv::UMat> images_f_umat(valid_images_warped.size());
        std::vector<cv::UMat> masks_warped_umat(valid_images_warped.size());
        for(size_t i=0; i<valid_images_warped.size(); ++i) {
            valid_images_warped[i].convertTo(images_f_umat[i], CV_32F);
            valid_masks_warped[i].copyTo(masks_warped_umat[i]);
        }
        
        compensator->feed(valid_corners, images_f_umat, masks_warped_umat);
        
        for(size_t i=0; i<valid_images_warped.size(); ++i) {
            compensator->apply((int)i, valid_corners[i], images_f_umat[i], masks_warped_umat[i]);
            // Convert back to 16S for blender input compatibility
            images_f_umat[i].convertTo(valid_images_warped[i], CV_16S);
        }

        // 3. Seam Finding (Lightweight DpSeamFinder for mobile stability)
        LOGD("Finding optimal seams (DP)...");
        cv::Ptr<cv::detail::SeamFinder> seam_finder = cv::makePtr<cv::detail::DpSeamFinder>(cv::detail::DpSeamFinder::COLOR);
        
        seam_finder->find(images_f_umat, valid_corners, masks_warped_umat);
        
        // Release intermediate float images immediately after seam finding
        for(auto& m : images_f_umat) m.release();
        images_f_umat.clear();

        // Sync back masks if changed by seam finder
        for(size_t i=0; i<valid_masks_warped.size(); ++i) {
            masks_warped_umat[i].copyTo(valid_masks_warped[i]);
            masks_warped_umat[i].release();
        }
        masks_warped_umat.clear();

        // 4. Blending (Multi-band frequency blending)
        LOGD("Blending panorama...");
        
        // Calculate optimal band count for smoothness
        float blend_strength = 5.0f;
        int num_bands = (int)ceil(log2(focal_length / blend_strength));
        cv::Ptr<cv::detail::MultiBandBlender> mb = cv::makePtr<cv::detail::MultiBandBlender>(false, num_bands);
        
        std::vector<cv::Size> sizes(valid_images_warped.size());
        for(size_t i=0; i<valid_images_warped.size(); ++i) {
             sizes[i] = valid_images_warped[i].size();
        }

        // Force ROI to handle 360 wrap-around equirectangular mapping
        // Width = 2 * PI * focal, Height = PI * focal for a full sphere
        cv::Rect dst_roi = cv::detail::resultRoi(valid_corners, sizes);
        
        // Ensure standard equirectangular 2:1 mapping if focal_length allows
        int full_width = (int)(2 * M_PI * focal_length);
        if (dst_roi.width < full_width * 0.8) {
             LOGD("Warning: Captured width (%d) is significantly less than full 360 (%d). Check focal_length.", dst_roi.width, full_width);
        }
        
        mb->prepare(dst_roi);

        for (size_t i = 0; i < valid_images_warped.size(); ++i) {
            mb->feed(valid_images_warped[i], valid_masks_warped[i], valid_corners[i]);
        }

        cv::Mat result, result_mask;
        mb->blend(result, result_mask);
        result.convertTo(result, CV_8U);

        if (!cv::imwrite(output_path, result)) {
            return "Error: Could not save final stitched 360 image";
        }

        LOGD("Metadata-Driven 360 Stitching Complete!");
        return nullptr;

    } catch (const std::exception& e) {
        LOGE("Native Exception: %s", e.what());
        return strdup(e.what());
    }
}

/**
 * Basic Image Denoising.
 */
__attribute__((visibility("default"))) __attribute__((used))
const char* denoise_image_native(const char* input_path, const char* output_path, float strength) {
    try {
        cv::Mat src = cv::imread(input_path);
        if (src.empty()) return "Error: Could not read image";
        cv::Mat dst;
        cv::fastNlMeansDenoisingColored(src, dst, strength, strength, 7, 21);
        cv::imwrite(output_path, dst);
        return nullptr;
    } catch (const std::exception& e) {
        return strdup(e.what());
    }
}

}
