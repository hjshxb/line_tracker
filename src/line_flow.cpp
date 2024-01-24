#include <opencv2/optflow.hpp>
#include <iostream>

#include "line_flow.h"


void BuildOpticalFlowPyramid(cv::InputArray _img,
                            cv::OutputArrayOfArrays pyramid_img,
                            const cv::Size& winsize,
                            int max_level,
                            int border_mode) {
    cv::Mat img = _img.getMat();
    CV_Assert(img.depth() == CV_8U && winsize.height > 2 && winsize.width > 2);
    pyramid_img.create(1, max_level + 1, 0, -1, true);

    cv::Mat prev_level_img = pyramid_img.getMatRef(0);
    cv::Size sz = img.size();

    for (int level = 0; level <= max_level; ++level) {
        cv::Mat& temp = pyramid_img.getMatRef(level);

        if (!temp.empty()) {
            temp.adjustROI(winsize.height, winsize.height, winsize.width, winsize.width);
        }
        if (temp.type() != img.type() || temp.cols != winsize.width * 2 + sz.width ||
                temp.rows != winsize.height * 2 + sz.height) {
            temp.create(sz.height + winsize.height * 2, sz.width + winsize.width * 2, img.type());
        }
        cv::Mat this_level_img = temp(cv::Rect(winsize.width, winsize.height, sz.width, sz.height));
        if (level != 0) {
            cv::pyrDown(prev_level_img, this_level_img, sz);
            cv::copyMakeBorder(this_level_img, temp, winsize.height, winsize.height,
                winsize.width, winsize.width, border_mode | cv::BORDER_ISOLATED);
        } else {
            cv::copyMakeBorder(img, temp, winsize.height, winsize.height,
                winsize.width, winsize.width, border_mode | cv::BORDER_ISOLATED);
        }
        temp.adjustROI(-winsize.height, -winsize.height, -winsize.width, -winsize.width);

        sz = cv::Size((sz.width + 1) / 2, (sz.height + 1) / 2);
        if (sz.width <= winsize.width || sz.height <= winsize.height) {
            pyramid_img.create(1, level + 1, 0, -1, true);
            return;
        }
        prev_level_img = this_level_img;
    }
}


void LineLKTracker::operator() (
            const cv::Mat& prev_img, const cv::Mat& next_img, 
            const std::vector<cv::Vec4f>& prev_lines, 
            std::vector<cv::Vec4f>& next_lines, 
            std::vector<uchar>& status) const {
    CV_Assert(max_level_ > 0 && winsize_.width > 2 && winsize_.height > 2);
    if (prev_lines.empty()) {
        return;
    }
    status.resize(prev_lines.size(), false);
    
    std::vector<cv::Mat> prev_pyramid;
    std::vector<cv::Mat> next_pyramid;
    int pyramid_level = cv::buildOpticalFlowPyramid(prev_img, prev_pyramid, winsize_, max_level_, false);
    pyramid_level = cv::buildOpticalFlowPyramid(next_img, next_pyramid, winsize_, pyramid_level, false);

    std::vector<std::vector<cv::Point2f>> prev_lines_kps(
        prev_lines.size(), std::vector<cv::Point2f>(n_track_));
    std::vector<std::vector<cv::Point2f>> next_lines_kps(
        prev_lines.size(), std::vector<cv::Point2f>());

    for (size_t i = 0; i < prev_lines.size(); ++i) {
        SamplePointsOnLine(prev_lines[i], prev_lines_kps[i]);
        next_lines_kps[i] = prev_lines_kps[i];
    }
    std::vector<float> g_param(prev_lines.size() * 3, 0.f);

    for (int level = pyramid_level; level >= 0; --level) {
        status.resize(prev_lines.size(), false);
        bool use_check = (level == 0) ? check_ : false;
        cv::parallel_for_(
            cv::Range(0, prev_lines.size()),
            LineLKSingleLevel(
                prev_pyramid[level], next_pyramid[level],
                prev_lines_kps, next_lines_kps,
                status, g_param, level, use_check));
    }

    next_lines = prev_lines;
    for (size_t i = 0; i < prev_lines.size(); ++i) {
        if (status[i]) {
            next_lines[i] = {next_lines_kps[i][0].x,
                             next_lines_kps[i][0].y,
                             next_lines_kps[i].back().x,
                             next_lines_kps[i].back().y};
        }
    }

}

void LineLKTracker::SamplePointsOnLine(const cv::Vec4f& line,
                        std::vector<cv::Point2f>& points) const {
    cv::Point2f start(line[0], line[1]);
    cv::Point2f end(line[2], line[3]);
    if (n_track_ <= 2) {
        points = {start, end};
        return;
    }
    points.resize(n_track_);
    points[0] = start;

    cv::Point2f delta = (end - start) / (n_track_ - 1);
    for (int i = 1; i < n_track_ - 1; i++) {
        points[i] = (start + i * delta);
    }
    points.back() = end;
}

/**class LineLKSingleLevel **/

void LineLKSingleLevel::operator()(const cv::Range& range) const {
    for (int id = range.start; id < range.end; ++id) {
        auto g_param = params_.begin() + id * 3;
        status_[id] = OneLineLKTracker(prev_lines_[id], g_param, next_lines_[id]);
        (*g_param) *= 2;
        (*(g_param + 1)) *= 2;
    }
}

bool LineLKSingleLevel::OneLineLKTracker(
            const std::vector<cv::Point2f>& prev_line,
            const std::vector<float>::iterator g_param,
            std::vector<cv::Point2f>& next_line) const {
    float prev_cost = 0;
    auto g1 = g_param, g2 = g_param + 1, g3 = g_param + 2;
    // Gauss-Newton iterations
    Eigen::Matrix3f H;
    Eigen::Vector3f b;

    for (int iter = 0; iter < criteria_.maxCount; ++iter) {
        float cost = 0;
        H.setZero();
        b.setZero();

        for (size_t i = 0; i < prev_line.size(); ++i) {
            cv::Point2f prev_points = prev_line[i] * (float)(1. / (1 << level_));
            cv::Point2f delta = (prev_line[i] - prev_line[0]) * (float)(1. / (1 << level_));
            // u' = u +　g1 - g3 * delta_v
            // v' = v +　g2 + g3 * delta_u
            float dx = (*g1) - (*g3) * delta.y;
            float dy = (*g2) + (*g3) * delta.x;

            prev_points -= half_win_;
            cost += CalOnePoint(prev_points, delta, dx, dy, iter, H, b);

        }
        if (iter > 0 && cost > prev_cost) break;
        prev_cost = cost;

        Eigen::Vector3f update = H.ldlt().solve(b);
        if (std::isnan(update(0) || std::isnan(update(2)))) {
            return false;
        }

        (*g1) += update(0);
        (*g2) += update(1);
        (*g3) += update(2);

        if (update.norm() < criteria_.EPS) {
            break;
        }

    }

    // Check angle
    // if (check_ && std::fabs(*g3) > 0.10) {
    //     std::cout << (*g3) << std::endl;
    //     return false;
    // }


    // update points
    int outlier_count = 0;
    for (size_t i = 0; i < next_line.size(); ++i) {
        const cv::Point2f& prev_kp = prev_line[i];
        cv::Point2f delta = prev_kp - prev_line[0];
        cv::Point2f& next_kp = next_line[i];
        float inv_scale = (1 << level_);
        next_kp.x = prev_kp.x + inv_scale * (*g1) - (*g3) * delta.y;
        next_kp.y = prev_kp.y + inv_scale * (*g2) + (*g3) * delta.x;

        // Check zncc
        if (level_ == 0 && check_) {
            float score = ZNCC(prev_kp, next_kp); 
            if (score < 0.80) {
                outlier_count++;
            }
            if (outlier_count > cvRound(0.2 * next_line.size())) {
                return false;
            }
        }
    }

    return true;
}

float LineLKSingleLevel::CalOnePoint(const cv::Point2f& point,
                                     const cv::Point2f& delta,
                                     float dx,
                                     float dy,
                                     int iter,
                                     Eigen::Matrix3f& H,
                                     Eigen::Vector3f& b) const {
    float sum_error = 0;
    Eigen::Vector3f J;
    for (int y = 0; y < winsize_.height; ++y) {
        for (int x = 0; x < winsize_.width; ++x) {
            float prev_pt_x = point.x + x;
            float prev_pt_y = point.y + y;
            float next_pt_x = prev_pt_x + dx;
            float next_pt_y = prev_pt_y + dy;

            float error =
                GetImageValue(prev_img_, prev_pt_x, prev_pt_y) -
                GetImageValue(next_img_, next_pt_x, next_pt_y);

            float Ix =
                0.5 * (GetImageValue(next_img_, next_pt_x + 1, next_pt_y) -
                        GetImageValue(next_img_, next_pt_x - 1, next_pt_y));
            float Iy =
                0.5 * (GetImageValue(next_img_, next_pt_x, next_pt_y + 1) -
                        GetImageValue(next_img_, next_pt_x, next_pt_y - 1));
            // jaco
            J << -Ix, -Iy, Ix * delta.y - Iy * delta.x;
            b += -(error * J);
            H += J * J.transpose();
            sum_error += error;
        }
    }
    return sum_error;
}

inline float LineLKSingleLevel::GetImageValue(const cv::Mat& img,
                                              float x,
                                              float y) const {
    // The img has been padding
    // x = (x < 0) ? 0 : ((img.cols - 1) < x) ? (img.cols - 1) : x;
    // y = (y < 0) ? 0 : ((img.rows - 1) < y) ? (img.rows - 1) : y;

    int ix = cvFloor(x);
    int iy = cvFloor(y);
    float a = x - ix;
    float b = y - iy;

    float w11 = (1.f - a) * (1.f - b);
    float w21 = a * (1.f - b);
    float w12 = (1.f - a) * b;
    float w22 = a * b;
    
    int cn = img.channels();
    int step = img.step1();

    const uchar* data = img.ptr() + iy * step + ix * cn;
    return (w11 * data[0] + w21 * data[cn] 
            + w12 * data[step] + w22 * data[step + cn]);
}

float LineLKSingleLevel::ZNCC(const cv::Point2f& prev_point,
               const cv::Point2f& next_point) const {
    
    std::vector<float> vec_prev_pixels, vec_next_pixels;
    double mean_prev = 0;
    double mean_next = 0;

    for (int y = -half_win_.y; y <= half_win_.y; ++y) {
        for (int x = -half_win_.x; x <= half_win_.x; ++x) {
            float prev_pixel = GetImageValue(prev_img_, prev_point.x + x, prev_point.y + y) / 255.0;
            float next_pixel = GetImageValue(next_img_, next_point.x + x, next_point.y + y) / 255.0;
            mean_prev += prev_pixel;
            mean_next += next_pixel;
            vec_prev_pixels.push_back(prev_pixel);
            vec_next_pixels.push_back(next_pixel);
        }
    }
    mean_prev /= (winsize_.height * winsize_.width);
    mean_next /= (winsize_.height * winsize_.width);

    double numerator = 0, demoniator1 = 0, demoniator2 = 0;
    for (size_t i = 0; i < vec_prev_pixels.size(); ++i) {
        numerator += (vec_prev_pixels[i] - mean_prev) * (vec_next_pixels[i] - mean_next);
        demoniator1 += (vec_prev_pixels[i] - mean_prev) * (vec_prev_pixels[i] - mean_prev);
        demoniator2 += (vec_next_pixels[i] - mean_next) * (vec_next_pixels[i] - mean_next);
    }
    return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);
    
}