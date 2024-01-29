#pragma once

#include <opencv2/imgproc.hpp>
#include <eigen3/Eigen/Dense>

void BuildOpticalFlowPyramid(cv::InputArray _img,
                            cv::OutputArrayOfArrays pyramid_img,
                            const cv::Size& winsize,
                            int max_level,
                            int border_mode = cv::BORDER_REFLECT_101);

/**
 * @brief  Class used for calculating a line optical flow. 
 * More detail in https://ieeexplore.ieee.org/abstract/document/9998999
 * @param max_level Maximum pyramid level; 
 * if set to 0, pyramids are not used (single level), if set to 1, two levels are used, and so on; 
 * @param n_track Number of sampling points of the line (including two endpoints)
 * @param check If set to True, using ZNCC on image (pyramid level is 0).
*/
class LineLKTracker {

public:
    LineLKTracker(const cv::Size& winsize = cv::Size(21, 21),
                  int max_level = 5,
                  int n_track = 3,
                  bool check = true,
                  const cv::Size& zncc_winsize = cv::Size(7, 7),
                  const cv::TermCriteria& criteria = cv::TermCriteria(
                    cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                    5,
                    0.01))
     : winsize_(winsize), max_level_(max_level), n_track_(n_track),
       check_(check), zncc_winsize_(zncc_winsize) ,criteria_(criteria) {}

 void operator()(const cv::Mat& prev_img,
                 const cv::Mat& next_img,
                 const std::vector<cv::Vec4f>& prev_lines,
                 std::vector<cv::Vec4f>& next_lines,
                 std::vector<uchar>& status) const;

 void SamplePointsOnLine(const cv::Vec4f& line,
                         std::vector<cv::Point2f>& points) const;

private:
    cv::Size winsize_;
    int max_level_;
    int n_track_;
    bool check_;
    cv::Size zncc_winsize_;
    cv::TermCriteria criteria_;
};


class LineLKSingleLevel : public cv::ParallelLoopBody {
public:
    LineLKSingleLevel(const cv::Mat& prev_img,
                      const cv::Mat& next_img,
                      const std::vector<std::vector<cv::Point2f>>& prev_lines,
                      std::vector<std::vector<cv::Point2f>>& next_lines,
                      std::vector<uchar>& status,
                      std::vector<float>& params,
                      int level,
                      bool check = true,
                      const cv::Size& winsize = cv::Size(21, 21),
                      const cv::Size& zncc_winsize = cv::Size(5, 5),
                      const cv::TermCriteria& criteria = cv::TermCriteria(
                       cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                       5,
                       0.01))
    :   prev_img_(prev_img),
        next_img_(next_img),
        prev_lines_(prev_lines),
        next_lines_(next_lines),
        status_(status),
        params_(params),
        level_(level),
        check_(check),
        winsize_(winsize),
        zncc_winsize_(zncc_winsize),
        criteria_(criteria) {
        half_win_ = cv::Point2f((winsize_.width - 1) * 0.5f, (winsize_.height - 1) * 0.5f);
        zncc_half_win_ = cv::Point2f((zncc_winsize_.width - 1) * 0.5f, (zncc_winsize_.height - 1) * 0.5f);
        if (params_.empty()) {
            params_.resize(prev_lines_.size() * 3, 0.f);
        }
    }

    void operator() (const cv::Range& range) const CV_OVERRIDE;

    bool OneLineLKTracker(const std::vector<cv::Point2f>& prev_line,
                          const std::vector<float>::iterator g_param,
                          std::vector<cv::Point2f>& next_line) const;


    template <typename T>
    const T& Clamp(const T& val, const T& low, const T& high) {
        return (val < low) ? low : (high < val) ? high : val;
    }

private:
    inline float GetImageValue(const cv::Mat& img, float x, float y) const;

    float ZNCC(const cv::Point2f& prev_point,
               const cv::Point2f& next_point) const;

    float CalOnePoint(const cv::Point2f& point,
                     const cv::Point2f& delta,
                     float dx,
                     float dy,
                     int iter,
                     Eigen::Matrix3f& H,
                     Eigen::Vector3f& b) const;

   private:
    const cv::Mat& prev_img_;
    const cv::Mat& next_img_;
    const std::vector<std::vector<cv::Point2f>>& prev_lines_;
    std::vector<std::vector<cv::Point2f>>& next_lines_;
    std::vector<uchar>& status_;
    std::vector<float>& params_;
    int level_;
    bool check_;
    cv::Size winsize_;
    cv::Size zncc_winsize_;
    cv::TermCriteria criteria_;
    cv::Point2f half_win_;
    cv::Point2f zncc_half_win_;

};