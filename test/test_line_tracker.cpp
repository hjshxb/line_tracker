
#include <iostream>
#include <glog/logging.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/optflow.hpp>

#include "timer.h"
#include "line_flow.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "optical_flow img1 img2\n";
        return 1;
    }
    cv::Mat img1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    // cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    // clahe->apply(img1,img1);
    // clahe->apply(img2,img2);

    cv::Ptr<cv::ximgproc::EdgeDrawing> ed = cv::ximgproc::createEdgeDrawing();
    ed->params.EdgeDetectionOperator = cv::ximgproc::EdgeDrawing::SOBEL;
    ed->params.MinLineLength = 0.1 * std::min(img1.cols, img1.rows);

    std::vector<cv::Vec4f> prev_lines;
    auto t1 = Timer::tic();
    ed->detectEdges(img1);
    LOG(INFO) << "Detect edges using " << Timer::toc(t1).count();
    t1 = Timer::tic();
    ed->detectLines(prev_lines);
    LOG(INFO) << "Detect lines using " << Timer::toc(t1).count();

    // LINE LK track
    LineLKTracker line_tracker(cv::Size(21, 21), 5, 3, true);
    std::vector<cv::Vec4f> next_lines;
    std::vector<uchar> status;
    t1 = Timer::tic();
    line_tracker(img1, img2, prev_lines, next_lines, status);
    LOG(INFO) << "line flow using size(21, 21) " << Timer::toc(t1).count();

    cv::Mat draw_img1;
    cv::cvtColor(img1, draw_img1, cv::COLOR_GRAY2BGR);

    for (size_t i = 0; i < prev_lines.size(); ++i) {
        cv::line(draw_img1,
                 cv::Point2f(prev_lines[i][0], prev_lines[i][1]),
                 cv::Point2f(prev_lines[i][2], prev_lines[i][3]),
                 cv::Scalar(0, 0, 255),
                 1,
                 cv::LINE_AA);
    }

    cv::Mat draw_img2;
    cv::cvtColor(img2, draw_img2, cv::COLOR_GRAY2BGR);
    int succ_sum = 0;

    for (size_t i = 0; i < next_lines.size(); ++i) {
        cv::line(draw_img2,
                 cv::Point2f(prev_lines[i][0], prev_lines[i][1]),
                 cv::Point2f(prev_lines[i][2], prev_lines[i][3]),
                 cv::Scalar(0, 255, 0),
                 1,
                 cv::LINE_AA);
        if (!status[i]) continue;
        succ_sum++;
        cv::line(draw_img2,
                 cv::Point2f(next_lines[i][0], next_lines[i][1]),
                 cv::Point2f(next_lines[i][2], next_lines[i][3]),
                 cv::Scalar(0, 0, 255),
                 1,
                 cv::LINE_AA);
        
        cv::line(draw_img2,
                 cv::Point2f(prev_lines[i][0], prev_lines[i][1]),
                 cv::Point2f(next_lines[i][0], next_lines[i][1]),
                 cv::Scalar(255, 0, 0),
                 1,
                 cv::LINE_AA);
        cv::line(draw_img2,
                 cv::Point2f(prev_lines[i][2], prev_lines[i][3]),
                 cv::Point2f(next_lines[i][2], next_lines[i][3]),
                 cv::Scalar(255, 0, 0),
                 1,
                 cv::LINE_AA);
    }
    LOG(INFO) 
        << "Successful tracking line: " << succ_sum 
        << "; total line: " << prev_lines.size()
        << "; ratio " << float(succ_sum) / prev_lines.size();

    cv::imshow("detected lines", draw_img1);
    // cv::imwrite("detected_lines.png", draw_img1);
    cv::imshow("predicted lines", draw_img2);
    // cv::imwrite("predicted_lines.png", draw_img2);
    cv::waitKey();

    return 0;
}