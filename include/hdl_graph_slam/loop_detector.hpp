// SPDX-License-Identifier: BSD-2-Clause

#ifndef LOOP_DETECTOR_HPP
#define LOOP_DETECTOR_HPP

#include <boost/format.hpp>
#include <hdl_graph_slam/keyframe.hpp>
#include <hdl_graph_slam/graph_slam.hpp>

#include <pcl/registration/ndt.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>

#include <pclomp/ndt_omp.h>
#include <pclomp/gicp_omp.h>
#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>

#include <g2o/types/slam3d/vertex_se3.h>

namespace hdl_graph_slam {

struct Loop {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Ptr = std::shared_ptr<Loop>;

  Loop(const KeyFrame::Ptr& key1, const KeyFrame::Ptr& key2, const Eigen::Matrix4f& relpose) : key1(key1), key2(key2), relative_pose(relpose) {}

public:
  KeyFrame::Ptr key1;
  KeyFrame::Ptr key2;
  Eigen::Matrix4f relative_pose;
};

/**
 * @brief this class finds loops by scam matching and adds them to the pose graph
 */
class LoopDetector {
public:
  typedef pcl::PointXYZI PointT;

  /**
   * @brief constructor
   * @param nh
   */
  LoopDetector() {
    distance_thresh = nh.param<double>("hdl_graph_slam/distance_thresh", 5.0);
    accum_distance_thresh = nh.param<double>("hdl_graph_slam/accum_distance_thresh", 8.0);
    distance_from_last_edge_thresh = nh.param<double>("hdl_graph_slam/min_edge_interval", 5.0);

    fitness_score_max_range = nh.param<double>("hdl_graph_slam/fitness_score_max_range", std::numeric_limits<double>::max());
    fitness_score_thresh = nh.param<double>("hdl_graph_slam/fitness_score_thresh", 0.5);

    registration = select_registration_method();
    last_edge_accum_distance = 0.0;
  }


  pcl::Registration<pcl::PointXYZI, pcl::PointXYZI>::Ptr select_registration_method() {
    // select a registration method (ICP, GICP, NDT)
    std::string registration_method = nh.param<std::string>("hdl_graph_slam/registration_method", "NDT_OMP");
    if(registration_method == "FAST_GICP") {
      std::cout << "registration: FAST_GICP" << std::endl;
      fast_gicp::FastGICP<PointT, PointT>::Ptr gicp(new fast_gicp::FastGICP<PointT, PointT>());
      gicp->setNumThreads(nh.param<int>("hdl_graph_slam/reg_num_threads", 0));
      gicp->setTransformationEpsilon(nh.param<double>("hdl_graph_slam/reg_transformation_epsilon", 0.01));
      gicp->setMaximumIterations(nh.param<int>("hdl_graph_slam/reg_maximum_iterations", 64));
      gicp->setMaxCorrespondenceDistance(nh.param<double>("hdl_graph_slam/reg_max_correspondence_distance", 2.5));
      gicp->setCorrespondenceRandomness(nh.param<int>("hdl_graph_slam/reg_correspondence_randomness", 20));
      return gicp;
    }
    else if(registration_method == "FAST_VGICP") {
      std::cout << "registration: FAST_VGICP" << std::endl;
      fast_gicp::FastVGICP<PointT, PointT>::Ptr vgicp(new fast_gicp::FastVGICP<PointT, PointT>());
      vgicp->setNumThreads(nh.param<int>("hdl_graph_slam/reg_num_threads", 0));
      vgicp->setResolution(nh.param<double>("hdl_graph_slam/reg_resolution", 1.0));
      vgicp->setTransformationEpsilon(nh.param<double>("hdl_graph_slam/reg_transformation_epsilon", 0.01));
      vgicp->setMaximumIterations(nh.param<int>("hdl_graph_slam/reg_maximum_iterations", 64));
      vgicp->setCorrespondenceRandomness(nh.param<int>("hdl_graph_slam/reg_correspondence_randomness", 20));
      return vgicp;
    } else if(registration_method == "ICP") {
      std::cout << "registration: ICP" << std::endl;
      pcl::IterativeClosestPoint<PointT, PointT>::Ptr icp(new pcl::IterativeClosestPoint<PointT, PointT>());
      icp->setTransformationEpsilon(nh.param<double>("hdl_graph_slam/reg_transformation_epsilon", 0.01));
      icp->setMaximumIterations(nh.param<int>("hdl_graph_slam/reg_maximum_iterations", 64));
      icp->setMaxCorrespondenceDistance(nh.param<double>("hdl_graph_slam/reg_max_correspondence_distance", 2.5));
      icp->setUseReciprocalCorrespondences(nh.param<bool>("hdl_graph_slam/reg_use_reciprocal_correspondences", false));
      return icp;
    } else if(registration_method.find("GICP") != std::string::npos) {
      if(registration_method.find("OMP") == std::string::npos) {
        std::cout << "registration: GICP" << std::endl;
        pcl::GeneralizedIterativeClosestPoint<PointT, PointT>::Ptr gicp(new pcl::GeneralizedIterativeClosestPoint<PointT, PointT>());
        gicp->setTransformationEpsilon(nh.param<double>("hdl_graph_slam/reg_transformation_epsilon", 0.01));
        gicp->setMaximumIterations(nh.param<int>("hdl_graph_slam/reg_maximum_iterations", 64));
        gicp->setUseReciprocalCorrespondences(nh.param<bool>("hdl_graph_slam/reg_use_reciprocal_correspondences", false));
        gicp->setMaxCorrespondenceDistance(nh.param<double>("hdl_graph_slam/reg_max_correspondence_distance", 2.5));
        gicp->setCorrespondenceRandomness(nh.param<int>("hdl_graph_slam/reg_correspondence_randomness", 20));
        gicp->setMaximumOptimizerIterations(nh.param<int>("hdl_graph_slam/reg_max_optimizer_iterations", 20));
        return gicp;
      } else {
        std::cout << "registration: GICP_OMP" << std::endl;
        pclomp::GeneralizedIterativeClosestPoint<PointT, PointT>::Ptr gicp(new pclomp::GeneralizedIterativeClosestPoint<PointT, PointT>());
        gicp->setTransformationEpsilon(nh.param<double>("hdl_graph_slam/reg_transformation_epsilon", 0.01));
        gicp->setMaximumIterations(nh.param<int>("hdl_graph_slam/reg_maximum_iterations", 64));
        gicp->setUseReciprocalCorrespondences(nh.param<bool>("hdl_graph_slam/reg_use_reciprocal_correspondences", false));
        gicp->setMaxCorrespondenceDistance(nh.param<double>("hdl_graph_slam/reg_max_correspondence_distance", 2.5));
        gicp->setCorrespondenceRandomness(nh.param<int>("hdl_graph_slam/reg_correspondence_randomness", 20));
        gicp->setMaximumOptimizerIterations(nh.param<int>("hdl_graph_slam/reg_max_optimizer_iterations", 20));
        return gicp;
      }
    } else {
      if(registration_method.find("NDT") == std::string::npos) {
        std::cerr << "warning: unknown registration type(" << registration_method << ")" << std::endl;
        std::cerr << "       : use NDT" << std::endl;
      }

      double ndt_resolution = nh.param<double>("hdl_graph_slam/reg_resolution", 0.5);
      if(registration_method.find("OMP") == std::string::npos) {
        std::cout << "registration: NDT " << ndt_resolution << std::endl;
        pcl::NormalDistributionsTransform<PointT, PointT>::Ptr ndt(new pcl::NormalDistributionsTransform<PointT, PointT>());
        ndt->setTransformationEpsilon(nh.param<double>("hdl_graph_slam/reg_transformation_epsilon", 0.01));
        ndt->setMaximumIterations(nh.param<int>("hdl_graph_slam/reg_maximum_iterations", 64));
        ndt->setResolution(ndt_resolution);
        return ndt;
      } else {
        int num_threads = nh.param<int>("hdl_graph_slam/reg_num_threads", 0);
        std::string nn_search_method = nh.param<std::string>("hdl_graph_slam/reg_nn_search_method", "DIRECT7");
        std::cout << "registration: NDT_OMP " << nn_search_method << " " << ndt_resolution << " (" << num_threads << " threads)" << std::endl;
        pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr ndt(new pclomp::NormalDistributionsTransform<PointT, PointT>());
        if(num_threads > 0) {
          ndt->setNumThreads(num_threads);
        }
        ndt->setTransformationEpsilon(nh.param<double>("hdl_graph_slam/reg_transformation_epsilon", 0.01));
        ndt->setMaximumIterations(nh.param<int>("hdl_graph_slam/reg_maximum_iterations", 64));
        ndt->setResolution(ndt_resolution);
        if(nn_search_method == "KDTREE") {
          ndt->setNeighborhoodSearchMethod(pclomp::KDTREE);
        } else if(nn_search_method == "DIRECT1") {
          ndt->setNeighborhoodSearchMethod(pclomp::DIRECT1);
        } else {
          ndt->setNeighborhoodSearchMethod(pclomp::DIRECT7);
        }
        return ndt;
      }
    }

    return nullptr;
  }



  /**
   * @brief detect loops and add them to the pose graph
   * @param keyframes       keyframes
   * @param new_keyframes   newly registered keyframes
   * @param graph_slam      pose graph
   */
  std::vector<Loop::Ptr> detect(const std::vector<KeyFrame::Ptr>& keyframes, const std::deque<KeyFrame::Ptr>& new_keyframes, hdl_graph_slam::GraphSLAM& graph_slam) {
    std::vector<Loop::Ptr> detected_loops;
    for(const auto& new_keyframe : new_keyframes) {
      auto candidates = find_candidates(keyframes, new_keyframe);
      auto loop = matching(candidates, new_keyframe, graph_slam);
      if(loop) {
        detected_loops.push_back(loop);
      }
    }

    return detected_loops;
  }

  double get_distance_thresh() const {
    return distance_thresh;
  }

private:
  /**
   * @brief find loop candidates. A detected loop begins at one of #keyframes and ends at #new_keyframe
   * @param keyframes      candidate keyframes of loop start
   * @param new_keyframe   loop end keyframe
   * @return loop candidates
   */
  std::vector<KeyFrame::Ptr> find_candidates(const std::vector<KeyFrame::Ptr>& keyframes, const KeyFrame::Ptr& new_keyframe) const {
    // too close to the last registered loop edge
    if(new_keyframe->accum_distance - last_edge_accum_distance < distance_from_last_edge_thresh) {
      return std::vector<KeyFrame::Ptr>();
    }

    std::vector<KeyFrame::Ptr> candidates;
    candidates.reserve(32);

    for(const auto& k : keyframes) {
      // traveled distance between keyframes is too small
      if(new_keyframe->accum_distance - k->accum_distance < accum_distance_thresh) {
        continue;
      }

      const auto& pos1 = k->node->estimate().translation();
      const auto& pos2 = new_keyframe->node->estimate().translation();

      // estimated distance between keyframes is too small
      double dist = (pos1.head<2>() - pos2.head<2>()).norm();
      if(dist > distance_thresh) {
        continue;
      }

      candidates.push_back(k);
    }

    return candidates;
  }

  /**
   * @brief To validate a loop candidate this function applies a scan matching between keyframes consisting the loop. If they are matched well, the loop is added to the pose graph
   * @param candidate_keyframes  candidate keyframes of loop start
   * @param new_keyframe         loop end keyframe
   * @param graph_slam           graph slam
   */
  Loop::Ptr matching(const std::vector<KeyFrame::Ptr>& candidate_keyframes, const KeyFrame::Ptr& new_keyframe, hdl_graph_slam::GraphSLAM& graph_slam) {
    if(candidate_keyframes.empty()) {
      return nullptr;
    }

    registration->setInputTarget(new_keyframe->cloud);

    double best_score = std::numeric_limits<double>::max();
    KeyFrame::Ptr best_matched;
    Eigen::Matrix4f relative_pose;

    std::cout << std::endl;
    std::cout << "--- loop detection ---" << std::endl;
    std::cout << "num_candidates: " << candidate_keyframes.size() << std::endl;
    std::cout << "matching" << std::flush;
    auto t1 = ros::Time::now();

    pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
    for(const auto& candidate : candidate_keyframes) {
      registration->setInputSource(candidate->cloud);
      Eigen::Isometry3d new_keyframe_estimate = new_keyframe->node->estimate();
      new_keyframe_estimate.linear() = Eigen::Quaterniond(new_keyframe_estimate.linear()).normalized().toRotationMatrix();
      Eigen::Isometry3d candidate_estimate = candidate->node->estimate();
      candidate_estimate.linear() = Eigen::Quaterniond(candidate_estimate.linear()).normalized().toRotationMatrix();
      Eigen::Matrix4f guess = (new_keyframe_estimate.inverse() * candidate_estimate).matrix().cast<float>();
      guess(2, 3) = 0.0;
      registration->align(*aligned, guess);
      std::cout << "." << std::flush;

      double score = registration->getFitnessScore(fitness_score_max_range);
      if(!registration->hasConverged() || score > best_score) {
        continue;
      }

      best_score = score;
      best_matched = candidate;
      relative_pose = registration->getFinalTransformation();
    }

    auto t2 = ros::Time::now();
    std::cout << " done" << std::endl;
    std::cout << "best_score: " << boost::format("%.3f") % best_score << "    time: " << boost::format("%.3f") % (t2 - t1).toSec() << "[sec]" << std::endl;

    if(best_score > fitness_score_thresh) {
      std::cout << "loop not found..." << std::endl;
      return nullptr;
    }

    std::cout << "loop found!!" << std::endl;
    std::cout << "relpose: " << relative_pose.block<3, 1>(0, 3) << " - " << Eigen::Quaternionf(relative_pose.block<3, 3>(0, 0)).coeffs().transpose() << std::endl;

    last_edge_accum_distance = new_keyframe->accum_distance;

    return std::make_shared<Loop>(new_keyframe, best_matched, relative_pose);
  }

private:
  double distance_thresh;                 // estimated distance between keyframes consisting a loop must be less than this distance
  double accum_distance_thresh;           // traveled distance between ...
  double distance_from_last_edge_thresh;  // a new loop edge must far from the last one at least this distance

  double fitness_score_max_range;  // maximum allowable distance between corresponding points
  double fitness_score_thresh;     // threshold for scan matching

  double last_edge_accum_distance;
  ros::NodeHandle nh;

  pcl::Registration<PointT, PointT>::Ptr registration;
};

}  // namespace hdl_graph_slam

#endif  // LOOP_DETECTOR_HPP
