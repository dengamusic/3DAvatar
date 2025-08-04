#pragma once
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <vector>
#include <algorithm>

struct PixelKP { int jid; double u, v; };

struct Sim3Params {
    double data[7]; // [s, aa(3), t(3)]
    double& scale()             { return data[0]; }
    double* aa_root()           { return data + 1; }
    double* trans()             { return data + 4; }
    const double& scale() const { return data[0]; }
    const double* aa_root() const { return data + 1; }
    const double* trans()   const { return data + 4; }
};

struct ReprojCostSim3 {
    ReprojCostSim3(const Eigen::Vector3d& J,
                   double u_obs, double v_obs,
                   double fx, double fy, double cx, double cy)
        : J_(J), u_obs_(u_obs), v_obs_(v_obs),
          fx_(fx), fy_(fy), cx_(cx), cy_(cy) {}

    template <typename T>
    bool operator()(const T* const sim3, T* residuals) const {
        // sim3 = [scale, angle-axis(3), translation(3)]
        const T& s = sim3[0];
        const T aa[3] = { sim3[1], sim3[2], sim3[3] };
        const T t[3]  = { sim3[4], sim3[5], sim3[6] };

        // Rotate 3D joint position by angle-axis
        T Jp[3] = { T(J_(0)), T(J_(1)), T(J_(2)) };
        T Jrot[3];
        ceres::AngleAxisRotatePoint(aa, Jp, Jrot);

        // Scale and translate
        T X[3] = { s * Jrot[0] + t[0],
                   s * Jrot[1] + t[1],
                   s * Jrot[2] + t[2] };

        // Project to image
        T u = T(fx_) * X[0] / X[2] + T(cx_);
        T v = T(fy_) * X[1] / X[2] + T(cy_);

        // Residual = (projected - observed)
        residuals[0] = u - T(u_obs_);
        residuals[1] = v - T(v_obs_);
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& J,
                                       double u_obs, double v_obs,
                                       double fx, double fy, double cx, double cy) {
        return new ceres::AutoDiffCostFunction<ReprojCostSim3, 2, 7>(
            new ReprojCostSim3(J, u_obs, v_obs, fx, fy, cx, cy)
        );
    }

    const Eigen::Vector3d J_;
    const double u_obs_, v_obs_;
    const double fx_, fy_, cx_, cy_;
};

inline std::pair<Sim3Params, std::string>
OptimizeSim3Reprojection(const Eigen::Matrix<double,3,Eigen::Dynamic>& avatarJoints,
                         const std::vector<PixelKP>& kps,
                         double fx, double fy, double cx, double cy,
                         const std::vector<int>& valid_joint_ids = {},
                         const Sim3Params* init = nullptr,
                         int max_iters = 100)
{
    // Initialize parameters
    Sim3Params params{};
    params.scale() = 1.0;
    params.aa_root()[0] = params.aa_root()[1] = params.aa_root()[2] = 0.0;
    params.trans()[0] = params.trans()[1] = 0.0;
    params.trans()[2] = 3.0;
    if (init) params = *init;

    ceres::Problem problem;
    ceres::LossFunction* loss = new ceres::HuberLoss(3.0);

    // Add reprojection residuals for each keypoint
    for (const auto& kp : kps) {
        if (kp.jid < 0 || kp.jid >= avatarJoints.cols()) continue;
        if (!valid_joint_ids.empty() &&
            std::find(valid_joint_ids.begin(), valid_joint_ids.end(), kp.jid) == valid_joint_ids.end()) {
            continue;
        }
        Eigen::Vector3d J = avatarJoints.col(kp.jid);
        ceres::CostFunction* cost = ReprojCostSim3::Create(J, kp.u, kp.v, fx, fy, cx, cy);
        problem.AddResidualBlock(cost, loss, params.data);
    }

    // Constrain scale to avoid degeneracy
    problem.SetParameterLowerBound(params.data, 0, 0.3);
    problem.SetParameterUpperBound(params.data, 0, 3.0);

    // Solve Sim3 optimization
    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_QR;
    opts.max_num_iterations = max_iters;
    opts.num_threads = 4;
    opts.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);
    return { params, summary.FullReport() };
}

struct ReprojCostFull {
    ReprojCostFull(int jid,
                   double u_obs, double v_obs,
                   double fx, double fy, double cx, double cy,
                   const std::vector<int>& parent,
                   const std::vector<Eigen::Vector3d>& offset,
                   const Eigen::Matrix3d& R0_init,
                   int nJ)
        : jid_(jid), u_obs_(u_obs), v_obs_(v_obs),
          fx_(fx), fy_(fy), cx_(cx), cy_(cy),
          parent_(parent), offset_(offset), R0_(R0_init), nJ_(nJ) {}

    template <typename T>
    bool operator()(T const* const* params, T* residuals) const {
        // Parameter layout (order must match AddParameterBlock calls):
        // 0: scale(1)
        // 1: rootAA(3)
        // 2: rootTrans(3)
        // 3..(3 + (nJ_-1) - 1): jointAA[1..nJ_-1](3 each)

        const T* scale    = params[0];
        const T* rootAA   = params[1];
        const T* rootT    = params[2];

        auto jointAA = [&](int j)->const T* {
            // j in [1..nJ_-1]
            return params[3 + (j - 1)];
        };

        // === start from this joint's base offset (in components) ===
        Eigen::Matrix<T,3,1> pos_T = offset_[jid_].template cast<T>();

        // Walk up the kinematic chain (exclude root 0)
        int cur = jid_;
        while (parent_[cur] != -1 && parent_[cur] != 0) {
            const int p = parent_[cur];

            // rotate by ancestor p
            const T* aa = jointAA(p);
            ceres::AngleAxisRotatePoint(aa, pos_T.data(), pos_T.data());

            // then translate by that parent's base offset (in components)
            pos_T += offset_[p].template cast<T>();

            cur = p;
        }


        // Fixed initial root orientation (double -> T per component)
        Eigen::Matrix<T,3,1> pos_cam;
        pos_cam(0) = T(R0_(0,0)) * pos_T(0) + T(R0_(0,1)) * pos_T(1) + T(R0_(0,2)) * pos_T(2);
        pos_cam(1) = T(R0_(1,0)) * pos_T(0) + T(R0_(1,1)) * pos_T(1) + T(R0_(1,2)) * pos_T(2);
        pos_cam(2) = T(R0_(2,0)) * pos_T(0) + T(R0_(2,1)) * pos_T(1) + T(R0_(2,2)) * pos_T(2);

        // Root rotation, then scale & translation
        ceres::AngleAxisRotatePoint(rootAA, pos_cam.data(), pos_cam.data());
        pos_cam(0) = (*scale) * pos_cam(0) + rootT[0];
        pos_cam(1) = (*scale) * pos_cam(1) + rootT[1];
        pos_cam(2) = (*scale) * pos_cam(2) + rootT[2];

        // Project
        T u = T(fx_) * pos_cam(0) / pos_cam(2) + T(cx_);
        T v = T(fy_) * pos_cam(1) / pos_cam(2) + T(cy_);
        residuals[0] = u - T(u_obs_);
        residuals[1] = v - T(v_obs_);
        return true;
    }



    const int jid_;
    const double u_obs_, v_obs_;
    const double fx_, fy_, cx_, cy_;
    const std::vector<int>& parent_;
    const std::vector<Eigen::Vector3d>& offset_;
    const Eigen::Matrix3d R0_;
    const int nJ_;
};


inline std::pair<bool, std::string>
OptimizePoseReprojection(const ark::AvatarModel& model,
                         ark::Avatar& avatar,
                         const std::vector<PixelKP>& kps,
                         double fx, double fy, double cx, double cy,
                         const std::vector<int>& valid_joint_ids,
                         Sim3Params& initSim3,
                         int max_iters = 100)
{
    int nJ = model.numJoints();
    // Prepare parent index array
    std::vector<int> parent(nJ);
    for (int j = 0; j < nJ; ++j) parent[j] = model.parent[j];

    // Compute base (zero-pose) joint positions in model coordinates
    ark::Avatar base_avatar(model);
    base_avatar.w.setZero();
    base_avatar.p = Eigen::Vector3d(0,0,0);
    base_avatar.r.clear();
    base_avatar.r.resize(nJ, Eigen::Matrix3d::Identity());
    base_avatar.update();
    Eigen::Matrix<double,3,Eigen::Dynamic> baseJoints = base_avatar.jointPos;
    // Translate so that root joint is at origin
    Eigen::Vector3d root_offset = baseJoints.col(0);
    for (int j = 0; j < nJ; ++j) {
        baseJoints.col(j) -= root_offset;
    }
    // Compute offset vectors from parent to each joint in base pose
    std::vector<Eigen::Vector3d> offset(nJ);
    offset[0] = Eigen::Vector3d(0,0,0);
    for (int j = 1; j < nJ; ++j) {
        int pj = parent[j];
        if (pj >= 0) {
            offset[j] = (baseJoints.col(j) - baseJoints.col(pj)).eval();
        } else {
            offset[j] = baseJoints.col(j);
        }
    }

    // Initial root orientation (yaw + flipY) as a fixed matrix
    Eigen::Matrix3d R0_init = avatar.r[0];

    // Set up parameter blocks for optimization
    double scale = initSim3.scale();
    double rootAA[3] = { initSim3.aa_root()[0], initSim3.aa_root()[1], initSim3.aa_root()[2] };
    double rootTrans[3] = { initSim3.trans()[0], initSim3.trans()[1], initSim3.trans()[2] };
    std::vector<std::array<double,3>> jointAA(nJ);
    for (int j = 0; j < nJ; ++j) {
        jointAA[j] = { 0.0, 0.0, 0.0 };
    }

    ceres::Problem problem;
    ceres::LossFunction* loss = new ceres::HuberLoss(3.0);
    // Add residuals for each observed keypoint
    for (const auto& kp : kps) {
        if (!valid_joint_ids.empty() &&
            std::find(valid_joint_ids.begin(), valid_joint_ids.end(), kp.jid) == valid_joint_ids.end()) {
            continue;
        }
        int jid = kp.jid;
        // Create cost functor for this joint
        // Create the DynamicAutoDiff cost and declare parameter block sizes
        auto* functor = new ReprojCostFull(jid, kp.u, kp.v, fx, fy, cx, cy, parent, offset, R0_init, nJ);
        auto* cost = new ceres::DynamicAutoDiffCostFunction<ReprojCostFull>(functor);

        // Order must match operator() layout:
        cost->AddParameterBlock(1); // scale
        cost->AddParameterBlock(3); // rootAA
        cost->AddParameterBlock(3); // rootTrans
        for (int j = 1; j < nJ; ++j) {
            cost->AddParameterBlock(3); // jointAA[j]
        }
        cost->SetNumResiduals(2);

        // Build the vector of parameter pointers (same order):
        std::vector<double*> params;
        params.reserve(3 + (nJ - 1));
        params.push_back(&scale);
        params.push_back(rootAA);
        params.push_back(rootTrans);
        for (int j = 1; j < nJ; ++j) {
            params.push_back(jointAA[j].data());
        }

        // Add residual
        problem.AddResidualBlock(cost, loss, params);

        // problem.AddResidualBlock(cost, loss,
        //     &scale, 
        //     rootAA, 
        //     rootTrans,
        //     jointAA[1].data(), jointAA[2].data(), jointAA[3].data(), jointAA[4].data(),
        //     jointAA[5].data(), jointAA[6].data(), jointAA[7].data(), jointAA[8].data(),
        //     jointAA[9].data(), jointAA[10].data(), jointAA[11].data(), jointAA[12].data(),
        //     jointAA[13].data(), jointAA[14].data(), jointAA[15].data(), jointAA[16].data(),
        //     jointAA[17].data(), jointAA[18].data(), jointAA[19].data(), jointAA[20].data(),
        //     jointAA[21].data(), jointAA[22].data(), jointAA[23].data());
    }

    // Fix joints that have no observations (e.g., feet and hands not observed by MediaPipe)
    if (nJ > 10) problem.SetParameterBlockConstant(jointAA[10].data());
    if (nJ > 11) problem.SetParameterBlockConstant(jointAA[11].data());
    if (nJ > 22) problem.SetParameterBlockConstant(jointAA[22].data());
    if (nJ > 23) problem.SetParameterBlockConstant(jointAA[23].data());
    // Bound scale to a reasonable range
    problem.SetParameterLowerBound(&scale, 0, 0.3);
    problem.SetParameterUpperBound(&scale, 0, 3.0);

    // Solve full pose optimization
    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::DENSE_QR;
    opts.max_num_iterations = max_iters;
    opts.num_threads = 4;
    opts.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(opts, &problem, &summary);

    // Update avatar with optimized pose
    // Compute root rotation matrix from optimized angle-axis
    Eigen::Vector3d aa_root(rootAA[0], rootAA[1], rootAA[2]);
    double theta = aa_root.norm();
    Eigen::Matrix3d R_root = Eigen::Matrix3d::Identity();
    if (theta > 1e-12) {
        R_root = Eigen::AngleAxisd(theta, aa_root / theta).toRotationMatrix();
    }
    // avatar.r[0] was initial R0_init, update to include optimized root rotation
    avatar.r[0] = R_root * avatar.r[0];
    // Set each joint's rotation from its optimized angle-axis
    for (int j = 1; j < nJ; ++j) {
        Eigen::Vector3d aa(jointAA[j][0], jointAA[j][1], jointAA[j][2]);
        double th = aa.norm();
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        if (th > 1e-12) {
            R = Eigen::AngleAxisd(th, aa / th).toRotationMatrix();
        }
        avatar.r[j] = R;
    }
    // Update translation
    avatar.p = Eigen::Vector3d(rootTrans[0], rootTrans[1], rootTrans[2]);

    // Store final scale and root transform in initSim3 for reference/output
    initSim3.scale() = scale;
    initSim3.aa_root()[0] = rootAA[0];
    initSim3.aa_root()[1] = rootAA[1];
    initSim3.aa_root()[2] = rootAA[2];
    initSim3.trans()[0] = rootTrans[0];
    initSim3.trans()[1] = rootTrans[1];
    initSim3.trans()[2] = rootTrans[2];

    bool success = summary.IsSolutionUsable();
    return { success, summary.FullReport() };
}
