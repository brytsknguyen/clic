/*
 * Continuous-Time Fixed-Lag Smoothing for LiDAR-Inertial-Camera SLAM
 * Copyright (C) 2022 Jiajun Lv
 * Copyright (C) 2022 Xiaolei Lang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include <estimator/factor/analytic_diff/rd_spline_view.h>
#include <estimator/factor/analytic_diff/so3_spline_view.h>

#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"
#define RESET "\033[0m"

using namespace Eigen;

namespace clic {
namespace analytic_derivative {

struct SplitSpineView : public So3SplineView, public RdSplineView {
  static constexpr int N = SplineOrder;  // Order of the spline.
  static constexpr int DEG = N - 1;      // Degree of the spline.
  static constexpr int DIM = 3;          // Dimension of euclidean vector space.

  using MatN = Eigen::Matrix<double, N, N>;
  using VecN = Eigen::Matrix<double, N, 1>;
  using Vec3 = Eigen::Matrix<double, 3, 1>;
  using Mat3 = Eigen::Matrix<double, 3, 3>;

  using VecD = Eigen::Matrix<double, DIM, 1>;
  using MatD = Eigen::Matrix<double, DIM, DIM>;

  using SO3 = Sophus::SO3<double>;

  using SO3View = So3SplineView;
  using R3View = RdSplineView;
  
  struct SplineIMUData {
    int64_t time_ns;
    Eigen::Vector3d gyro;
    VecD accel;
    SO3 R_inv;
    size_t start_idx;
  };

  SplitSpineView() {}

  static Eigen::Matrix3d rightJacobian(const Eigen::Vector3d &phi)
  {
      Matrix3d Jr;
      Sophus::rightJacobianSO3(phi, Jr);
      return Jr;
  }

  static Eigen::Matrix3d rightJacobianInv(const Eigen::Vector3d &phi)
  {
      Matrix3d Jr_inv;
      Sophus::rightJacobianInvSO3(phi, Jr_inv);
      return Jr_inv;
  }

  static void SplioJw(double const* const* knots, size_t u, double s, double Dt_inv, Vector3d GRAV,
                      Eigen::Matrix<double, N, 1> lambda_R,
                      Eigen::Matrix<double, N, 1> lambda_R_dot,
                      Eigen::Matrix<double, N, 1> lambda_p_ddot)
  {
    using namespace Eigen;
    using namespace std;

    using Quaternd = Eigen::Quaterniond;
    using SO3d = Sophus::SO3<double>;

    double const* const* &parameters = knots;

    // Indexing offsets for the states
    size_t R_offset = u;             // for quaternion
    size_t P_offset = R_offset + N;  // for position
    size_t B_offset = P_offset + N;  // for bias

    // Map parameters to the control point states
    SO3d Rot[N]; Vector3d pos[N];
    for(int j = 0; j < N; j++)
    {
        Rot[j] = Eigen::Map<SO3d const>(parameters[R_offset + j]);
        pos[j] = Eigen::Map<Vector3d const>(parameters[P_offset + j]);

        // printf("SPLIO: p%d. lambda_a: %f\n", j, lambda_p_ddot(j));
        // std::cout << pos[j].transpose() << std::endl;
    }

    // Map parameters to the bias
    Vector3d biasW = Eigen::Map<Vector3d const>(parameters[B_offset + 0]);
    Vector3d biasA = Eigen::Map<Vector3d const>(parameters[B_offset + 1]);

    // The following use the formulations in the paper 2020 CVPR:
    // "Efficient derivative computation for cumulative b-splines on lie groups."
    // Sommer, Christiane, Vladyslav Usenko, David Schubert, Nikolaus Demmel, and Daniel Cremers.
    // Some errors in the paper is corrected

    // Calculate the delta terms: delta(1) ... delta(N-1), where delta(j) = Log( R(j-1)^-1 * R(j) ),
    // delta(0) is an extension in the paper as the index starts at 1
    Vector3d delta[N]; delta[0] = Rot[0].log();
    for (int j = 1; j < N; j++)
        delta[j] = (Rot[j-1].inverse()*Rot[j]).log();

    // The inverse right Jacobian Jr(dj) = d(deltaj)/d(Rj). Note that d( d[j+1] )/d( R[j] ) = -Jr(-d[j+1])
    Matrix3d ddelta_dR[N];
    for (int j = 0; j < N; j++)
        ddelta_dR[j] = rightJacobianInv(delta[j]);

    Matrix3d JrLambaDelta[N];
    for (int j = 0; j < N; j++)
        JrLambaDelta[j] = rightJacobian(lambda_R[j]*delta[j]);    

    // Calculate the A terms: A(1) ... A(N-1), where A(j) = Log( lambda(j) * d(j) ), A(0) is an extension
    SO3d A[N]; A[0] = Rot[0];
    for (int j = 1; j < N; j++)
        A[j] = SO3d::exp(lambda_R(j)*delta[j]).matrix();

    // Calculate the P terms: P(0) ... P(N-1) = I, where P(j) = A(N-1)^-1 A(N-2)^-1 ... A(j+1)^-1
    SO3d P[N]; P[N-1] = SO3d(Quaternd(1, 0, 0, 0));
    for (int j = N - 1; j >= 1; j--)
        P[j-1] = P[j]*A[j].inverse();

    // Jacobian d(Rt)/d(Rj). Derived from the chain rule:
    // d(Rt)/d(Rj) = d(Rt(rho))/d(rho) [ d(rho)/d(dj) . d(dj)/d(Rj) + d(rho)/d(d[j+1]) d(d[j+1]))/d(Rj) ]
    // by using equation (57) in the TUM CVPR paper and some manipulation, we obtain
    // d(Rt)/d(R[j]) = lambda[j] * P[j] * Jr( lambda[j] delta[j] ) * Jr^-1(delta[j])
    //                 - lambda[j+1] * P[j+1] * Jr( lambda[j+1] delta[j+1] ) * Jr^-1( -delta[j+1] )
    Matrix3d dRt_dR[N];
    for (int j = 0; j < N; j++)
    {
        if (j == N-1)
            dRt_dR[j] = lambda_R[j] * P[j].matrix() * JrLambaDelta[j] * ddelta_dR[j];
        else
            dRt_dR[j] = lambda_R[j] * P[j].matrix() * JrLambaDelta[j] * ddelta_dR[j]
                        - lambda_R[j+1] * P[j+1].matrix() * JrLambaDelta[j+1] * ddelta_dR[j+1].transpose();

        // printf("SPLIO dRt_dR%d\n", j);
        // std::cout << dRt_dR[j] << std::endl;
    }

    // Calculate the omega terms: omega(1) ... omega(N), using equation (38), omega(N) is the angular velocity
    Vector3d omega[N+1]; omega[0] = Vector3d(0, 0, 0); omega[1] = Vector3d(0, 0, 0);
    for (int j = 1; j < N+1; j++)
        omega[j] = A[j-1].inverse()*omega[j-1] + lambda_R_dot(j-1)*delta[j-1];

    // Jacobian of d(omega)/d(deltaj)
    Matrix3d domega_ddelta[N];
    for (int j = 1; j < N; j++)
        domega_ddelta[j] = P[j].matrix()*(lambda_R(j)
                                            *A[j].matrix().transpose()
                                            *SO3d::hat(omega[j])
                                            *JrLambaDelta[j].transpose()
                                          + lambda_R_dot[j]*Matrix3d::Identity());

    // Jacobian of d(omega)/d(Rj)
    Matrix3d domega_dR[N];
    for (int j = 0; j < N; j++)
    {
        domega_dR[j].setZero();

        if (j == 0)
            domega_dR[j] = -domega_ddelta[1]*ddelta_dR[1].transpose();
        else if (j == N-1)
            domega_dR[j] =  domega_ddelta[j]*ddelta_dR[j];
        else
            domega_dR[j] =  domega_ddelta[j]*ddelta_dR[j] - domega_ddelta[j+1]*ddelta_dR[j+1].transpose();

        printf("SPLIO: J_omega_R%d\n", j);
        std::cout << domega_dR[j] << std::endl;
    }

    // Predicted gyro is the last omega term
    Vector3d gyro = omega[N];

    // Predicted orientation from Rt^-1 = P(N-1)R(0)^-1
    SO3d R_W_Bt = (P[0]*Rot[0].inverse()).inverse();

    // printf("SPLIO R_W_Bt:\n");
    // std::cout << R_W_Bt.matrix() << std::endl;

    // Predicted acceleration
    Vector3d a_inW_Bt(0, 0, 0);
    for(int j = 0; j < N; j++)
        a_inW_Bt += lambda_p_ddot(j)*pos[j];

    Vector3d a_plus_g = a_inW_Bt + GRAV;
    Vector3d acce = R_W_Bt.inverse().matrix() * a_plus_g;
    Matrix3d da_dRt = SO3d::hat(acce);

    // printf("SPLIO aW:\n");
    // std::cout << a_inW_Bt.transpose() << std::endl;

    // printf("SPLIO g:\n");
    // std::cout << GRAV.transpose() << std::endl;

    // printf("SPLIO aginB:\n");
    // std::cout << a_plus_g.transpose() << std::endl;

    // printf("SPLIO R_W_Bt inv:\n");
    // std::cout << R_W_Bt.inverse().matrix() << std::endl;

    // printf("SPLIO acce:\n");
    // std::cout << acce.transpose() << std::endl;

    // Jacobian of acceleration against d(Rt^-1 (a + g)) / d(Rt)

    // Jacobian of acceleration against rotation knots
    Matrix3d da_dR[N];
    for(int j = 0; j < N; j++)
    {
      da_dR[j] = da_dRt*dRt_dR[j];
      // printf("SPLIO: J_a_R%d\n", j);
      // std::cout << da_dR[j] << std::endl;
    }

    std::cout << std::endl;

    Matrix<double, N, 1> lambda_;
    Matrix<double, N, 1> lambda_dot_;
    Matrix<double, N, 1> lambda_ddot_;
    {
        // Blending matrix
        Matrix<double, N, N> B = clic::computeBlendingMatrix<N, double, false>();

        // Blending matrix
        Matrix<double, N, N> Btilde = clic::computeBlendingMatrix<N, double, true>();
        
        // Time powers
        Matrix<double, N, 1> U;
        for(int j = 0; j < N; j++)
            U(j) = std::pow(s, j);

        // Time power derivative
        Matrix<double, N, 1> Udot = Matrix<double, N, 1>::Zero();
        for(int j = 1; j < N; j++)
            Udot(j) = j*std::pow(s, j-1);

        // Time power derivative
        Matrix<double, N, 1> Uddot = Matrix<double, N, 1>::Zero();
        for(int j = 2; j < N; j++)
            Uddot(j) = j*(j-1)*std::pow(s, j-2);

        // Lambda
        lambda_ = Btilde*U;

        // Lambda dot
        lambda_dot_ = Dt_inv*Btilde*Udot;

        // Lambda ddot
        lambda_ddot_ = Dt_inv*Dt_inv*B*Uddot;
    }

    // printf("CLIC Lambda: \n");
    // cout << lambda_R.transpose() << endl;
    // printf("SPLIO Lambda: \n");
    // cout << lambda_.transpose() << endl;

    // printf("CLIC Lambda dot: \n");
    // cout << lambda_R_dot.transpose() << endl;
    // printf("SPLIO Lambda dot: \n");
    // cout << lambda_dot_.transpose() << endl;

    // printf("CLIC Lambda ddot: \n");
    // cout << lambda_p_ddot.transpose() << endl;
    // printf("SPLIO Lambda ddot: \n");
    // cout << lambda_ddot_.transpose() << endl;

    // printf("R3 blend\n");
    // cout << R3View::blending_matrix_ << endl;

    // printf("R3 blend with time\n");
    // Matrix<double, N, 1> Up;
    // R3View::template baseCoeffsWithTimeR3<2>(Up, s);
    // cout << Up.transpose() << endl;
  }

  static void TumJw(double const* const* knots, size_t s, double u)
  {
    VecN p;
    SO3View::baseCoeffsWithTime<0>(p, u);

    VecN coeff = SO3View::blending_matrix_ * p;

    SO3 res = Eigen::Map<SO3d const>(knots[s]);

    Mat3 J_helper;

    struct JacobianStruct
    {
        size_t start_idx;
        std::array<Mat3, N> d_val_d_knot;
    };

    JacobianStruct* J = new JacobianStruct;

    if (J)
    {
        size_t start_idx = s;
        J_helper.setIdentity();
    }

    for (int i = 0; i < DEG; i++)
    {
        const SO3 &p0 = Eigen::Map<SO3d const>(knots[s + i]);
        const SO3 &p1 = Eigen::Map<SO3d const>(knots[s + i + 1]);

        SO3 r01 = p0.inverse() * p1;
        Vec3 delta = r01.log();
        Vec3 kdelta = delta * coeff[i + 1];

        if (J)
        {
            Mat3 Jl_inv_delta, Jl_k_delta;

            Sophus::leftJacobianInvSO3(delta, Jl_inv_delta);
            Sophus::leftJacobianSO3(kdelta, Jl_k_delta);

            J->d_val_d_knot[i] = J_helper;
            J_helper = coeff[i + 1] * res.matrix() * Jl_k_delta * Jl_inv_delta *
                        p0.inverse().matrix();
            J->d_val_d_knot[i] -= J_helper;

            printf(KGRN "Tum dRt_dR%d\n", i);
            std::cout << J->d_val_d_knot[i] << std::endl;
            printf(RESET);
        }
        res *= SO3::exp(kdelta);
    }

    if (J)
    {
      J->d_val_d_knot[DEG] = J_helper;
      printf(KGRN "Tum dRt_dR%d\n", DEG);
      std::cout << J->d_val_d_knot[DEG] << std::endl;
      printf(RESET);
    }
  }

  static SplineIMUData Evaluate(
      const int64_t time_ns, const SplineSegmentMeta<N>& splne_meta,
      double const* const* knots, const Vec3& gravity,
      typename SO3View::JacobianStruct* J_rot_w = nullptr,
      typename SO3View::JacobianStruct* J_rot_a = nullptr,
      typename R3View::JacobianStruct* J_pos = nullptr) {
    SplineIMUData spline_imu_data;
    spline_imu_data.time_ns = time_ns;

    std::pair<double, size_t> ui = splne_meta.computeTIndexNs(time_ns);
    size_t s = ui.second;
    double u = ui.first;

    size_t R_knot_offset = s;
    size_t P_knot_offset = s + splne_meta.NumParameters();
    spline_imu_data.start_idx = s;

    VecN Up, lambda_a;
    R3View::template baseCoeffsWithTimeR3<2>(Up, u);
    lambda_a = splne_meta.pow_inv_dt[2] * R3View::blending_matrix_ * Up;

    VecN Ur, lambda_R;
    SO3View::template baseCoeffsWithTime<0>(Ur, u);
    lambda_R = SO3View::blending_matrix_ * Ur;

    VecN Uw, lambda_w;
    SO3View::template baseCoeffsWithTime<1>(Uw, u);
    lambda_w = splne_meta.pow_inv_dt[1] * SO3View::blending_matrix_ * Uw;

    VecD accelerate;
    accelerate.setZero();

    if (J_pos) J_pos->start_idx = s;
    for (int i = 0; i < N; i++) {
      Eigen::Map<VecD const> p(knots[P_knot_offset + i]);
      accelerate += lambda_a[i] * p;

      // printf(KRED "CLIC p%d. lambda_a: %f\n", i, lambda_a(i));
      // std::cout << p.transpose() << std::endl;
      // printf(RESET);

      if (J_pos) J_pos->d_val_d_knot[i] = lambda_a[i];
    }

    Vec3 d_vec[DEG];        // d_1, d_2, d_3
    SO3 A_rot_inv[DEG];     // A_1_inv, A_2_inv, A_3_inv
    SO3 A_accum_inv;        // A_3_inv * A2_inv * A1_inv
    Mat3 A_post_inv[N];     // A_3_inv*A2_inv*A1_inv, A_3_inv*A2_inv, A_3_inv, I
    Mat3 Jr_dvec_inv[DEG];  // Jr_inv(d1), Jr_inv(d2), Jr_inv(d3)
    Mat3 Jr_kdelta[DEG];    // Jr(-kd1), Jr(-kd2), Jr(-kd3)

    A_post_inv[N - 1] = A_accum_inv.matrix();  // Identity Matrix
    /// 2 1 0
    for (int i = DEG - 1; i >= 0; i--) {
      Eigen::Map<SO3 const> R0(knots[R_knot_offset + i]);
      Eigen::Map<SO3 const> R1(knots[R_knot_offset + i + 1]);

      d_vec[i] = (R0.inverse() * R1).log();

      Vec3 k_delta = lambda_R[i + 1] * d_vec[i];
      A_rot_inv[i] = Sophus::SO3d::exp(-k_delta);
      A_accum_inv *= A_rot_inv[i];

      if (J_rot_w || J_rot_a) {
        A_post_inv[i] = A_accum_inv.matrix();

        Sophus::rightJacobianInvSO3(d_vec[i], Jr_dvec_inv[i]);
        Sophus::rightJacobianSO3(-k_delta, Jr_kdelta[i]);
      }
    }

    /// Omega(j)
    Vec3 omega[N];  // w(1), w(2), w(3), w(4)
    {
      omega[0] = Vec3::Zero();
      for (int i = 0; i < DEG; i++) {
        omega[i + 1] = A_rot_inv[i] * omega[i] + lambda_w[i + 1] * d_vec[i];
      }
      spline_imu_data.gyro = omega[3];

      Eigen::Map<SO3 const> Ri(knots[R_knot_offset]);
      SO3 R_inv = A_accum_inv * Ri.inverse();

      spline_imu_data.accel = R_inv * (accelerate + gravity);
      spline_imu_data.R_inv = R_inv;
    }

    if (J_rot_w) {
      J_rot_w->start_idx = s;
      for (int i = 0; i < N; i++) {
        J_rot_w->d_val_d_knot[i].setZero();
      }

      // d(omega) / d(d_j)
      Mat3 d_omega_d_delta[DEG];  // w(4)/d1, w(4)/d2, w(4)/d3
      d_omega_d_delta[0] = lambda_w[1] * A_post_inv[1];
      for (int i = 1; i < DEG; i++) {
        d_omega_d_delta[i] = lambda_R[i + 1] * A_post_inv[i] *
                                 SO3::hat(omega[i]) * Jr_kdelta[i] +
                             lambda_w[i + 1] * A_post_inv[i + 1];
      }

      for (int i = 0; i < DEG; i++) {
        J_rot_w->d_val_d_knot[i] -=
            d_omega_d_delta[i] * Jr_dvec_inv[i].transpose();
        J_rot_w->d_val_d_knot[i + 1] += d_omega_d_delta[i] * Jr_dvec_inv[i];

        printf("CLIC: J_omega_R%d\n", i);
        std::cout << J_rot_w->d_val_d_knot[i] << std::endl;
        if (i == DEG-1)
        {
          printf("CLIC: J_omega_R%d\n", i+1);
          std::cout << J_rot_w->d_val_d_knot[i+1] << std::endl;
        }
      }
    }

    if (J_rot_a) {
      // for accelerate jacobian
      Mat3 R_accum[DEG];  // R_i, R_i*A_1, R_i*A_1*A_2
      Eigen::Map<SO3 const> R0(knots[R_knot_offset]);
      R_accum[0] = R0.matrix();
      /// 1 2
      for (int i = 1; i < DEG; i++) {
        R_accum[i] = R_accum[i - 1] * A_rot_inv[i - 1].matrix().transpose();
      }

      J_rot_a->start_idx = s;
      for (int i = 0; i < N; i++) {
        J_rot_a->d_val_d_knot[i].setZero();
      }

      Mat3 lhs =
          spline_imu_data.R_inv.matrix() * SO3::hat(accelerate + gravity);
      J_rot_a->d_val_d_knot[0] += lhs * R_accum[0];
      for (int i = 0; i < DEG; i++) {
        Mat3 d_a_d_delta = lambda_R[i + 1] * lhs * R_accum[i] * Jr_kdelta[i];

        J_rot_a->d_val_d_knot[i] -= d_a_d_delta * Jr_dvec_inv[i].transpose();
        J_rot_a->d_val_d_knot[i + 1] += d_a_d_delta * Jr_dvec_inv[i];

        // printf(KRED "CLIC: J_a_R%d\n", i);
        // std::cout << J_rot_a->d_val_d_knot[i] << std::endl;
        // if (i == DEG-1)
        // {
        //   printf(KRED "CLIC: J_a_R%d\n", i+1);
        //   std::cout << J_rot_a->d_val_d_knot[i+1] << std::endl;
        // }
        // printf(RESET);
      }

      // printf(KRED "CLIC R_W_Bt:\n" );
      // std::cout << spline_imu_data.R_inv.inverse().matrix() << std::endl;
      // printf(RESET);

      // printf(KRED "CLIC acce:\n" );
      // std::cout << (spline_imu_data.R_inv*(accelerate + gravity)).transpose() << std::endl;
      // printf(RESET);
    }

    SplioJw(knots, ui.second, ui.first, splne_meta.pow_inv_dt[1], gravity, lambda_R, lambda_w, lambda_a);

    // TumJw(knots, ui.second, ui.first);

    return spline_imu_data;
  }
};

};  // namespace analytic_derivative

}  // namespace clic