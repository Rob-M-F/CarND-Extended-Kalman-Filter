#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd y = z - H_ * x_;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();
  x_ = x_ + (K * y);
  P_ = (MatrixXd::Identity(4, 4) - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
    float pi = 3.14159;
    float px = x_[0], py = x_[1];
    float vx = x_[2], vy = x_[3];

    float rho = sqrt(px*px + py*py);
    float theta = atan2(py, px);
    float rho_d = 0;

    while (theta > pi) 
      {theta -= 2*pi;}
    while (theta < -pi) 
      {theta += 2*pi;}

    // Extra logic to handle out of bounds values in the data.
    if ((z[1] > pi) & (theta < 0))
      {theta += 2*pi;}
    if ((z[1] < -pi) & (theta > 0)) 
      {theta -= 2*pi;}

    if (fabs(rho) >= 0.0001)
      {rho_d = (px*vx + py*vy)/rho;}

    VectorXd pred = VectorXd(3);
    pred << rho, theta, rho_d;
    VectorXd y = z - pred;

    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd K = P_ * Ht * S.inverse();

    x_ = x_ + (K * y);
    P_ = (MatrixXd::Identity(4, 4) - K * H_) * P_;
}
