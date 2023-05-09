#include "vins2ground.hpp"

vins2world::vins2world()
{
}

double vins2world::projectPoint(Eigen::Vector3d &point, Eigen::Vector3d &projected)
{
    Eigen::Vector3d v = point-plane_origin_vins_;
    double dist = (v.transpose()*plane_vec_vins_)(0,0);
    projected = point - dist*plane_vec_vins_;
    return dist;
}

void vins2world::getInverse(Eigen::Matrix4d &T, Eigen::Matrix4d &T_inverse)
{
    T_inverse = Eigen::Matrix4d::Identity();
    Eigen::Matrix3d RotationMat = T.block<3, 3>(0, 0);
    Eigen::Vector3d translationVec = T.block<3, 1>(0, 3);
    T_inverse.block<3, 3>(0, 0) = RotationMat.transpose();
    T_inverse.block<3, 1>(0, 3) = -RotationMat.transpose() * translationVec;
}

void vins2world::calcT(Eigen::Matrix4d &T_CamWorld, Eigen::Matrix4d &T_CamVins)
{

    Eigen::Matrix4d T_VinsCam, T_WorldVins;
    getInverse(T_CamVins, T_VinsCam);
    T_VinsWorld = T_CamWorld*T_VinsCam;
    getInverse(T_VinsWorld, T_WorldVins);

    t_WorldVins = T_WorldVins.block<3, 1>(0, 3);
    R_WorldVins = T_WorldVins.block<3, 3>(0, 0);
    t_VinsWorld = T_VinsWorld.block<3, 1>(0, 3);
    R_VinsWorld = T_VinsWorld.block<3, 3>(0, 0);
    if(!received_first){
        t_VinsWorld_first_ = t_VinsWorld;
        R_VinsWorld_first_ = R_VinsWorld;
        received_first = true;
        T_vins2world_first_ = Eigen::Matrix4d::Identity();
        T_vins2world_first_.block<3,3>(0,0) = R_VinsWorld_first_;
        T_vins2world_first_.block<3,1>(0,3) = t_VinsWorld_first_;
    }
    calcParams();
}

void vins2world::calcParams()
{
    plane_vec_vins_ =  R_WorldVins * plane_vec_world_;
    plane_vec_vins_ = plane_vec_vins_/plane_vec_vins_.norm();
    //ax+by+cz+d=0
    Eigen::Vector3d world_plane_point(0.0,0.0,z_origin_plane_world_);
    Eigen::Vector3d world_point_in_vins_ = R_WorldVins * world_plane_point + t_WorldVins;
    d_vins_ = -((plane_vec_vins_.transpose() * world_point_in_vins_)(0,0));
    z_origin_plane_vins_ = -d_vins_/plane_vec_vins_(2);
    plane_origin_vins_ = Eigen::Vector3d(0.0,0.0,z_origin_plane_vins_);
}

void vins2world::setWorldGround(const double a, const double b, const double c, const double d)
{
    plane_vec_world_ = Eigen::Vector3d(a, b, c);
    plane_vec_world_ = plane_vec_world_/plane_vec_world_.norm();
    d_world_ = d;
    z_origin_plane_world_ = -d_world_/c;
}

void vins2world::setVinsGround(const double a, const double b, const double c, const double d){
    plane_vec_vins_(0) = a;
    plane_vec_vins_(1) = b;
    plane_vec_vins_(2) = c;
    plane_vec_vins_ = plane_vec_vins_/plane_vec_vins_.norm();
    double z = -d/c;
    plane_origin_vins_(0) = 0.0;
    plane_origin_vins_(1) = 0.0;
    plane_origin_vins_(2) = z;
}