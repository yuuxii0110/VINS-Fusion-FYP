#include <eigen3/Eigen/Dense>
#include <iostream>
class vins2world
{
    public:
        vins2world();
        void setWorldGround(const double a, const double b,const double c, const double d);
        void setVinsGround(const double a, const double b, const double c, const double d);
        void calcT(Eigen::Matrix4d &T_CamWorld, Eigen::Matrix4d &T_CamVins);
        void calcParams();
        double projectPoint(Eigen::Vector3d &v, Eigen::Vector3d &projected);

        Eigen::Matrix4d T_VinsWorld;
        Eigen::Vector3d plane_vec_vins_;
        Eigen::Vector3d plane_origin_vins_;
        double d_vins_;
        double z_origin_plane_vins_;
        Eigen::Vector3d t_VinsWorld_first_;
        Eigen::Matrix3d R_VinsWorld_first_;
        Eigen::Matrix4d T_vins2world_first_;
        bool received_first = false;
    private:
        Eigen::Vector3d t_VinsWorld, t_WorldVins;
        Eigen::Matrix3d R_VinsWorld, R_WorldVins;
        Eigen::Vector3d plane_vec_world_;
        double d_world_;
        double z_origin_plane_world_;

        void getInverse(Eigen::Matrix4d &T, Eigen::Matrix4d &T_inverse);
};
