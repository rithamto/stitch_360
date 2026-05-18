#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <regex>
#include <cmath>
#include <functional>
#include <numeric>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/stitching/detail/warpers.hpp>
#include <opencv2/stitching/warpers.hpp>
#include <filesystem>
#include <chrono>
#include <cstdlib>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;
using namespace cv;
using namespace cv::detail;
namespace fs = std::filesystem;

struct Metadata {
    int id;
    double y, p, r, f;
    string filename, path, row;
    int cluster_id;
};

struct MatchEdge {
    int ia, ib;
    vector<Point2f> pts_a, pts_b;
};

// ============ Rotation Matrix ============
inline Mat get_rotation_matrix_f(double y, double p, double r) {
    y *= M_PI / 180.0; p *= M_PI / 180.0; r *= M_PI / 180.0;
    Mat Ry = (Mat_<float>(3,3) << cos(y),0,sin(y), 0,1,0, -sin(y),0,cos(y));
    Mat Rp = (Mat_<float>(3,3) << 1,0,0, 0,cos(p),-sin(p), 0,sin(p),cos(p));
    Mat Rr = (Mat_<float>(3,3) << cos(r),-sin(r),0, sin(r),cos(r),0, 0,0,1);
    return Ry * Rp * Rr;
}

// ============ Warp Points (Spherical) ============
inline vector<Point2f> warp_points(const vector<Point2f>& pts, const Mat& K, const Mat& R, double focal) {
    if (pts.empty()) return {};
    float fx=K.at<float>(0,0), fy=K.at<float>(1,1), cx=K.at<float>(0,2), cy=K.at<float>(1,2);
    vector<Point2f> out(pts.size());
    for (size_t i = 0; i < pts.size(); i++) {
        float xn = (pts[i].x - cx)/fx, yn = (pts[i].y - cy)/fy;
        Mat ray = (Mat_<float>(3,1) << xn, yn, 1.0f);
        Mat rr = R * ray;
        float rx=rr.at<float>(0), ry_=rr.at<float>(1), rz=rr.at<float>(2);
        float norm = sqrt(rx*rx + ry_*ry_ + rz*rz) + 1e-6f;
        out[i].x = (float)(focal * atan2(rx, rz));
        out[i].y = (float)(focal * (asin(max(-1.0f, min(1.0f, ry_/norm))) + M_PI/2));
    }
    return out;
}

// ============ CSV Loader ============
inline map<string,Metadata> load_metadata_csv(const string& csv_path) {
    map<string,Metadata> meta;
    ifstream file(csv_path);
    if (!file.is_open()) return meta;
    string line; getline(file, line);
    vector<string> headers;
    stringstream ss(line); string col;
    while(getline(ss, col, ',')) headers.push_back(col);
    while (getline(file, line)) {
        stringstream ss2(line); string val; Metadata m; m.f=0; m.cluster_id=-1;
        int ci=0;
        while(getline(ss2, val, ',')) {
            if (ci<(int)headers.size()) {
                if (headers[ci]=="filename") m.filename=val;
                else if (headers[ci]=="yaw") m.y=stod(val);
                else if (headers[ci]=="pitch") m.p=stod(val);
                else if (headers[ci]=="roll") m.r=stod(val);
                else if (headers[ci]=="focal_length"||headers[ci]=="fx") m.f=stod(val);
                else if (headers[ci]=="cluster_id") m.cluster_id=stoi(val);
            }
            ci++;
        }
        meta[m.filename]=m;
    }
    return meta;
}

inline string get_row_name(double pitch) {
    if (pitch > 70) return "Zenith";
    if (pitch > 17.5) return "Top";
    if (pitch >= -17.5) return "Horizon";
    if (pitch >= -70) return "Bottom";
    return "Nadir";
}

// ============ SIFT Matching (ratio test 0.75) ============
inline bool get_sift_matches(const Mat& img1, const Mat& img2,
    vector<Point2f>& mkpts0, vector<Point2f>& mkpts1, int max_dim=1600) {
    Mat g1, g2;
    if (img1.channels()==3) cvtColor(img1,g1,COLOR_BGR2GRAY); else g1=img1;
    if (img2.channels()==3) cvtColor(img2,g2,COLOR_BGR2GRAY); else g2=img2;
    double s1 = min(1.0, (double)max_dim/max(g1.rows,g1.cols));
    double s2 = min(1.0, (double)max_dim/max(g2.rows,g2.cols));
    Mat sg1, sg2;
    resize(g1,sg1,Size(),s1,s1); resize(g2,sg2,Size(),s2,s2);
    auto sift = SIFT::create();
    vector<KeyPoint> kp1,kp2; Mat d1,d2;
    sift->detectAndCompute(sg1,noArray(),kp1,d1);
    sift->detectAndCompute(sg2,noArray(),kp2,d2);
    if (d1.empty()||d2.empty()||kp1.size()<2||kp2.size()<2) return false;
    BFMatcher bf;
    vector<vector<DMatch>> matches;
    bf.knnMatch(d1,d2,matches,2);
    for (auto& m : matches) {
        if (m.size()>=2 && m[0].distance < 0.75f*m[1].distance) {
            Point2f p1=kp1[m[0].queryIdx].pt, p2=kp2[m[0].trainIdx].pt;
            p1.x/=(float)s1; p1.y/=(float)s1;
            p2.x/=(float)s2; p2.y/=(float)s2;
            mkpts0.push_back(p1); mkpts1.push_back(p2);
        }
    }
    return mkpts0.size() > 8;
}

// ============ Levenberg-Marquardt Solver (soft_l1) ============
struct LMConfig { int max_nfev=600; double f_scale=1.0; };

inline vector<double> lm_solve(
    function<vector<double>(const vector<double>&)> resfn,
    const vector<double>& x0, const LMConfig& cfg)
{
    int n=(int)x0.size();
    vector<double> x=x0;
    auto r = resfn(x);
    int m=(int)r.size();
    int nfev=1;

    auto cost_fn = [&](const vector<double>& res) {
        double c=0;
        for (double v : res) { double z=v*v/(cfg.f_scale*cfg.f_scale); c+=2.0*(sqrt(1.0+z)-1.0)*cfg.f_scale*cfg.f_scale; }
        return c;
    };

    double cost = cost_fn(r);
    double lambda = 1e-3;
    cout << "[LM] Initial cost: " << cost << " (params=" << n << ", residuals=" << m << ")" << endl;

    for (int iter=0; iter<200 && nfev<cfg.max_nfev; iter++) {
        // Numerical Jacobian
        double eps=1e-7;
        Mat J(m,n,CV_64F);
        for (int j=0; j<n && nfev<cfg.max_nfev; j++) {
            vector<double> xp=x; xp[j]+=eps;
            auto rp=resfn(xp); nfev++;
            for (int i=0;i<m;i++) J.at<double>(i,j)=(rp[i]-r[i])/eps;
        }
        if (nfev>=cfg.max_nfev) break;

        // Robust weights
        Mat W(m,m,CV_64F,Scalar(0));
        for (int i=0;i<m;i++) {
            double z=r[i]*r[i]/(cfg.f_scale*cfg.f_scale);
            W.at<double>(i,i)=1.0/sqrt(1.0+z);
        }

        // Weighted J and r
        Mat Jw = W * J;
        Mat rv(m,1,CV_64F); for(int i=0;i<m;i++) rv.at<double>(i,0)=r[i];
        Mat rw = W * rv;

        Mat JtJ = Jw.t() * Jw;
        Mat Jtr = Jw.t() * rw;

        // Damping
        for (int i=0;i<n;i++) JtJ.at<double>(i,i) += max(1e-6, JtJ.at<double>(i,i)) * lambda;

        Mat dx;
        solve(JtJ, -Jtr, dx, DECOMP_CHOLESKY);

        vector<double> xn(n);
        for (int i=0;i<n;i++) xn[i]=x[i]+dx.at<double>(i,0);

        auto rn=resfn(xn); nfev++;
        double cn=cost_fn(rn);

        if (cn<cost) { x=xn; r=rn; cost=cn; lambda*=0.5; lambda=max(lambda,1e-10); }
        else { lambda*=2.0; lambda=min(lambda,1e10); }

        double dn=norm(dx);
        double xn_=0; for(auto v:x) xn_+=v*v; xn_=sqrt(xn_)+1e-10;
        if (dn/xn_<1e-8) break;
    }
    cout << "[LM] Final cost: " << cost << " (" << nfev << " evals)" << endl;
    return x;
}

// ============ solve_global_bundle ============
inline void solve_global_bundle(vector<Metadata>& frames, const vector<MatchEdge>& edges, const Mat& K_opt,
    vector<double>& out_y, vector<double>& out_p, vector<double>& out_r, double& out_f)
{
    int N=(int)frames.size();
    vector<double> iy(N),ip(N),ir(N);
    for (int i=0;i<N;i++){iy[i]=frames[i].y; ip[i]=frames[i].p; ir[i]=frames[i].r;}
    double init_f=K_opt.at<float>(0,0);
    float cx=K_opt.at<float>(0,2), cy=K_opt.at<float>(1,2);

    auto resfn = [&](const vector<double>& v) -> vector<double> {
        vector<double> res;
        double f=v[3*N];
        Mat K=(Mat_<float>(3,3)<<(float)f,0,cx, 0,(float)f,cy, 0,0,1);
        double circ=2*M_PI*f;
        for(int i=0;i<N;i++) res.push_back((v[i]-iy[i])*f*0.1);
        for(int i=0;i<N;i++) res.push_back((v[N+i]-ip[i])*f*5.0);
        for(int i=0;i<N;i++) res.push_back((v[2*N+i]-ir[i])*f*10.0);
        res.push_back((f-init_f)*0.5);
        res.push_back((v[0]-iy[0])*f*100.0);
        for (const auto& e : edges) {
            Mat Ra=get_rotation_matrix_f(v[e.ia],v[N+e.ia],v[2*N+e.ia]);
            Mat Rb=get_rotation_matrix_f(v[e.ib],v[N+e.ib],v[2*N+e.ib]);
            auto pa=warp_points(e.pts_a,K,Ra,f);
            auto pb=warp_points(e.pts_b,K,Rb,f);
            for(size_t i=0;i<pa.size();i++){
                double dx=pa[i].x-pb[i].x;
                if(dx>circ/2) dx-=circ; if(dx<-circ/2) dx+=circ;
                res.push_back(dx);
                res.push_back(pa[i].y-pb[i].y);
            }
        }
        return res;
    };

    vector<double> x0;
    for(int i=0;i<N;i++) x0.push_back(iy[i]);
    for(int i=0;i<N;i++) x0.push_back(ip[i]);
    for(int i=0;i<N;i++) x0.push_back(ir[i]);
    x0.push_back(init_f);

    LMConfig cfg; cfg.max_nfev=600; cfg.f_scale=1.0;
    auto sol=lm_solve(resfn,x0,cfg);

    out_y.assign(sol.begin(),sol.begin()+N);
    out_p.assign(sol.begin()+N,sol.begin()+2*N);
    out_r.assign(sol.begin()+2*N,sol.begin()+3*N);
    out_f=sol[3*N];
    cout<<"[INFO] Optimized Focal: "<<out_f<<" (Initial: "<<init_f<<")"<<endl;
}

// ============ Seam Finding ============
inline vector<UMat> find_seams(const vector<UMat>& imgs, const vector<UMat>& msks,
    const vector<Point>& corners, double scale=0.15)
{
    vector<UMat> result;
    for(auto& m:msks) result.push_back(m.clone());
    if(imgs.size()<=1) return result;
    vector<UMat> sm_imgs, sm_msks;
    vector<Point> sm_corners;
    for(size_t i=0;i<imgs.size();i++){
        UMat fi,si,mi;
        imgs[i].convertTo(fi,CV_32F);
        resize(fi,si,Size(),scale,scale);
        resize(msks[i],mi,Size(),scale,scale,INTER_NEAREST);
        sm_imgs.push_back(si); sm_msks.push_back(mi);
        sm_corners.push_back(Point(cvRound(corners[i].x*scale),cvRound(corners[i].y*scale)));
    }
    Ptr<SeamFinder> sf=makePtr<DpSeamFinder>(DpSeamFinder::COLOR);
    sf->find(sm_imgs,sm_corners,sm_msks);
    for(size_t i=0;i<imgs.size();i++)
        resize(sm_msks[i],result[i],msks[i].size(),0,0,INTER_NEAREST);
    return result;
}

// ============ Blend Images (matches Python blend_images) ============
inline pair<Mat,Mat> blend_images(const vector<UMat>& imgs, const vector<UMat>& msks,
    const vector<Point>& corners, int out_w, int out_h, float blend_strength=2.5f)
{
    float bw=sqrt((float)out_w*out_h)*blend_strength/100.0f;
    int nb=max(2,(int)ceil(log(bw)/log(2.0)));
    Ptr<MultiBandBlender> blender=makePtr<MultiBandBlender>(0,nb);
    blender->prepare(Rect(0,0,out_w,out_h));
    int fed=0;
    Mat kernel=Mat::ones(3,3,CV_8U);
    for(size_t i=0;i<imgs.size();i++){
        Mat img=imgs[i].getMat(ACCESS_READ);
        Mat m=msks[i].getMat(ACCESS_READ).clone();
        if(img.empty()||m.empty()) continue;
        dilate(m,m,kernel,Point(-1,-1),2);
        if(countNonZero(m)==0) continue;
        int xo=corners[i].x, yo=corners[i].y;
        int shifts[]={-out_w,0,out_w};
        for(int sh:shifts){
            int dx=xo+sh;
            if(dx+img.cols<=0||dx>=out_w) continue;
            int x0=max(0,dx),x1=min(out_w,dx+img.cols);
            int y0=max(0,yo),y1=min(out_h,yo+img.rows);
            if(x1>x0&&y1>y0){
                Mat si=img(Rect(x0-dx,y0-yo,x1-x0,y1-y0));
                Mat sm=m(Rect(x0-dx,y0-yo,x1-x0,y1-y0));
                if(si.empty()||sm.empty()||countNonZero(sm)==0) continue;
                Mat si16; si.convertTo(si16,CV_16S);
                blender->feed(si16,sm,Point(x0,y0));
                fed++;
            }
        }
    }
    Mat res,rmask;
    if(fed==0){ res=Mat::zeros(out_h,out_w,CV_8UC3); rmask=Mat::zeros(out_h,out_w,CV_8U); }
    else { blender->blend(res,rmask); convertScaleAbs(res,res); }
    return {res,rmask};
}

// ============ Crop Strip ============
inline void get_cropped_strip(const Mat& img, const Mat& msk, UMat& out_img, UMat& out_msk, Point& corner) {
    vector<Point> pts;
    findNonZero(msk,pts);
    if(pts.empty()){ img.copyTo(out_img); msk.copyTo(out_msk); corner=Point(0,0); return; }
    Rect bb=boundingRect(pts);
    img(bb).copyTo(out_img);
    msk(bb).copyTo(out_msk);
    corner=bb.tl();
}
