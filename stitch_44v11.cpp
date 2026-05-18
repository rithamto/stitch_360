#include "stitch_helpers.h"

int main(int argc, char** argv) {
    auto t_start = chrono::high_resolution_clock::now();
    _putenv_s("OPENCV_OPENCL_DEVICE", "disabled");
    cv::ocl::setUseOpenCL(false);
    string input_dir = "c:/Project/drive-download-20260513T035900Z-3-001";
    int pano_width = 10000;
    float scale = 1.0f;
    if (argc > 1) input_dir = argv[1];
    if (argc > 2) pano_width = stoi(argv[2]);

    // 1. Load Frames & Metadata
    cout << "[INFO] Loading metadata from " << input_dir << "/metadata.csv" << endl;
    auto csv_meta = load_metadata_csv(input_dir + "/metadata.csv");
    vector<Metadata> frames;
    map<int, Mat> images_dict;

    regex id_regex("frame_(\\d+)_");

    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.path().extension() != ".jpg") continue;
        string fname = entry.path().filename().string();
        if (csv_meta.find(fname) == csv_meta.end()) continue;

        Metadata m = csv_meta[fname];
        m.path = entry.path().string();
        m.row = get_row_name(m.p);

        smatch match;
        if (regex_search(fname, match, id_regex)) m.id = stoi(match[1].str());
        else {
            regex simple_id("(\\d+)");
            if (regex_search(fname, match, simple_id)) m.id = stoi(match[1].str());
            else m.id = 0;
        }

        // Zenith/Nadir special handling
        if (m.row == "Zenith") m.cluster_id = 100;
        else if (m.row == "Nadir") m.cluster_id = 200;
        else m.cluster_id = -1;

        frames.push_back(m);
    }

    if (frames.empty()) { cerr << "[ERROR] No valid frames found!" << endl; return -1; }

    // 1b. Improved Clustering by Yaw Proximity
    vector<Metadata*> normal_frames;
    for (auto& m : frames) {
        if (m.cluster_id == -1) normal_frames.push_back(&m);
    }

    if (!normal_frames.empty()) {
        sort(normal_frames.begin(), normal_frames.end(),
             [](const Metadata* a, const Metadata* b){ return a->y < b->y; });
        int current_cid = 0;
        double last_yaw = normal_frames[0]->y;
        for (auto* m : normal_frames) {
            double diff = abs(m->y - last_yaw);
            if (diff > 180) diff = 360 - diff;
            if (diff > 15.0) current_cid++;
            m->cluster_id = current_cid;
            last_yaw = m->y;
        }
        cout << "[INFO] Grouped " << normal_frames.size() << " frames into "
             << current_cid + 1 << " vertical clusters based on yaw." << endl;
    }

    sort(frames.begin(), frames.end(), [](const Metadata& a, const Metadata& b){ return a.id < b.id; });

    // Pre-load images
    for (auto& m : frames) {
        Mat img = imread(m.path);
        if (scale != 1.0f) resize(img, img, Size(), scale, scale);
        images_dict[m.id] = img;
    }

    Mat img0 = images_dict[frames[0].id];
    int h0 = img0.rows, w0 = img0.cols;
    double focal = frames[0].f * scale;
    if (focal <= 0) focal = 2411.0;
    Mat K_opt = (Mat_<float>(3,3) << (float)focal, 0, w0/2.0f, 0, (float)focal, h0/2.0f, 0, 0, 1);

    // 2. Global Matching (geometry-aware, matching Python strategy)
    vector<MatchEdge> all_edges;

    auto match_and_add = [&](int idx1, int idx2) -> bool {
        if (idx1 == idx2) return false;
        vector<Point2f> pts_a, pts_b;
        bool ok = get_sift_matches(images_dict[frames[idx1].id], images_dict[frames[idx2].id], pts_a, pts_b);
        if (ok && pts_a.size() > 8) {
            all_edges.push_back({idx1, idx2, pts_a, pts_b});
            return true;
        }
        return false;
    };

    cout << "[INFO] Performing Global Matching..." << endl;

    // Build cluster map
    map<int, vector<int>> clusters;
    for (int i = 0; i < (int)frames.size(); i++)
        clusters[frames[i].cluster_id].push_back(i);

    // Intra-cluster (vertical neighbors by pitch)
    for (auto& [cid, indices] : clusters) {
        if (cid < 0) continue;
        sort(indices.begin(), indices.end(),
             [&](int a, int b){ return frames[a].p < frames[b].p; });
        for (int i = 0; i < (int)indices.size()-1; i++)
            match_and_add(indices[i], indices[i+1]);
    }

    // Inter-cluster (horizontal neighbors)
    vector<int> normal_cids;
    for (auto& [cid, _] : clusters)
        if (cid >= 0 && cid < 100) normal_cids.push_back(cid);
    sort(normal_cids.begin(), normal_cids.end(),
         [&](int a, int b){
             auto avg = [&](int c) {
                 double s=0; for(int i:clusters[c]) s+=frames[i].y;
                 return s/clusters[c].size();
             };
             return avg(a) < avg(b);
         });

    for (int i = 0; i < (int)normal_cids.size(); i++) {
        int c1 = normal_cids[i], c2 = normal_cids[(i+1) % normal_cids.size()];
        for (int i1 : clusters[c1])
            for (int i2 : clusters[c2])
                if (abs(frames[i1].p - frames[i2].p) < 20)
                    match_and_add(i1, i2);
    }

    // Zenith↔Top, Nadir↔Bottom
    vector<int> zen_idx, nad_idx, top_idx, bot_idx;
    for (int i = 0; i < (int)frames.size(); i++) {
        if (frames[i].row=="Zenith") zen_idx.push_back(i);
        else if (frames[i].row=="Nadir") nad_idx.push_back(i);
        else if (frames[i].row=="Top") top_idx.push_back(i);
        else if (frames[i].row=="Bottom") bot_idx.push_back(i);
    }
    for (int z : zen_idx) for (int t : top_idx) {
        double yd = abs(frames[z].y - frames[t].y);
        if (yd > 180) yd = 360 - yd;
        if (yd < 60) match_and_add(z, t);
    }
    for (int n : nad_idx) for (int b : bot_idx) {
        double yd = abs(frames[n].y - frames[b].y);
        if (yd > 180) yd = 360 - yd;
        if (yd < 60) match_and_add(n, b);
    }

    // 3. Global Bundle Adjustment
    cout << "[INFO] Global Bundle Adjustment with " << all_edges.size() << " match pairs..." << endl;
    vector<double> opt_y, opt_p, opt_r;
    double opt_f;
    solve_global_bundle(frames, all_edges, K_opt, opt_y, opt_p, opt_r, opt_f);

    for (int i = 0; i < (int)frames.size(); i++) {
        frames[i].y = opt_y[i]; frames[i].p = opt_p[i]; frames[i].r = opt_r[i];
        frames[i].f = opt_f;
    }

    // 4. Warping
    int out_w = pano_width, out_h = out_w / 2;
    float target_focal = out_w / (2.0f * (float)M_PI);
    cout << "[INFO] Warping frames to Spherical Panorama..." << endl;

    Ptr<WarperCreator> warper_creator = makePtr<cv::SphericalWarper>();
    Ptr<RotationWarper> warper = warper_creator->create(target_focal);

    // Store all warped data
    vector<UMat> all_warped_imgs, all_warped_msks;
    vector<Point> all_corners;
    map<int, vector<int>> cluster_img_indices;  // cluster_id -> indices into all_warped

    for (int i = 0; i < (int)frames.size(); i++) {
        double work_focal = frames[i].f * scale;
        double work_scale = target_focal / work_focal;
        int interp = work_scale < 1.0 ? INTER_AREA : INTER_LANCZOS4;

        Mat img_resized;
        resize(images_dict[frames[i].id], img_resized, Size(), work_scale, work_scale, interp);

        Mat_<float> K = Mat::eye(3, 3, CV_32F);
        K(0,0) = target_focal; K(1,1) = target_focal;
        K(0,2) = img_resized.cols / 2.0f;
        K(1,2) = img_resized.rows / 2.0f;

        Mat R = get_rotation_matrix_f(frames[i].y, frames[i].p, frames[i].r);

        UMat warped_img, mask_base, warped_mask;
        mask_base.create(img_resized.size(), CV_8U);
        mask_base.setTo(Scalar(255));

        Point corner = warper->warp(img_resized, K, R, INTER_LANCZOS4, BORDER_REFLECT, warped_img);
        warper->warp(mask_base, K, R, INTER_NEAREST, BORDER_CONSTANT, warped_mask);

        // Trim Zenith/Nadir masks
        if (frames[i].row == "Zenith") {
            int gmy = (int)(out_h * 0.35);
            int lmy = gmy - corner.y;
            if (lmy > 0 && lmy < warped_mask.rows)
                warped_mask(Rect(0, lmy, warped_mask.cols, warped_mask.rows - lmy)).setTo(Scalar(0));
        } else if (frames[i].row == "Nadir") {
            int gminy = (int)(out_h * 0.65);
            int lminy = gminy - corner.y;
            if (lminy > 0 && lminy < warped_mask.rows)
                warped_mask(Rect(0, 0, warped_mask.cols, lminy)).setTo(Scalar(0));
        }

        int idx = (int)all_warped_imgs.size();
        all_warped_imgs.push_back(warped_img);
        all_warped_msks.push_back(warped_mask);
        all_corners.push_back(corner);
        cluster_img_indices[frames[i].cluster_id].push_back(idx);

        cout << "Warped image " << i+1 << "/" << frames.size() << endl;
    }

    // 5. Global Exposure Compensation
    cout << "[INFO] Applying Global Exposure Compensation..." << endl;
    {
        double es = 0.1;
        vector<UMat> is2(all_warped_imgs.size()), ms2(all_warped_imgs.size());
        vector<Point> cs2(all_warped_imgs.size());
        for (size_t i = 0; i < all_warped_imgs.size(); i++) {
            resize(all_warped_imgs[i], is2[i], Size(), es, es);
            resize(all_warped_msks[i], ms2[i], Size(), es, es, INTER_NEAREST);
            cs2[i] = Point(cvRound(all_corners[i].x*es), cvRound(all_corners[i].y*es));
        }
        auto comp = ExposureCompensator::createDefault(ExposureCompensator::GAIN_BLOCKS);
        comp->feed(cs2, is2, ms2);
        for (size_t i = 0; i < all_warped_imgs.size(); i++)
            comp->apply((int)i, all_corners[i], all_warped_imgs[i], all_warped_msks[i]);
    }

    // 6. Hierarchical Seam Finding & Smooth Proxy Generation
    cout << "[INFO] Generating smooth cluster proxies and finding final seams..." << endl;
    vector<int> sorted_cids;
    for (auto& [cid, _] : cluster_img_indices) sorted_cids.push_back(cid);
    sort(sorted_cids.begin(), sorted_cids.end());

    vector<UMat> cluster_proxies, cluster_proxy_masks;
    vector<Point> cluster_proxy_corners;

    vector<UMat> final_imgs, final_msks;
    vector<Point> final_corners;
    vector<int> final_img_cluster_map;

    // Step 6a: Intra-cluster seams and smooth proxy
    for (int ci = 0; ci < (int)sorted_cids.size(); ci++) {
        int cid = sorted_cids[ci];
        auto& indices = cluster_img_indices[cid];
        cout << "[INFO] Processing cluster: " << cid << " (" << indices.size() << " images)..." << endl;

        vector<UMat> c_imgs, c_msks;
        vector<Point> c_corners;
        for (int idx : indices) {
            c_imgs.push_back(all_warped_imgs[idx]);
            c_msks.push_back(all_warped_msks[idx]);
            c_corners.push_back(all_corners[idx]);
        }

        vector<UMat> seam_msks;
        if (c_imgs.size() > 1)
            seam_msks = find_seams(c_imgs, c_msks, c_corners, 0.15);
        else
            for (auto& m : c_msks) seam_msks.push_back(m.clone());

        for (int i = 0; i < (int)c_imgs.size(); i++) {
            final_imgs.push_back(c_imgs[i]);
            final_msks.push_back(seam_msks[i]);
            final_corners.push_back(c_corners[i]);
            final_img_cluster_map.push_back(ci);
        }

        // Blend cluster into smooth proxy
        auto [strip_img, strip_msk] = blend_images(c_imgs, seam_msks, c_corners, out_w, out_h, 1.5f);

        UMat c_img, c_msk;
        Point c_corner;
        get_cropped_strip(strip_img, strip_msk, c_img, c_msk, c_corner);
        cluster_proxies.push_back(c_img);
        cluster_proxy_masks.push_back(c_msk);
        cluster_proxy_corners.push_back(c_corner);
    }

    // Step 6b: Inter-cluster seams using smooth proxies
    cout << "[INFO] Finding seams between smooth cluster proxies..." << endl;
    vector<UMat> final_cluster_masks = find_seams(cluster_proxies, cluster_proxy_masks,
                                                   cluster_proxy_corners, 0.15);

    // Step 6c: Project inter-cluster seams back
    cout << "[INFO] Projecting seams back for single-pass blending..." << endl;
    int num_clusters = (int)sorted_cids.size();

    vector<Mat> global_cluster_masks(num_clusters);
    for (int ci = 0; ci < num_clusters; ci++) {
        global_cluster_masks[ci] = Mat::zeros(out_h, out_w, CV_8U);
        Mat c_mask = final_cluster_masks[ci].getMat(ACCESS_READ);
        int cx_ = cluster_proxy_corners[ci].x, cy_ = cluster_proxy_corners[ci].y;
        int shifts[] = {-out_w, 0, out_w};
        for (int sh : shifts) {
            int dx = cx_ + sh;
            if (dx + c_mask.cols <= 0 || dx >= out_w) continue;
            int x0=max(0,dx), x1=min(out_w,dx+c_mask.cols);
            int y0=max(0,cy_), y1=min(out_h,cy_+c_mask.rows);
            if (x1>x0 && y1>y0) {
                Mat sub = c_mask(Rect(x0-dx, y0-cy_, x1-x0, y1-y0));
                Mat roi = global_cluster_masks[ci](Rect(x0, y0, x1-x0, y1-y0));
                bitwise_or(roi, sub, roi);
            }
        }
    }

    Mat dil_kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    for (int idx = 0; idx < (int)final_imgs.size(); idx++) {
        int ci = final_img_cluster_map[idx];
        Mat dilated;
        dilate(global_cluster_masks[ci], dilated, dil_kernel, Point(-1,-1), 1);

        Mat m = final_msks[idx].getMat(ACCESS_READ);
        int xo = final_corners[idx].x, yo = final_corners[idx].y;
        Mat new_m = Mat::zeros(m.size(), CV_8U);

        int shifts[] = {-out_w, 0, out_w};
        for (int sh : shifts) {
            int dx = xo + sh;
            if (dx + m.cols <= 0 || dx >= out_w) continue;
            int x0=max(0,dx), x1=min(out_w,dx+m.cols);
            int y0=max(0,yo), y1=min(out_h,yo+m.rows);
            if (x1>x0 && y1>y0) {
                Rect sr(x0-dx, y0-yo, x1-x0, y1-y0);
                Rect dr(x0, y0, x1-x0, y1-y0);
                Mat sub_m = m(sr), sub_c = dilated(dr);
                Mat valid; bitwise_and(sub_m, sub_c, valid);
                Mat nm = new_m(sr);
                bitwise_or(nm, valid, nm);
            }
        }
        new_m.copyTo(final_msks[idx]);
    }

    // 7. Final Blending (Single Pass) - blend_strength=2.0 matching Python
    cout << "[INFO] Final single-pass blending of original pixels..." << endl;
    auto [final_pano, _] = blend_images(final_imgs, final_msks, final_corners, out_w, out_h, 2.0f);

    // 8. Save
    cout << "[INFO] Writing output..." << endl;
    fs::path input_path(input_dir);
    string out_path = "panorama_360_v11_cpp_" + input_path.filename().string() + ".jpg";
    vector<int> params = {IMWRITE_JPEG_QUALITY, 95};
    imwrite(out_path, final_pano, params);

    cout << "[INFO] Done! Saved to " << out_path << endl;
    auto t_end = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double>(t_end - t_start).count();
    cout << "[INFO] Total Execution Time: " << elapsed << " seconds." << endl;
    return 0;
}
