#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include <fstream>

#include <libutils/timer.h>
#include <libutils/rasserts.h>
#include <libutils/thread_mutex.h>
#include <libutils/string_utils.h>

#include <phg/utils/point_cloud_export.h>
#include <phg/utils/cameras_bundler_import.h>
#include <phg/mvs/depth_maps/pm_depth_maps.h>
#include <phg/mvs/depth_maps/pm_geometry.h>
#include <phg/mvs/point_cloud_filtering/point_cloud_filtering_processor.h>

#include "utils/test_utils.h"

//________________________________________________________________________________
// Datasets:

// Скачайте и распакуйте архивы с картами глубины так чтобы в DATASET_DIR были папки depthmaps_downscaleN с .exr float32-картинками - картами глубины
// - saharov32  (downscales:     x4, x2, x1) - https://disk.yandex.com/d/2fWAdzpM4ibYBg
// - herzjesu25 (downscales: x8, x4, x2, x1) - https://disk.yandex.com/d/n3MyKUjvuVPF6Q

#define DATASET_DIR                  "saharov32"
#define DATASET_DOWNSCALE            4

//#define DATASET_DIR                  "herzjesu25"
//#define DATASET_DOWNSCALE            8
//________________________________________________________________________________

#define CAMERAS_LIMIT                5


void checkFloat32ImageReadWrite()
{
    // проверяем что OpenCV успешно пишет и читает float32 exr-файлы (OpenEXR)
    std::string test_image_path = "test_image_32f.exr";
    int rows = 10;
    int cols = 20;
    cv::Mat img32f(rows, cols, CV_32FC1);
    for (int j = 0; j < rows; ++j) {
        for (int i = 0; i < cols; ++i) {
            img32f.at<float>(j, i) = (j * cols + i) * 32.125f;
        }
    }
    cv::imwrite(test_image_path, img32f);

    cv::Mat copy = cv::imread(test_image_path, cv::IMREAD_UNCHANGED);
    if (copy.empty()) {
        throw std::runtime_error("Can't read float32 image: " + to_string(test_image_path));
    }
    rassert(copy.type() == CV_32FC1, 2381294810217);
    rassert(copy.cols == img32f.cols, 2371827410218);
    rassert(copy.rows == img32f.rows, 2371827410219);

    for (int j = 0; j < rows; ++j) {
        for (int i = 0; i < cols; ++i) {
            rassert(img32f.at<float>(j, i) == copy.at<float>(j, i), 2381924819223);
        }
    }
}

std::vector<cv::Mat> loadDepthMaps(const std::string &datasetDir, int downscale, const Dataset &dataset)
{
    checkFloat32ImageReadWrite();

    timer t;

    std::vector<cv::Mat> depth_maps(dataset.ncameras);

    for (int ci = 0; ci < dataset.cameras_labels.size(); ++ci) {
        std::string camera_label = dataset.cameras_labels[ci];
        // удаляем расширение картинки (.png, .jpg)
        std::string camera_label_without_extension = camera_label.substr(0, camera_label.find_last_of("."));

        std::string depth_map_path = std::string("data/src/datasets/") + DATASET_DIR + "/depthmaps_downscale" + to_string(downscale) + "/" + camera_label_without_extension + ".exr";
        depth_maps[ci] = cv::imread(depth_map_path, cv::IMREAD_UNCHANGED);

        if (depth_maps[ci].empty()) {
            throw std::runtime_error("Can't read depth map: " + to_string(depth_map_path)); // может быть вы забыли скачать и распаковать в правильную папку карты глубины? см. подробнее выше - "Datasets:"
        }

        rassert(depth_maps[ci].type() == CV_32FC1, 2381294810206);
        rassert(depth_maps[ci].cols == dataset.cameras_imgs[ci].cols, 2371827410207); // т.к. картинки мы тоже уменьшили в downscale раз
        rassert(depth_maps[ci].rows == dataset.cameras_imgs[ci].rows, 2371827410208);
    }

    std::cout << DATASET_DIR << " dataset: " << dataset.ncameras << " depth maps (x" << downscale << " downscale)" << " loaded in " << t.elapsed() << " s" << std::endl;

    return depth_maps;
}

TEST (test_point_cloud_filtering, SingleDepthMapFiltering) {
    // Этот тест берет отдельную карту глубины и фильтрует ее на базе несколько соседних
    // Результаты см. в папке data/debug/test_point_cloud_filtering/SingleFiltering
    // Интересно сравнить это новое облако (после фильтрации) со старым оригинальным облаком (по сырой карте глубины)
    Dataset dataset = loadDataset(DATASET_DIR, DATASET_DOWNSCALE);

    std::vector<cv::Mat> depth_maps = loadDepthMaps(DATASET_DIR, DATASET_DOWNSCALE, dataset);

    for (size_t ref_ci = 0; ref_ci < dataset.ncameras; ++ref_ci) {
        std::vector<vector3d> points;
        std::vector<double> radiuses;
        std::vector<cv::Vec3b> colors;
        std::vector<vector3d> normals;

        const cv::Mat &ref_depth_map = depth_maps[ref_ci];
        phg::buildPoints(ref_depth_map, dataset.cameras_imgs[ref_ci], dataset.cameras_P[ref_ci], dataset.calibration,
                         points, radiuses, normals, colors);

        std::string depth_map_points_path = std::string("data/debug/") + getTestSuiteName() + "/" + getTestName() + "/" + to_string(ref_ci) + "_depth_map.ply";
        phg::exportPointCloud(points, depth_map_points_path, colors, normals);
        std::cout << "Dense cloud built from raw depth map exported to " << depth_map_points_path << std::endl;
    }

    // Учитываем все камеры, перебираем какая карта глубины - центральная (т.е. сейчас фильтруется)
    size_t from_ci = 0;
    size_t to_ci = dataset.ncameras;
    for (size_t ref_ci = 0; ref_ci < dataset.ncameras; ++ref_ci) {
        std::cout << "Processing cameras " << dataset.cameras_labels[from_ci] << " - " << dataset.cameras_labels[to_ci - 1]
            << " (with " << dataset.cameras_labels[ref_ci] << " as reference camera)..." << std::endl;

        const cv::Mat &ref_depth_map = depth_maps[ref_ci];

        // Нам нужно репроецировать соседние камеры в нажу плоскость основного изображения
        // по сути это преподсчитанные карты глубины отвечающие на вопрос
        // "лежит ли на пути нашего пикселя-луча такая-то камера"
        std::vector<cv::Mat> ref_depth_map_neighbs_reprojs(to_ci - from_ci);

        const matrix34d ref_PtoLocal = dataset.cameras_P[ref_ci];
        const matrix34d ref_PtoWorld = phg::invP(ref_PtoLocal);
        std::vector<matrix34d> neighbs_PtoLocal(to_ci - from_ci);
        std::vector<matrix34d> neighbs_PtoWorld(to_ci - from_ci);
        for (size_t neighb_ci = from_ci; neighb_ci < to_ci; ++neighb_ci) {
            matrix34d neighb_PtoLocal = dataset.cameras_P[neighb_ci];
            matrix34d neighb_PtoWorld = phg::invP(neighb_PtoLocal);
            neighbs_PtoLocal[neighb_ci - from_ci] = neighb_PtoLocal;
            neighbs_PtoWorld[neighb_ci - from_ci] = neighb_PtoWorld;
        }

        // Пробегаем по всем соседним камерам
        for (size_t neighb_ci = from_ci; neighb_ci < to_ci; ++neighb_ci) {
            if (neighb_ci == ref_ci)
                continue; // сами себя репроецировать не будем
            const cv::Mat &neighb_depth_map = depth_maps[neighb_ci];
            cv::Mat ref_depth_map_neighb_reprojected = cv::Mat::zeros(ref_depth_map.rows, ref_depth_map.cols, CV_32FC1);

            // Пробегаем по всем пикселям соседней карты глубины (neighb_depth_map)
            // чтобы репроецировать эти точки в плоскость центральной камеры (ref_depth_map_neighb_reprojected)
            #pragma omp parallel for schedule(dynamic, 1)
            for (ptrdiff_t nj = 0; nj < neighb_depth_map.rows; ++nj) {
                for (ptrdiff_t ni = 0; ni < neighb_depth_map.cols; ++ni) {
                    float neighb_d = neighb_depth_map.at<float>(nj, ni);
                    if (neighb_d == 0.0f)
                        continue;
                    rassert(neighb_d > 0.0f, 23718397120164);
                    // TODO
                    vector3d global_p = phg::unproject(vector3d(ni + 0.5, nj + 0.5, neighb_d),
                                                       dataset.calibration, neighbs_PtoWorld[neighb_ci - from_ci]);
                    // TODO
                    vector3d ref_pixel = phg::project(global_p, dataset.calibration, ref_PtoLocal);
                    ptrdiff_t ref_i = ref_pixel[0];
                    ptrdiff_t ref_j = ref_pixel[1];
                    float ref_new_depth = ref_pixel[2]; // напоминаю что это не глубина, а проекция на оптическую ось Z
                    if (ref_new_depth < 0.0f)
                        continue; // TODO
                    if (ref_i < 0 || ref_i >= ref_depth_map.cols || ref_j < 0 || ref_j >= ref_depth_map.rows)
                        continue; // TODO

                    // смотрим на репроецированную (в плоскость центральной камеры) версию карты глубины соседа
                    // находим адрес пикселя в который наша точка спроецировалась
                    float* reprojected_depth_ptr = &ref_depth_map_neighb_reprojected.at<float>(ref_j, ref_i);
                    {
                        // нам нужно сделать синхронизацию через мьютекс чтобы если какой-то другой поток тоже
                        // спроецировал свою точку в этот пиксель - мы не затерли друг друга,
                        // вместо этого мы хотим чтобы победил кто-то один, причем детерминированно
                        // (обычная критическая секция нам не подходит т.к. будет слишком медленной т.к. потоки будут постоянно синхронизированны)
                        // (поэтому мы используем своеобразную HashMap-у из адресов памяти в мьютексы)
                        Lock lock(MutexPool::instance()->get(reprojected_depth_ptr));
                        float old_depth = *reprojected_depth_ptr;
                        if (old_depth == 0.0f) { // если пока что в этот пиксель никакая точка не была спроецирована - пишем себя
                            *reprojected_depth_ptr = ref_new_depth;
                        } else if (ref_new_depth < old_depth) { // TODO а кто должен быть записан в зависимости от величены?
                            *reprojected_depth_ptr = ref_new_depth;
                        }
                    }
                }
            }

            ref_depth_map_neighbs_reprojs[neighb_ci - from_ci] = ref_depth_map_neighb_reprojected;
        }

        cv::Mat ref_depth_map_filtered = cv::Mat::zeros(ref_depth_map.rows, ref_depth_map.cols, CV_32FC1);
        // Теперь когда ref_depth_map_neighbs_reprojs преподготовлены - т.е. мы легко можем узнавать кто лежал на нашем пути
        // мы можем посчитать функцию стабильности и отфильтровать глубины центральной камеры.
        // При этом нужно еще не забыть учесть в противовес а кого заслонили мы?
        #pragma omp parallel for schedule(dynamic, 1)
        for (ptrdiff_t ref_j = 0; ref_j < ref_depth_map.rows; ++ref_j) {
            for (ptrdiff_t ref_i = 0; ref_i < ref_depth_map.cols; ++ref_i) {
                float ref_d = ref_depth_map.at<float>(ref_j, ref_i);
                if (ref_d == 0.0f)
                    continue;

                std::vector<float> ray_depths;
                ray_depths.reserve(to_ci - from_ci + 1);
                ray_depths.push_back(ref_d);
                // Сохраним перечень кандидатов в ответ (все преподготовленные репроецированные карты глубины)
                for (size_t neighb_ci = from_ci; neighb_ci < to_ci; ++neighb_ci) {
                    if (neighb_ci == ref_ci)
                        continue;
                    float neighb_reprojected_depth = ref_depth_map_neighbs_reprojs[neighb_ci - from_ci].at<float>(ref_j, ref_i);
                    if (neighb_reprojected_depth == 0.0f)
                        continue;
                    ray_depths.push_back(neighb_reprojected_depth);
                }
                std::sort(ray_depths.begin(), ray_depths.end());

                for (size_t di = 0; di < ray_depths.size(); ++di) {
                    float cur_d = ray_depths[di];
                    int noccludes = di;

                    int nfree_space_violations = 0;
                    // Пробегаем по всем соседним камерам
                    for (size_t neighb_ci = from_ci; neighb_ci < to_ci; ++neighb_ci) {
                        if (neighb_ci == ref_ci)
                            continue; // сами себя репроецировать не будем
                        vector3d global_p = phg::unproject(vector3d(ref_i + 0.5, ref_j + 0.5, cur_d),
                                                           dataset.calibration, ref_PtoWorld);
                        vector3d neighb_pixel = phg::project(global_p, dataset.calibration, neighbs_PtoLocal[neighb_ci - from_ci]);

                        ptrdiff_t neighb_i = neighb_pixel[0];
                        ptrdiff_t neighb_j = neighb_pixel[1];
                        float neighb_ref_repro_depth = neighb_pixel[2]; // напоминаю что это не глубина, а проекция на оптическую ось Z
                        if (neighb_ref_repro_depth < 0.0f)
                            continue; // TODO
                        if (neighb_i < 0 || neighb_i >= depth_maps[neighb_ci].cols || neighb_j < 0 || neighb_j >= depth_maps[neighb_ci].rows)
                            continue; // TODO

                        float neighb_d = depth_maps[neighb_ci].at<float>(neighb_j, neighb_i);
                        if (neighb_d == 0.0f)
                            continue;
                        if (neighb_ref_repro_depth < neighb_d) { // TODO правильная ли это оценка диапазона погрешности проверки? как вам кажется?
                            ++nfree_space_violations;
                        }
                    }

                    int nstability = noccludes - nfree_space_violations;
                    if (nstability >= 0) {
                        ref_depth_map_filtered.at<float>(ref_j, ref_i) = cur_d;
                        break;
                    }
                }
            }
        }

        std::vector<vector3d> points;
        std::vector<double> radiuses;
        std::vector<cv::Vec3b> colors;
        std::vector<vector3d> normals;

        phg::buildPoints(ref_depth_map_filtered, dataset.cameras_imgs[ref_ci], dataset.cameras_P[ref_ci], dataset.calibration,
                         points, radiuses, normals, colors);

        std::string depth_map_points_path = std::string("data/debug/") + getTestSuiteName() + "/" + getTestName() + "/filtered_" + to_string(ref_ci) + "_depth_map.ply";
        phg::exportPointCloud(points, depth_map_points_path, colors, normals);
        std::cout << "Dense cloud built from filtered depth map exported to " << depth_map_points_path << std::endl;
    }
}

// TODO:
// 1) note that MS can't import resulting point cloud because PLY_BIN_BIGEND is not supported
// 2) find dataset with noisy depth maps (built wth SGM?) so that filtering led to satisfying results