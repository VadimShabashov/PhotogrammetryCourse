#pragma clang diagnostic push
#pragma ide diagnostic ignored "openmp-use-default-none"
#include "sift.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <libutils/rasserts.h>

// Ссылки:
// [lowe04] - Distinctive Image Features from Scale-Invariant Keypoints, David G. Lowe, 2004
//
// Примеры реализаций (стоит обращаться только если совсем не понятны какие-то места):
// 1) https://github.com/robwhess/opensift/blob/master/src/sift.c
// 2) https://gist.github.com/lxc-xx/7088609 (адаптация кода с первой ссылки)
// 3) https://github.com/opencv/opencv/blob/1834eed8098aa2c595f4d1099eeaa0992ce8b321/modules/features2d/src/sift.dispatch.cpp (адаптация кода с первой ссылки)
// 4) https://github.com/opencv/opencv/blob/1834eed8098aa2c595f4d1099eeaa0992ce8b321/modules/features2d/src/sift.simd.hpp (адаптация кода с первой ссылки)

#define DEBUG_ENABLE     1
#define DEBUG_PATH       std::string("data/debug/test_sift/debug/")

#define NOCTAVES                    3                    // число октав
#define OCTAVE_NLAYERS              3                    // в [lowe04] это число промежуточных степеней размытия картинки в рамках одной октавы обозначается - s, т.е. s слоев в каждой октаве
#define OCTAVE_GAUSSIAN_IMAGES      (OCTAVE_NLAYERS + 3)
#define OCTAVE_DOG_IMAGES           (OCTAVE_NLAYERS + 2)
#define INITIAL_IMG_SIGMA           0.75                 // предполагаемая степень размытия изначальной картинки
#define INPUT_IMG_PRE_BLUR_SIGMA    1.0                  // сглаживание изначальной картинки
   
#define SUBPIXEL_FITTING_ENABLE      0    // такие тумблеры включающие/выключающие очередное улучшение алгоритма позволяют оценить какой вклад эта фича вносит в качество результата если в рамках уже готового алгоритма попробовать ее включить/выключить

#define ORIENTATION_NHISTS           36   // число корзин при определении ориентации ключевой точки через гистограммы
#define ORIENTATION_WINDOW_R         3    // минимальный радиус окна в рамках которого будет выбрана ориентиация (в пикселях), R=3 => 5x5 окно
#define ORIENTATION_VOTES_PEAK_RATIO 0.80 // 0.8 => если гистограмма какого-то направления получила >= 80% от максимального чиссла голосов - она тоже победила

#define DESCRIPTOR_SIZE            4 // 4x4 гистограммы декскриптора
#define DESCRIPTOR_NBINS           8 // 8 корзин-направлений в каждой гистограмме дескриптора (4х4 гистограммы, каждая по 8 корзин, итого 4x4x8=128 значений в дескрипторе)
#define DESCRIPTOR_SAMPLES_N       4 // 4x4 замера для каждой гистограммы дескриптора (всего гистограмм 4х4) итого 16х16 замеров
#define DESCRIPTOR_SAMPLE_WINDOW_R 1.0 // минимальный радиус окна в рамках которого строится гистограмма из 8 корзин-направлений (т.е. для каждого из 16 элементов дескриптора), R=1 => 1x1 окно


void phg::SIFT::detectAndCompute(const cv::Mat &originalImg, std::vector<cv::KeyPoint> &kps, cv::Mat &desc) {
    // используйте дебаг в файлы как можно больше, это очень удобно и потраченное время окупается крайне сильно,
    // ведь пролистывать через окошки показывающие картинки долго, и по ним нельзя проматывать назад, а по файлам - можно
    // вы можете запустить алгоритм, сгенерировать десятки картинок со всеми промежуточными визуализациями и после запуска
    // посмотреть на те этапы к которым у вас вопросы или про которые у вас опасения
    if (DEBUG_ENABLE) cv::imwrite(DEBUG_PATH + "00_input.png", originalImg);

    cv::Mat img = originalImg.clone();
    // для удобства используем черно-белую картинку и работаем с вещественными числами (это еще и может улучшить точность)
    if (originalImg.type() == CV_8UC1) { // greyscale image
        img.convertTo(img, CV_32FC1, 1.0);
    } else if (originalImg.type() == CV_8UC3) { // BGR image
        img.convertTo(img, CV_32FC3, 1.0);
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    } else {
        rassert(false, 14291409120);
    }
    if (DEBUG_ENABLE) cv::imwrite(DEBUG_PATH + "01_grey.png", img);
    cv::GaussianBlur(img, img, cv::Size(0, 0), INPUT_IMG_PRE_BLUR_SIGMA, INPUT_IMG_PRE_BLUR_SIGMA);
    if (DEBUG_ENABLE) cv::imwrite(DEBUG_PATH + "02_grey_blurred.png", img);

    // Scale-space extrema detection
    std::vector<cv::Mat> gaussianPyramid;
    std::vector<cv::Mat> DoGPyramid;
    buildPyramids(img, gaussianPyramid, DoGPyramid);

    findLocalExtremasAndDescribe(gaussianPyramid, DoGPyramid, kps, desc);
}

void phg::SIFT::buildPyramids(const cv::Mat &imgOrg, std::vector<cv::Mat> &gaussianPyramid, std::vector<cv::Mat> &DoGPyramid) {
    gaussianPyramid.resize(NOCTAVES * OCTAVE_GAUSSIAN_IMAGES);

    const double k = pow(2.0, 1.0 / OCTAVE_NLAYERS); // [lowe04] k = 2^{1/s} а у нас s=OCTAVE_NLAYERS

    // строим пирамиду гауссовых размытий картинки
    for (size_t octave = 0; octave < NOCTAVES; ++octave) {
        // First image in octave
        if (octave == 0) {
            gaussianPyramid[0] = imgOrg.clone();
        } else {
            // берем картинку с предыдущей октавы и уменьшаем ее в два раза без какого бы то ни было дополнительного размытия
            // (сигмы должны совпадать)
            // тут есть очень важный момент, мы должны указать fx=0.5, fy=0.5 иначе при нечетном размере картинка будет
            // не идеально 2 пикселя в один схлопываться - а слегка смещаться
            cv::resize(
                gaussianPyramid[(octave - 1) * OCTAVE_GAUSSIAN_IMAGES + OCTAVE_NLAYERS],
                gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES],
                cv::Size(),
                0.5, 0.5, cv::INTER_NEAREST
            );
        }

        // Other images in octave
        #pragma omp parallel for
        for (ptrdiff_t layer = 1; layer < OCTAVE_GAUSSIAN_IMAGES; ++layer) {
            // если есть два последовательных гауссовых размытия с sigma1 и sigma2, то результат будет с sigma12=sqrt(sigma1^2 + sigma2^2) => sigma2=sqrt(sigma12^2-sigma1^2)
            double sigmaStartOctave = INITIAL_IMG_SIGMA * pow(2.0, octave);                     // sigma1 - сигма начала октавы
            double sigmaCur = sigmaStartOctave * pow(k, layer);                                 // sigma12 - сигма до которой мы хотим дойти на текущем слое
            double sigma = sqrt(sigmaCur * sigmaCur - sigmaStartOctave * sigmaStartOctave);        // sigma2 - сигма которую надо добавить чтобы довести sigma1 до sigma12

            cv::GaussianBlur(
                    gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES],
                    gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + layer],
                    cv::Size(),
                    sigma, sigma
            );
        }
    }

    for (size_t octave = 0; octave < NOCTAVES; ++octave) {
        for (size_t layer = 0; layer < OCTAVE_GAUSSIAN_IMAGES; ++layer) {
            double sigmaCur = INITIAL_IMG_SIGMA * pow(2.0, octave) * pow(k, layer);
            if (DEBUG_ENABLE) cv::imwrite(DEBUG_PATH + "pyramid/o" + to_string(octave) + "_l" + to_string(layer) + "_s" + to_string(sigmaCur) + ".png", gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + layer]);

            // Можно заметить, что i картинка из j октавы имеет такую же сигму, что и (i+s) картинка из (j-1) октавы.
            // (так получается, т.к. на самом деле есть нахлест; OCTAVE_GAUSSIAN_IMAGES больше, чем нужно)
            // Хочется сделать downsample и сравнить. Такая проверка не идеальна, т.к.
            // размытие и resize не коммутируют. Между некоторыми элементами выходит довольно приличное различие.
            // Но если брать среднее, то в целом неплохо.
            if (octave > 0 && layer + OCTAVE_NLAYERS < OCTAVE_GAUSSIAN_IMAGES) {
                // Current image
                cv::Mat curImg = gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + layer];

                // Image with the same sigma from previous octave after downsample
                cv::Mat sameSigmaImg;
                cv::resize(
                    gaussianPyramid[(octave - 1) * OCTAVE_GAUSSIAN_IMAGES + (layer + OCTAVE_NLAYERS)],
                    sameSigmaImg,
                    cv::Size(),
                    0.5, 0.5, cv::INTER_NEAREST
                );

                double diff = cv::mean(cv::abs(curImg - sameSigmaImg))[0];
                rassert(diff < 10.0, 19287317823);
            }
        }
    }

    DoGPyramid.resize(NOCTAVES * OCTAVE_DOG_IMAGES);

    // строим пирамиду разниц гауссиан слоев (Difference of Gaussian, DoG), т.к. вычитать надо из слоя слой в рамках
    // одной и той же октавы - то есть приятный параллелизм на уровне октав
    #pragma omp parallel for
    for (ptrdiff_t octave = 0; octave < NOCTAVES; ++octave) {
        for (size_t layer = 1; layer < OCTAVE_GAUSSIAN_IMAGES; ++layer) {
            int prevLayer = layer - 1;
            cv::Mat imgPrevGaussian = gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + prevLayer];
            cv::Mat imgCurGaussian  = gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + layer];

            cv::Mat imgCurDoG = imgCurGaussian.clone();
            // обратите внимание что т.к. пиксели картинки из одного ряда лежат в памяти подряд, поэтому если вложенный
            // цикл бежит подряд по одному и тому же ряду
            // то код работает быстрее т.к. он будет более cache-friendly, можете сравнить оценить ускорение добавив
            // замер времени построения пирамиды: timer t; double time_s = t.elapsed();
            for (size_t j = 0; j < imgCurDoG.rows; ++j) {
                for (size_t i = 0; i < imgCurDoG.cols; ++i) {
                    imgCurDoG.at<float>(j, i) = imgCurGaussian.at<float>(j, i) - imgPrevGaussian.at<float>(j, i);
                }
            }
            int dogLayer = layer - 1;
            DoGPyramid[octave * OCTAVE_DOG_IMAGES + dogLayer] = imgCurDoG;
        }
    }

    // нам нужны padding-картинки по краям октавы чтобы извлекать экстремумы,
    // но в статье предлагается не s+2 а s+3:
    // [lowe04] We must produce s + 3 images in the stack of blurred images for each octave, so that final extrema
    // detection covers a complete octave
    // Кажется, это нужно, потому что у нас s+1 картинка в октаве (sigma~k^i, i in [0,...,s])
    // Для паддинга нужны еще две картинки -> s+3 всего.

    for (size_t octave = 0; octave < NOCTAVES; ++octave) {
        for (size_t layer = 0; layer < OCTAVE_DOG_IMAGES; ++layer) {
            double sigmaCur = INITIAL_IMG_SIGMA * pow(2.0, octave) * pow(k, layer);
            if (DEBUG_ENABLE) cv::imwrite(DEBUG_PATH + "pyramidDoG/o" + to_string(octave) + "_l" + to_string(layer) + "_s" + to_string(sigmaCur) + ".png", DoGPyramid[octave * OCTAVE_DOG_IMAGES + layer]);

            // Для проверки можно сложить все DoG и последнюю гауссову. Должны получить исходное изображение.
            // Но погрешность будет большая, т.к. придется делать resize.
            // В целом можно было бы сложить для каждой октавы и сравнивать с начальной гауссовой из этой октавы.
        }
    }
}

namespace {
    std::pair<float, float> parabolaFitting(float x0, float x1, float x2) {
        rassert((x1 >= x0 && x1 >= x2) || (x1 <= x0 && x1 <= x2), 12541241241241);

        // a*0^2+b*0+c=x0
        // a*1^2+b*1+c=x1
        // a*2^2+b*2+c=x2

        // c=x0
        // a+b+x0=x1     (2)
        // 4*a+2*b+x0=x2 (3)

        // (3)-2*(2): 2*a-y0=y2-2*y1; a=(y2-2*y1+y0)/2
        // (2):       b=y1-y0-a
        float a = (x2 - 2.0f * x1 + x0) / 2.0f;
        float b = x1 - x0 - a;
        float c = x0;

        // extremum is at -b/(2*a), but our system coordinate start (i) is at 1, so minus 1
        float extremum = - b / (2.0f * a);
        float shift = extremum - 1.0f;

        //  Parabola value at extremum
        float extremum_val = a * (extremum * extremum) + b * extremum + c;

        return {shift, extremum_val - x1};
    }
}

void phg::SIFT::findLocalExtremasAndDescribe(const std::vector<cv::Mat> &gaussianPyramid, const std::vector<cv::Mat> &DoGPyramid,
                                             std::vector<cv::KeyPoint> &keyPoints, cv::Mat &desc) {
    std::vector<std::vector<float>> pointsDesc;

    // 3.1 Local extrema detection
    #pragma omp parallel // запустили каждый вычислительный поток процессора
    {
        // каждый поток будет складировать свои точки в свой личный вектор (чтобы не было гонок и не были нужны точки синхронизации)
        std::vector<cv::KeyPoint> thread_points;
        std::vector<std::vector<float>> thread_descriptors;

        for (size_t octave = 0; octave < NOCTAVES; ++octave) {
            double octave_downscale = pow(2.0, octave);
            for (size_t layer = 1; layer + 1 < OCTAVE_DOG_IMAGES; ++layer) {
                const cv::Mat prev = DoGPyramid[octave * OCTAVE_DOG_IMAGES + layer - 1];
                const cv::Mat cur  = DoGPyramid[octave * OCTAVE_DOG_IMAGES + layer];
                const cv::Mat next = DoGPyramid[octave * OCTAVE_DOG_IMAGES + layer + 1];
                const cv::Mat DoGs[3] = {prev, cur, next};

                // теперь каждый поток обработает свой кусок картинки 
                #pragma omp for
                for (int j = 1; j < cur.rows - 1; ++j) {
                    for (int i = 1; i + 1 < cur.cols; ++i) {
                        bool is_max = true;
                        bool is_min = true;
                        float center = DoGs[1].at<float>(j, i);
                        for (int dz = -1; dz <= 1 && (is_min || is_max); ++dz) {
                        for (int dy = -1; dy <= 1 && (is_min || is_max); ++dy) {
                        for (int dx = -1; dx <= 1 && (is_min || is_max); ++dx) {
                            if (dx != 0 || dy != 0 || dz != 0) {
                                if (DoGs[1 + dz].at<float>(j + dy, i + dx) >= center) {
                                    is_max = false;
                                }

                                if (DoGs[1 + dz].at<float>(j + dy, i + dx) <= center) {
                                    is_min = false;
                                }
                            }
                        }
                        }
                        }

                        bool is_extremum = (is_min || is_max);
                        if (!is_extremum) {
                            // очередной элемент cascade filtering, если не экстремум - сразу заканчиваем
                            // обработку этого пикселя
                            continue;
                        }

                        // 4 Accurate keypoint localization
                        cv::KeyPoint kp;

#if SUBPIXEL_FITTING_ENABLE
                        auto [dx, dvalue_x] = parabolaFitting(
                            DoGs[1].at<float>(j, i - 1),
                            center,
                            DoGs[1].at<float>(j, i + 1)
                        );

                        auto [dy, dvalue_y] = parabolaFitting(
                            DoGs[1].at<float>(j - 1, i),
                            center,
                            DoGs[1].at<float>(j + 1, i)
                        );

                        float dvalue = center + (dvalue_x + dvalue_y) / 2;
#else
                        float dx = 0.0f;
                        float dy = 0.0f;
                        float dvalue = 0.0f;
#endif

                        float contrast = center + dvalue;
                        if (contrast < contrast_threshold / OCTAVE_NLAYERS) {
                            // Порог контрастности должен уменьшаться, т.к. при увеличении числа слоев, в соседних слоях
                            // пятно будет регистрироваться все лучше. Значение бдет расти, поэтому разница между
                            // точками будет уменьшаться.
                            continue;
                        }

                        kp.pt = cv::Point2f((i + 0.5 + dx) * octave_downscale, (j + 0.5 + dy) * octave_downscale);

                        kp.response = fabs(contrast);

                        const double k = pow(2.0, 1.0 / OCTAVE_NLAYERS); // [lowe04] k = 2^{1/s} а у нас s=OCTAVE_NLAYERS
                        double sigmaCur = INITIAL_IMG_SIGMA * pow(2.0, octave) * pow(k, layer);
                        kp.size = 2.0 * sigmaCur * 5.0;

                        // 5 Orientation assignment
                        cv::Mat img = gaussianPyramid[octave * OCTAVE_GAUSSIAN_IMAGES + layer];
                        std::vector<float> votes;
                        float biggestVote;
                        int oriRadius = (int) (ORIENTATION_WINDOW_R * (1.0 + k * (layer - 1)));

                        if (!buildLocalOrientationHists(img, i, j, oriRadius, votes, biggestVote)) {
                            continue;
                        }

                        for (size_t bin = 0; bin < ORIENTATION_NHISTS; ++bin) {
                            float prevValue = votes[(bin + ORIENTATION_NHISTS - 1) % ORIENTATION_NHISTS];
                            float value = votes[bin];
                            float nextValue = votes[(bin + 1) % ORIENTATION_NHISTS];
                            if (value > prevValue && value > nextValue && votes[bin] > biggestVote * ORIENTATION_VOTES_PEAK_RATIO) {
                                kp.angle = (bin + 0.5) * (360.0 / ORIENTATION_NHISTS);
                                rassert(kp.angle >= 0.0 && kp.angle <= 360.0, 123512412412);
                                
                                std::vector<float> descriptor;
                                double descrSampleRadius = (DESCRIPTOR_SAMPLE_WINDOW_R * (1.0 + k * (layer - 1)));
                                if (!buildDescriptor(img, kp.pt.x, kp.pt.y, descrSampleRadius, kp.angle, descriptor))
                                    continue;

                                thread_points.push_back(kp);
                                thread_descriptors.push_back(descriptor);
                            }
                        }
                    }
                }
            }
        }

        // в критической секции объединяем все массивы детектированных точек
        #pragma omp critical
        {
            keyPoints.insert(keyPoints.end(), thread_points.begin(), thread_points.end());
            pointsDesc.insert(pointsDesc.end(), thread_descriptors.begin(), thread_descriptors.end());
        }
    }

    rassert(pointsDesc.size() == keyPoints.size(), 12356351235124);
    desc = cv::Mat(pointsDesc.size(), DESCRIPTOR_SIZE * DESCRIPTOR_SIZE * DESCRIPTOR_NBINS, CV_32FC1);
    for (size_t j = 0; j < pointsDesc.size(); ++j) {
        rassert(pointsDesc[j].size() == DESCRIPTOR_SIZE * DESCRIPTOR_SIZE * DESCRIPTOR_NBINS, 1253351412421);
        for (size_t i = 0; i < pointsDesc[i].size(); ++i) {
            desc.at<float>(j, i) = pointsDesc[j][i];
        }
    }
}

bool phg::SIFT::buildLocalOrientationHists(const cv::Mat &img, int i, int j, int radius,
                                           std::vector<float> &votes, float &biggestVote) {
    // 5 Orientation assignment
    votes.resize(ORIENTATION_NHISTS, 0.0f);
    biggestVote = 0.0;

    // Check that whole circle with center in i,j can fit into image
    if (i-1 < radius - 1 || i+1 + radius - 1 >= img.cols || j-1 < radius - 1 || j+1 + radius - 1 >= img.rows)
        return false;

    float sum[ORIENTATION_NHISTS] = {0.0f};

    for (int y = j - radius + 1; y < j + radius; ++y) {
        for (int x = i - radius + 1; x < i + radius; ++x) {
            // m(x, y)=(L(x + 1, y) − L(x − 1, y))^2 + (L(x, y + 1) − L(x, y − 1))^2
            double magnitude = pow(img.at<float>(y + 1, x) - img.at<float>(y - 1, x), 2.0) +
                               pow(img.at<float>(y, x + 1) - img.at<float>(y, x - 1), 2.0);

            // orientation == theta
            // atan( (L(x, y + 1) − L(x, y − 1)) / (L(x + 1, y) − L(x − 1, y)) )
            double orientation = atan2(
             img.at<float>(y + 1, x) - img.at<float>(y - 1, x),
             img.at<float>(y, x + 1) - img.at<float>(y, x - 1)
            );

            orientation = orientation * 180.0 / M_PI;
            orientation = (orientation + 90.0);
            if (orientation <  0.0)   orientation += 360.0;
            if (orientation >= 360.0) orientation -= 360.0;

            rassert(orientation >= 0.0 && orientation < 360.0, 5361615612);
            static_assert(360 % ORIENTATION_NHISTS == 0, "Inappropriate bins number!");

            auto bin = static_cast<size_t>(orientation / (360.0 / ORIENTATION_NHISTS));
            rassert(bin < ORIENTATION_NHISTS, 361236315613);

            sum[bin] += static_cast<float>(magnitude);
        }
    }

    for (size_t bin = 0; bin < ORIENTATION_NHISTS; ++bin) {
        votes[bin] = sum[bin];
        biggestVote = std::max(biggestVote, sum[bin]);
    }

    return true;
}

bool phg::SIFT::buildDescriptor(const cv::Mat &img, float px, float py, double descrSampleRadius, float angle,
                                std::vector<float> &descriptor) {
    cv::Mat relativeShiftRotation = cv::getRotationMatrix2D(cv::Point2f(0.0f, 0.0f), -angle, 1.0);

    const double smpW = 2.0 * descrSampleRadius - 1.0;

    descriptor.resize(DESCRIPTOR_SIZE * DESCRIPTOR_SIZE * DESCRIPTOR_NBINS, 0.0f);
    for (int hstj = 0; hstj < DESCRIPTOR_SIZE; ++hstj) { // перебираем строку в решетке гистограмм
        for (int hsti = 0; hsti < DESCRIPTOR_SIZE; ++hsti) { // перебираем колонку в решетке гистограмм

            float sum[DESCRIPTOR_NBINS] = {0.0f};

            for (int smpj = 0; smpj < DESCRIPTOR_SAMPLES_N; ++smpj) { // перебираем строчку замера для текущей гистограммы
                for (int smpi = 0; smpi < DESCRIPTOR_SAMPLES_N; ++smpi) { // перебираем столбик очередного замера для текущей гистограммы
                    for (int smpy = 0; smpy < smpW; ++smpy) { // перебираем ряд пикселей текущего замера
                        for (int smpx = 0; smpx < smpW; ++smpx) { // перебираем столбик пикселей текущего замера
                            cv::Point2f shift(
                                    ((-DESCRIPTOR_SIZE / 2.0 + hsti) * DESCRIPTOR_SAMPLES_N + smpi) * smpW,
                                    ((-DESCRIPTOR_SIZE / 2.0 + hstj) * DESCRIPTOR_SAMPLES_N + smpj) * smpW
                            );

                            std::vector<cv::Point2f> shiftInVector(1, shift);
                            cv::transform(shiftInVector, shiftInVector, relativeShiftRotation); // преобразуем относительный сдвиг с учетом ориентации ключевой точки
                            shift = shiftInVector[0];

                            int x = (int) (px + shift.x);
                            int y = (int) (py + shift.y);

                            if (y - 1 < 0 || y + 1 >= img.rows || x - 1 < 0 || x + 1 >= img.cols)
                                return false;

                            double magnitude = pow(img.at<float>(y + 1, x) - img.at<float>(y - 1, x), 2.0) +
                                               pow(img.at<float>(y, x + 1) - img.at<float>(y, x - 1), 2.0);

                            double orientation = atan2(
                                    img.at<float>(y + 1, x) - img.at<float>(y - 1, x),
                                    img.at<float>(y, x + 1) - img.at<float>(y, x - 1)
                            );

                            orientation = orientation * 180.0 / M_PI;
                            orientation = (orientation + 90.0);
                            if (orientation <  0.0)   orientation += 360.0;
                            if (orientation >= 360.0) orientation -= 360.0;

                            // Чтобы дескриптор был инвариантен к поворотам, необходимо градиенты повернуть в соответствии
                            // с углом, который нашли для ключевой точки. Но мы это уже сделали выше, когда вращали shift.
                            // Кажется, дополнительно ничего делать не надо.

                            rassert(orientation >= 0.0 && orientation < 360.0, 3515215125412);
                            static_assert(360 % DESCRIPTOR_NBINS == 0, "Inappropriate bins number!");

                            auto bin = static_cast<size_t>(orientation / (360.0 / DESCRIPTOR_NBINS));
                            rassert(bin < DESCRIPTOR_NBINS, 361236315613);

                            sum[bin] += static_cast<float>(magnitude);
                        }
                    }
                }
            }

            // Find histogram magnitude
            double hist_magnitude = 0;
            for (float val : sum) {
                hist_magnitude += val * val;
            }
            hist_magnitude = sqrt(hist_magnitude);

            float *votes = &(descriptor[(hstj * DESCRIPTOR_SIZE + hsti) * DESCRIPTOR_NBINS]); // нашли где будут лежать корзины нашей гистограммы
            for (int bin = 0; bin < DESCRIPTOR_NBINS; ++bin) {
                votes[bin] = sum[bin] / static_cast<float>(hist_magnitude);
            }
        }
    }

    return true;
}

#pragma clang diagnostic pop