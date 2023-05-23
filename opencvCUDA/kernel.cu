#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

#define BLOCKSIZE 16

__global__ void pca_covmatrix(unsigned char* input_image, float* cov_matrix, int width, int height)
{
    int col = (blockIdx.x * blockDim.x) + threadIdx.x;
    int row = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (col < width  && row < height )
    {
        // Adım 1: Giriş matrisini hesapla
        float input_matrix[3][3];
        int pixel_index = (row * width + col) * 3;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                input_matrix[i][j] = input_image[pixel_index + j];
            }
            pixel_index = pixel_index + (width * 3);
        }

        // Adım 2: Giriş Matrisin Her sütunun ortalamasını hesaplayın ve orijinal matristen çıkarın
        float column_mean[3] = { 0.0,0.0,0.0 };
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                column_mean[j] = column_mean[j] + input_matrix[i][j];
            }
        }
        for (int i = 0; i < 3; i++)
        {
            column_mean[i] /= 3.0;
        }
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                input_matrix[i][j] = input_matrix[i][j] - column_mean[j];
            }
        }

        float covariance_matrix[3][3] = { { 0.0, 0.0, 0.0 },
                                          { 0.0, 0.0, 0.0 },
                                          { 0.0, 0.0, 0.0 } };

        // Adım 3: Kovaryans matrisi oluşturma
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    covariance_matrix[i][j] = covariance_matrix[i][j] + (input_matrix[k][i] * input_matrix[k][j]);
                }
                covariance_matrix[i][j] /= 2.0;
            }
        }

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                cov_matrix[(row * width + col) * 9 + i * 3 + j] = covariance_matrix[i][j];
            }
        }
    }
}

int main()
{
    cv::VideoCapture cap(0);

    if (!cap.isOpened())
    {
        std::cerr << "Kamera açılamadı" << std::endl;
        return -1;
    }

    while (true)
    {
        cv::Mat frame;
        cap >> frame;

        int cols = frame.cols;
        int rows = frame.rows;

        cudaError_t err;
        unsigned char* pca_input_frame;
        float* cov_matrix;

        err = cudaMalloc(&pca_input_frame, cols * rows * sizeof(unsigned char) * 3);
        if (err != cudaSuccess)
        {
            std::cerr << "Hata: " << cudaGetErrorString(err) << std::endl;
            break;
        }

        err = cudaMalloc(&cov_matrix, (cols) * (rows) * sizeof(float) * 9);
        if (err != cudaSuccess)
        {
            std::cerr << "Hata: " << cudaGetErrorString(err) << std::endl;
            break;
        }

        dim3 threadsPerBlock(BLOCKSIZE, BLOCKSIZE);
        dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x, (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

        // Giriş görüntüsünü CUDA belleğine kopyala
        cudaMemcpy(pca_input_frame, frame.data, cols * rows * sizeof(unsigned char) * 3, cudaMemcpyHostToDevice);

        // PCA Covariance Matrix hesaplamasını gerçekleştir
        pca_covmatrix <<< blocksPerGrid, threadsPerBlock >> > (pca_input_frame, cov_matrix, cols, rows);
        cudaDeviceSynchronize();

        // Kovaryans matrisini CUDA belleğinden ana belleğe kopyala
        float* host_cov_matrix = new float[(cols) * (rows) * 9];
        cudaMemcpy(host_cov_matrix, cov_matrix, (cols) * (rows) * sizeof(float) * 9, cudaMemcpyDeviceToHost);

        std::cout << "Covariance Matrix:\n";
        for (int i = 0; i < 10 ; i++) { //rows
            for (int j = 0; j < 10 ; j++) { //cols
                float* cov_ptr = host_cov_matrix + ((i * (cols) + j) * 9);
                std::cout << "Pixel (" << i << ", " << j << "):\n";
                for (int k = 0; k < 9; k++) {
                    std::cout << cov_ptr[k] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }

        // Kovaryans matrisini kullanarak işlemler yapabilirsiniz

        // Bellekleri serbest bırak
        cudaFree(pca_input_frame);
        cudaFree(cov_matrix);
        delete[] host_cov_matrix;

        // Çıktıyı göster
        cv::imshow("Görüntü", frame);

        char k = cv::waitKey(0);
        if (k == 27) // ASCII kodu 27 => ESC tuşu
        {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
