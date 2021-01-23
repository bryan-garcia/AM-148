#include "CImg.h"
#include "cudaimg.h"

int main() {

    CImg<unsigned char> ana_img("ana_sbk.bmp");

    unsigned char* blur1_data = NaiveGaussianBlur(ana_img);

    CImg<unsigned char> blur1_img(ana_img.width(), ana_img.height(), 1, 3);

    memcpy(blur1_img.data(), blur1_data, ana_img.size());

    blur1_img.save("blur_ana_sbk_naive.bmp");



    unsigned char* blur2_data = SharedGaussianBlur(ana_img);

    CImg<unsigned char> blur2_img(ana_img.width(), ana_img.height(), 1, 3);

    memcpy(blur2_img.data(), blur2_data, ana_img.size());

    blur2_img.save("blur_ana_sbk_shared.bmp");

    return 0;
}