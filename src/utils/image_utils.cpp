#include "image_utils.h"

#include <algorithm>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

ncnn::Mat load_image_to_ncnn_mat(const std::string& image_path) {
    int w, h, c;
    unsigned char* data = stbi_load(image_path.c_str(), &w, &h, &c, 3);
    if (!data) {
        return ncnn::Mat();
    }

    ncnn::Mat bgr(w, h, 3, (size_t)1u);
    unsigned char* dst = (unsigned char*)bgr.data;
    
    for (int i = 0; i < w * h; i++) {
        dst[i * 3 + 0] = data[i * 3 + 2];
        dst[i * 3 + 1] = data[i * 3 + 1];
        dst[i * 3 + 2] = data[i * 3 + 0];
    }

    stbi_image_free(data);
    return bgr;
}

ncnn::Mat ncnn_mat_resize(const ncnn::Mat& src, int target_w, int target_h) {
    if (ncnn_mat_empty(src)) {
        return ncnn::Mat();
    }

    ncnn::Mat dst(target_w, target_h, 3, (size_t)1u);
    
    ncnn::resize_bilinear_c3((const unsigned char*)src.data, src.w, src.h, src.w * 3,
                              (unsigned char*)dst.data, target_w, target_h, target_w * 3);

    return dst;
}

ncnn::Mat ncnn_mat_resize_bicubic(const ncnn::Mat& src, int target_w, int target_h) {
    if (ncnn_mat_empty(src)) {
        return ncnn::Mat();
    }

    // ncnn::resize_bicubic works on planar float Mats. Convert the interleaved
    // BGR u8 buffer to planar BGR float, resize, then pack back to interleaved u8.
    ncnn::Mat src_planar(src.w, src.h, 3);
    for (int y = 0; y < src.h; y++) {
        const unsigned char* row = (const unsigned char*)src.data + (size_t)y * src.w * 3;
        float* b = src_planar.channel(0).row(y);
        float* g = src_planar.channel(1).row(y);
        float* r = src_planar.channel(2).row(y);
        for (int x = 0; x < src.w; x++) {
            b[x] = (float)row[x * 3 + 0];
            g[x] = (float)row[x * 3 + 1];
            r[x] = (float)row[x * 3 + 2];
        }
    }

    ncnn::Mat dst_planar(target_w, target_h, 3);
    ncnn::resize_bicubic(src_planar, dst_planar, target_w, target_h);

    ncnn::Mat dst(target_w, target_h, 3, (size_t)1u);
    unsigned char* dst_data = (unsigned char*)dst.data;
    for (int y = 0; y < target_h; y++) {
        const float* b = dst_planar.channel(0).row(y);
        const float* g = dst_planar.channel(1).row(y);
        const float* r = dst_planar.channel(2).row(y);
        unsigned char* dst_row = dst_data + (size_t)y * target_w * 3;
        for (int x = 0; x < target_w; x++) {
            dst_row[x * 3 + 0] = (unsigned char)std::min(255.0f, std::max(0.0f, b[x]));
            dst_row[x * 3 + 1] = (unsigned char)std::min(255.0f, std::max(0.0f, g[x]));
            dst_row[x * 3 + 2] = (unsigned char)std::min(255.0f, std::max(0.0f, r[x]));
        }
    }

    return dst;
}

bool ncnn_mat_empty(const ncnn::Mat& mat) {
    return mat.empty() || mat.w <= 0 || mat.h <= 0;
}

#include <limits>
#include <stdexcept>
namespace ncnn_llm { namespace minicpm {
namespace {
void require(bool ok, const char* reason) { if (!ok) throw std::invalid_argument(reason); }
int divide_even(int n, int divisor) {
    // Python round, independent of C++ floating-point rounding mode.
    int q = n / divisor, r = n % divisor;
    if (2LL*r > divisor || (2LL*r == divisor && q % 2)) ++q;
    require(static_cast<int64_t>(q)*divisor <= std::numeric_limits<int>::max(), "resize dimension overflow");
    return std::max(q * divisor, divisor);
}
}
int merge_factor(const std::string& mode) {
    require(mode == "16x" || mode == "4x", "MiniCPM downsample mode must be 16x or 4x");
    return mode == "16x" ? 4 : 2;
}
Thumbnail plan_thumbnail(int h, int w, int p, int scale, const std::string& mode,
                         bool slice_mode, int max_slices) {
    const int merge = merge_factor(mode);
    require(h > 0 && w > 0 && p > 0 && p <= std::numeric_limits<int>::max()/4 && scale > 0,
            "invalid thumbnail dimensions");
    require(max_slices >= 1, "max_slices must be positive");
    const double ratio = static_cast<double>(h)*w / (static_cast<double>(scale)*scale);
    require(!slice_mode || max_slices == 1 || ratio <= 1.0,
            "MiniCPM detail slices are unsupported; explicitly disable slicing or use max_slices=1");
    // No slices => allow_upscale=True, including small images. HF truncates
    // height before calculating width, then rounds both to patch_size*4.
    const double aspect = static_cast<double>(w)/h;
    const double resized_h = scale/std::sqrt(aspect);
    require(resized_h < std::numeric_limits<int>::max(), "resize height overflow");
    h = static_cast<int>(resized_h);
    require(h*aspect < std::numeric_limits<int>::max(), "resize width overflow");
    w = static_cast<int>(h*aspect);
    h = divide_even(h, p*4); w = divide_even(w, p*4);
    return {h, w, h/p, w/p, h/p/merge, w/p/merge};
}
std::vector<float> pack_rgb(const std::vector<uint8_t>& rgb, int h, int w, int p) {
    require(h>0 && w>0 && p>0 && h%p==0 && w%p==0, "invalid NaViT patch dimensions");
    require(static_cast<uint64_t>(h)*w*3 == rgb.size(), "RGB buffer size mismatch");
    std::vector<float> out(rgb.size());
    size_t dst=0;
    for (int c=0;c<3;++c) for (int py=0;py<p;++py)
        for (int gy=0;gy<h/p;++gy) for (int gx=0;gx<w/p;++gx) for (int px=0;px<p;++px) {
            const size_t src=(static_cast<size_t>(gy*p+py)*w+gx*p+px)*3+c;
            out[dst++]=(static_cast<float>(rgb[src])/255.f-0.5f)/0.5f;
        }
    return out;
}
std::vector<int> nearest_position_ids(int h, int w, int side) {
    require(h>0 && w>0 && side>0 && static_cast<int64_t>(side)*side<=std::numeric_limits<int>::max(),
            "invalid learned position grid");
    // torch's default float32 arange boundaries, bucketize(...,right=True).
    const double step=1.0/side;
    std::vector<float> boundaries;
    for (int i=0;i<side-1;++i) boundaries.push_back(static_cast<float>(step+i*step));
    std::vector<int> result;
    for (int y=0;y<h;++y) for (int x=0;x<w;++x) {
        int by=std::upper_bound(boundaries.begin(),boundaries.end(),static_cast<float>(y)/h)-boundaries.begin();
        int bx=std::upper_bound(boundaries.begin(),boundaries.end(),static_cast<float>(x)/w)-boundaries.begin();
        result.push_back(by*side+bx);
    }
    return result;
}
} }

// Copyright (c) 2026 ncnn_llm authors. Apache-2.0.
// Fixed-point conventions adapted from Pillow 12.3.0 Resample.c:
// https://github.com/python-pillow/Pillow/blob/12.3.0/src/libImaging/Resample.c
/*
The Python Imaging Library (PIL) is
    Copyright (c) 1997-2011 by Secret Labs AB
    Copyright (c) 1995-2011 by Fredrik Lundh and contributors
Pillow is the friendly PIL fork. It is
    Copyright (c) 2010 by Jeffrey 'Alex' Clark and contributors
Like PIL, Pillow is licensed under the open source MIT-CMU License:
By obtaining, using, and/or copying this software and/or its associated
documentation, you agree that you have read, understood, and will comply
with the following terms and conditions:
Permission to use, copy, modify and distribute this software and its
documentation for any purpose and without fee is hereby granted,
provided that the above copyright notice appears in all copies, and that
both that copyright notice and this permission notice appear in supporting
documentation, and that the name of Secret Labs AB or the author not be
used in advertising or publicity pertaining to distribution of the software
without specific, written prior permission.
SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR ANY SPECIAL,
INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
*/
#include "image_utils.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ncnn_llm { namespace minicpm { namespace {
constexpr int kPrecision=22;
struct Taps { int first; std::vector<int32_t> weight; };
size_t buffer_size(int h,int w) {
    if(h<=0 || w<=0 || static_cast<uint64_t>(h)*w>std::numeric_limits<size_t>::max()/3)
        throw std::invalid_argument("invalid bicubic image dimensions");
    return static_cast<size_t>(h)*w*3;
}
double cubic(double x) {
    x=std::abs(x);
    if(x<1.0) return (1.5*x-2.5)*x*x+1.0;
    if(x<2.0) return (((x-5.0)*x+8.0)*x-4.0)*-0.5;
    return 0.0;
}
std::vector<Taps> coefficients(int source,int target) {
    const double scale=static_cast<double>(source)/target;
    const double stretch=std::max(scale,1.0), support=2.0*stretch;
    std::vector<Taps> result(target);
    for(int dst=0;dst<target;++dst) {
        const double center=(dst+0.5)*scale;
        const int begin=static_cast<int>(std::max(0.0,center-support+0.5));
        const int end=static_cast<int>(std::min(static_cast<double>(source),center+support+0.5));
        auto& taps=result[dst]; taps.first=begin;
        std::vector<double> weight(end-begin);
        double sum=0;
        for(int s=begin;s<end;++s) {
            const double v=cubic((s-center+0.5)*(1.0/stretch));
            weight[s-begin]=v; sum+=v;
        }
        taps.weight.reserve(weight.size());
        for(double value:weight) {
            if(sum!=0.0) value/=sum;
            const double fixed=value*(1<<kPrecision);
            taps.weight.push_back(static_cast<int32_t>(fixed+(value<0?-0.5:0.5)));
        }
    }
    return result;
}
uint8_t finish(int64_t value) {
    // Rounding bias already accumulated. Avoid signed negative right shift.
    if(value<=0) return 0;
    return static_cast<uint8_t>(std::min<int64_t>(255,value/(1<<kPrecision)));
}
} // namespace
std::vector<uint8_t> resize_bicubic_rgb(const std::vector<uint8_t>& pixels,
    int h,int w,int th,int tw,bool bgr) {
    if(pixels.size()!=buffer_size(h,w)) throw std::invalid_argument("bicubic input size mismatch");
    buffer_size(th,tw);
    std::vector<uint8_t> current=pixels;
    if(bgr) for(size_t i=0;i<current.size();i+=3) std::swap(current[i],current[i+2]);
    // Pillow 12.3.0 quantizes/clips after each pass, horizontal first.
    if(w!=tw) {
        const auto taps=coefficients(w,tw);
        std::vector<uint8_t> next(buffer_size(h,tw));
        for(int y=0;y<h;++y) for(int x=0;x<tw;++x) for(int c=0;c<3;++c) {
            int64_t value=1<<(kPrecision-1);
            for(size_t k=0;k<taps[x].weight.size();++k)
                value+=static_cast<int64_t>(current[(static_cast<size_t>(y)*w+taps[x].first+k)*3+c])*taps[x].weight[k];
            next[(static_cast<size_t>(y)*tw+x)*3+c]=finish(value);
        }
        current.swap(next);
    }
    if(h!=th) {
        const auto taps=coefficients(h,th);
        std::vector<uint8_t> next(buffer_size(th,tw));
        for(int y=0;y<th;++y) for(int x=0;x<tw;++x) for(int c=0;c<3;++c) {
            int64_t value=1<<(kPrecision-1);
            for(size_t k=0;k<taps[y].weight.size();++k)
                value+=static_cast<int64_t>(current[((taps[y].first+k)*tw+x)*3+c])*taps[y].weight[k];
            next[(static_cast<size_t>(y)*tw+x)*3+c]=finish(value);
        }
        current.swap(next);
    }
    return current;
}
} }
