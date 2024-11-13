__kernel void separate(__global uchar* fin, __global float* rchan, __global float* gchan,
                       __global float* bchan, int width, int height)
{
    int w = get_global_id(0);
    int h = get_global_id(1);
    uchar* pin = fin + (width * h * 3) + w * 3;
    rchan[width * h + w] = pin[2] / 255.0f;
    gchan[width * h + w] = pin[1] / 255.0f;
    bchan[width * h + w] = pin[0] / 255.0f;
}

__kernel void calc_ab(__global float* fin, __global float* aout, __global float* bout, int width,
                      int height, int r, float eps)
{
    int w = get_global_id(0);
    int h = get_global_id(1);

    float sum = 0.f;
    float sum_square = 0.f;
    int n = 0;
    for (int iw = w - r; iw <= w + r; ++iw) {
        for (int ih = h - r; ih <= h + r; ++ih) {
            if (iw >= 0 && ih >= 0 && iw < width && ih < height) {
                float v = fin[width * ih + iw];
                sum += v;
                sum_square += v * v;
                n += 1;
            }
        }
    }

    float fi = sum / n;
    float fii = sum_square / n;
    float cov = fii - fi * fi;
    int idx = width * h + w;
    float a = cov / (cov + eps);
    aout[idx] = a;
    bout[idx] = (1 - a) * fi;
}

__kernel void linear_conv(__global float* fin, __global float* a, __global float* b, int r,
                          __global uchar* fout, int width, int height, int color_idx,
                          float enhance_k, float complex_k, int output_mode)
{
    int w = get_global_id(0);
    int h = get_global_id(1);

    float sum_a = 0.f;
    float sum_b = 0.f;
    int n = 0;
    for (int iw = w - r; iw <= w + r; ++iw) {
        for (int ih = h - r; ih <= h + r; ++ih) {
            if (iw >= 0 && ih >= 0 && iw < width && ih < height) {
                sum_a += a[width * ih + iw];
                sum_b += b[width * ih + iw];
                n += 1;
            }
        }
    }

    float mean_a = sum_a / n;
    float mean_b = sum_b / n;
    int idx = width * h + w;
    float q = mean_a * fin[idx] + mean_b;
    float qo;
    if (output_mode == 0) {  // structure
        qo = enhance_k * (fin[idx] - q) + q;
    } else if (output_mode == 1) {  // texture
        qo = enhance_k * (fin[idx] - q) + fin[idx];
    } else {  // complex
        qo = enhance_k * (fin[idx] - q) + complex_k * q + (1 - complex_k) * fin[idx];
    }

    uchar* pout = fout + (width * h * 3) + w * 3;
    *(pout + color_idx) = clamp(round(qo * 255), 0.0f, 255.0f);
}
