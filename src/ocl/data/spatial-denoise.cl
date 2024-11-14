sampler_t const g_sa_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE;
sampler_t const g_sa_clamp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP;

__kernel void copy_first_frame(__global uchar* fin, __global uchar* fout,
                               __write_only image2d_t last_rgb, int width, int height)
{
    int w = get_global_id(0);
    int h = get_global_id(1);

    uchar3 cur_bgr = vload3(0, fin + (width * h * 3) + w * 3);
    vstore3(cur_bgr, 0, fout + (width * h * 3) + w * 3);
    write_imageui(last_rgb, (int2)(w, h), (uint4)(cur_bgr.x, cur_bgr.y, cur_bgr.z, 0));
}

__kernel void copy_comp_edge(__read_only image2d_t imgI, __write_only image2d_t imgComp, int width,
                             int height, int offset_y)
{
    int w = get_global_id(0);
    int h = get_global_id(1) + offset_y;

    uint4 data = read_imageui(imgI, g_sa_none, (int2)(w, h));
    write_imageui(imgComp, (int2)(w, h), data);
}

__kernel void motionEstTSS(__global uchar* imgP, __read_only image2d_t imgI, int mbSize,
                           __write_only image2d_t imgComp, int width, int height,
                           __local float* imgP_reg, __local float* imgI_reg, __local float* costs,
                           __local int* x, __local int* y, __write_only image2d_t alphaComp,
                           float factor, float curveth, int bWeight)
{
    int cloc = get_local_id(0);
    int ddHor = cloc == 4 ? 0 : (cloc < 2 ? 0 : (cloc == 2 ? -1 : 1));
    int ddVer = cloc == 4 ? 0 : (cloc >= 2 ? 0 : (cloc == 0 ? -1 : 1));

    int i = get_global_id(1) / 1 * mbSize;
    int j = get_global_id(0) / 5 * mbSize;

    int copy_img_per = (mbSize + 4) / 5;
    for (int r = 0; r < mbSize; ++r) {
        for (int c = 0; c < copy_img_per; ++c) {
            int ic = c + cloc * copy_img_per;
            if (ic < mbSize) {
                uchar3 p1 = vload3(0, imgP + (width * (i + r) * 3) + (j + ic) * 3);
                imgP_reg[mbSize * r + ic] = 0.299 * p1.z + 0.587 * p1.y + 0.114 * p1.x;

                uint4 i1 = read_imageui(imgI, g_sa_none, (int2)(j + ic, i + r));
                imgI_reg[mbSize * r + ic] = 0.299 * i1.z + 0.587 * i1.y + 0.114 * i1.x;
            }
        }
    }

    if (cloc == 0) {
        *x = j;
        *y = i;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float cost_min;
    for (int stepSize = 2; stepSize >= 1; stepSize /= 2) {
        int refVer = *y + ddVer * stepSize;
        int refHor = *x + ddHor * stepSize;
        if (refVer < 0 || refVer + mbSize > height || refHor < 0 || refHor + mbSize > width) {
            costs[cloc] = 65537;
        } else {
            float blk_err = 0;
            for (int r = 0; r < mbSize; r += 2) {
                for (int c = 0; c < mbSize; c += 2) {
                    float curr = imgP_reg[mbSize * r + c];
                    float ref = 0;
                    if (refVer + r >= i && refVer + r < i + mbSize && refHor + c >= j &&
                        refHor + c < j + mbSize) {
                        ref = imgI_reg[mbSize * (refVer + r - i) + (refHor + c - j)];
                    } else {
                        uint4 i1 = read_imageui(imgI, g_sa_none, (int2)(refHor + c, refVer + r));
                        ref = 0.299 * i1.z + 0.587 * i1.y + 0.114 * i1.x;
                    }
                    blk_err += fabs(curr - ref);
                }
            }
            costs[cloc] = blk_err / (mbSize * mbSize / 4);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (cloc == 0) {
            cost_min = costs[4];
            int dx = 0;
            int dy = 0;
            for (int m = 0; m < 5; ++m) {
                if (costs[m] < cost_min) {
                    cost_min = costs[m];
                    dx = m == 4 ? 0 : (m < 2 ? 0 : (m == 2 ? -1 : 1));
                    dy = m == 4 ? 0 : (m >= 2 ? 0 : (m == 0 ? -1 : 1));
                }
            }
            *x += dx * stepSize;
            *y += dy * stepSize;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    int copy_imgComp_per = (mbSize + 4) / 5;
    for (int r = 0; r < mbSize; ++r) {
        for (int c = 0; c < copy_imgComp_per; ++c) {
            int ic = c + cloc * copy_imgComp_per;
            if (ic < mbSize) {
                uint4 data = read_imageui(imgI, g_sa_none, (int2)((*x + ic), (*y + r)));
                write_imageui(imgComp, (int2)(j + ic, i + r), data);
            }
        }
    }

    if (bWeight && cloc == 0) {
        float diff_val = pow(cost_min / factor, 3.0f);
        float alpha = diff_val / (diff_val + curveth);
        write_imagef(alphaComp, (int2)(get_global_id(0) / 5, get_global_id(1)),
                     (float4)(alpha, 0, 0, 0));
    }
}

__kernel void calc_alpha(__global uchar* curr_img, __read_only image2d_t pre_iir_comp,
                         __global uchar* image_out, __read_only image2d_t alphaComp, int width,
                         int height, __local float* alpha, int mbSize, int bWeight, float mAlpha,
                         __write_only image2d_t pre_img)
{
    int w = get_global_id(0);
    int h = get_global_id(1);
    int cloc = get_local_id(0);
    int rloc = get_local_id(1);

    if (cloc == 0 && rloc == 0) {
        if (bWeight) {
            *alpha = 1.0;
            int iw = w / mbSize;
            int ih = h / mbSize;
            if (iw < width / mbSize && ih < height / mbSize) {
                *alpha = read_imagef(alphaComp, g_sa_none, (int2)(iw, ih)).x;
            }
        } else {
            *alpha = mAlpha;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uchar3 cur_data = vload3(0, curr_img + (width * h * 3) + w * 3);
    float3 cur_rgb = (float3)(cur_data.x, cur_data.y, cur_data.z);

    uint4 comp_data = read_imageui(pre_iir_comp, g_sa_none, (int2)(w, h));
    float3 comp_rgb = (float3)(comp_data.x, comp_data.y, comp_data.z);

    float3 out_rgb = round(cur_rgb * (*alpha) + (1 - (*alpha)) * comp_rgb);
    uchar3 out_data = (uchar3)(out_rgb.x, out_rgb.y, out_rgb.z);

    vstore3(out_data, 0, image_out + (width * h * 3) + w * 3);
    write_imageui(pre_img, (int2)(w, h), (uint4)(out_data.x, out_data.y, out_data.z, 0));
}