use std::{ops::Range, sync::Arc};

use super::MID;
use cfg_if::cfg_if;

const UNITS: i16 = 400_i16;
const FT_SCALE: i16 = 255;
const SCALE: i16 = 64;
const MAX: i16 = FT_SCALE;

#[derive(Debug, Copy, Clone)]
#[repr(C, align(64))]
pub struct Align<T>(pub T);

#[derive(Debug, Clone)]
pub struct Incremental<const INPUT: usize, const OUTPUT: usize> {
    weights: Arc<Align<[[i16; OUTPUT]; INPUT]>>,
    out: Align<[i16; OUTPUT]>,
}

impl<const INPUT: usize, const OUTPUT: usize> Incremental<INPUT, OUTPUT> {
    pub fn new(weights: Arc<Align<[[i16; OUTPUT]; INPUT]>>, bias: Align<[i16; OUTPUT]>) -> Self {
        Self { weights, out: bias }
    }

    pub fn reset(&mut self, bias: [i16; OUTPUT]) {
        self.out.0 = bias;
    }

    pub fn update_features(&mut self, added_features: &[usize], removed_features: &[usize]) {
        cfg_if! {
            if #[cfg(target_feature = "avx2")] {
                const CHUNKS: usize = 256;
            } else {
                const CHUNKS: usize = 128;
            }
        }
        for start in 0..(OUTPUT + CHUNKS - 1) / CHUNKS {
            let range = start * CHUNKS..(start * CHUNKS + CHUNKS).min(OUTPUT);
            let mut out_reg = [0; CHUNKS];
            out_reg[..range.len()].copy_from_slice(&self.out.0[range.clone()]);
            self.update_chunk::<1>(added_features, &mut out_reg, range.clone());
            self.update_chunk::<-1>(removed_features, &mut out_reg, range.clone());
            self.out.0[range.clone()].copy_from_slice(&out_reg[..range.len()]);
        }
    }

    fn update_chunk<const SIGN: i16>(
        &self,
        feature_indices: &[usize],
        reg: &mut [i16],
        chunk: Range<usize>,
    ) {
        for &index in feature_indices {
            for (out, &weight) in reg.iter_mut().zip(&self.weights.0[index][chunk.clone()]) {
                *out += weight * SIGN;
            }
        }
    }

    pub fn get(&self) -> &Align<[i16; OUTPUT]> {
        let slice = &self.out.0;
        // SAFETY: The resulting slice is indeed OUTPUT long,
        // and we check that the slice is aligned to 64 bytes.
        // additionally, we're generating the reference from our own data,
        // so we know that the lifetime is valid.
        unsafe {
            // don't immediately cast to Align64, as we want to check the alignment first.
            let ptr = slice.as_ptr();
            assert_eq!(ptr.align_offset(64), 0);
            // alignments are sensible, so we can safely cast.
            #[allow(clippy::cast_ptr_alignment)]
            &*ptr.cast()
        }
    }
}

#[derive(Debug, Clone)]
pub struct Dense<const OUTPUT: usize> {
    weights: Arc<[Align<[i16; MID * 2]>; OUTPUT]>,
    bias: Align<[i32; OUTPUT]>,
}

impl<const OUTPUT: usize> Dense<OUTPUT> {
    pub fn new(weights: Arc<[Align<[i16; MID * 2]>; OUTPUT]>, bias: Align<[i32; OUTPUT]>) -> Self {
        Self { weights, bias }
    }

    pub fn feed_forward(
        &self,
        stm: &Align<[i16; MID]>,
        nstm: &Align<[i16; MID]>,
        bucket: usize,
    ) -> i32 {
        let weights = &self.weights[bucket];

        let out = flatten(stm, nstm, weights);

        out + self.bias.0[bucket]
    }
}

pub fn scale_network_output(x: i32) -> i16 {
    (x * UNITS as i32 / (FT_SCALE as i32 * SCALE as i32)) as i16
}

fn flatten(
    us: &Align<[i16; MID]>,
    them: &Align<[i16; MID]>,
    weights: &Align<[i16; MID * 2]>,
) -> i32 {
    #[cfg(target_feature = "avx2")]
    unsafe {
        avx2::flatten(us, them, weights)
    }
    #[cfg(target_feature = "neon")]
    unsafe {
        neon::flatten(us, them, weights)
    }
    #[cfg(not(any(target_feature = "avx2", target_feature = "neon")))]
    generic::flatten(us, them, weights)
}

/// Non-SIMD implementation of the forward pass.
#[cfg(not(target_feature = "avx2"))]
mod generic {
    use super::super::MID as LAYER_1_SIZE;
    use super::{Align, MAX};

    #[allow(clippy::cast_possible_truncation)]
    fn screlu(x: i16) -> i32 {
        let x = x.clamp(0, MAX);
        let x = i32::from(x);
        x * x
    }

    /// Execute an activation on the partial activations,
    /// and accumulate the result into a sum.
    pub fn flatten(
        us: &Align<[i16; LAYER_1_SIZE]>,
        them: &Align<[i16; LAYER_1_SIZE]>,
        weights: &Align<[i16; LAYER_1_SIZE * 2]>,
    ) -> i32 {
        let mut sum: i32 = 0;
        for (&i, &w) in us.0.iter().zip(&weights.0[..LAYER_1_SIZE]) {
            sum += screlu(i) * i32::from(w);
        }
        for (&i, &w) in them.0.iter().zip(&weights.0[LAYER_1_SIZE..]) {
            sum += screlu(i) * i32::from(w);
        }
        sum / MAX as i32
    }
}

/// SIMD implementation of the forward pass.
#[cfg(target_feature = "avx2")]
mod avx2 {
    use super::super::MID as LAYER_1_SIZE;
    use super::{Align, MAX};
    use std::arch::x86_64::{
        __m256i, _mm256_add_epi32, _mm256_castsi256_si128, _mm256_extracti128_si256,
        _mm256_load_si256, _mm256_madd_epi16, _mm256_max_epi16, _mm256_min_epi16,
        _mm256_mullo_epi16, _mm256_set1_epi16, _mm256_setzero_si256, _mm_add_epi32,
        _mm_cvtsi128_si32, _mm_shuffle_epi32, _mm_unpackhi_epi64,
    };

    type Vec256 = __m256i;

    #[inline]
    unsafe fn load_i16s<const VEC_SIZE: usize>(
        acc: &Align<[i16; VEC_SIZE]>,
        start_idx: usize,
    ) -> Vec256 {
        _mm256_load_si256(acc.0.as_ptr().add(start_idx).cast())
    }

    #[inline]
    unsafe fn horizontal_sum_i32(sum: Vec256) -> i32 {
        let upper_128 = _mm256_extracti128_si256::<1>(sum);
        let lower_128 = _mm256_castsi256_si128(sum);
        let sum_128 = _mm_add_epi32(upper_128, lower_128);
        let upper_64 = _mm_unpackhi_epi64(sum_128, sum_128);
        let sum_64 = _mm_add_epi32(upper_64, sum_128);
        let upper_32 = _mm_shuffle_epi32::<0b00_00_00_01>(sum_64);
        let sum_32 = _mm_add_epi32(upper_32, sum_64);

        _mm_cvtsi128_si32(sum_32)
    }

    /// Execute an activation on the partial activations,
    /// and accumulate the result into a sum.
    pub unsafe fn flatten(
        us: &Align<[i16; LAYER_1_SIZE]>,
        them: &Align<[i16; LAYER_1_SIZE]>,
        weights: &Align<[i16; LAYER_1_SIZE * 2]>,
    ) -> i32 {
        const CHUNK: usize = 16;

        let mut sum = _mm256_setzero_si256();
        let min = _mm256_setzero_si256();
        let max = _mm256_set1_epi16(MAX);

        // the following code uses a trick devised by the author of the Lizard chess engine.
        // we're implementing the function f(x) = clamp(x, 0, MAX)^2 * w,
        // and we do this in the following manner:
        // 1. load the input, x
        // 2. compute v := clamp(x, 0, MAX)
        // 3. load the weight, w
        // 4. compute t := v * w via truncating 16-bit multiply.
        //    this step relies on our invariant that v * w fits in i16.
        // 5. compute product := v * t via horizontally accumulating
        //    expand-to-i32 multiply.
        // 6. add product to the running sum.
        // at this point we've computed clamp(x, 0, MAX)^2 * w
        // by doing (clamp(x, 0, MAX) * w) * clamp(x, 0, MAX).
        // the clever part is step #4, which the compiler cannot know to do.

        // accumulate the first half of the weights
        for i in 0..LAYER_1_SIZE / CHUNK {
            let x = load_i16s(us, i * CHUNK);
            let v = _mm256_min_epi16(_mm256_max_epi16(x, min), max);
            let w = load_i16s(weights, i * CHUNK);
            let t = _mm256_mullo_epi16(v, w);
            let product = _mm256_madd_epi16(v, t);
            sum = _mm256_add_epi32(sum, product);
        }

        // accumulate the second half of the weights
        for i in 0..LAYER_1_SIZE / CHUNK {
            let x = load_i16s(them, i * CHUNK);
            let v = _mm256_min_epi16(_mm256_max_epi16(x, min), max);
            let w = load_i16s(weights, LAYER_1_SIZE + i * CHUNK);
            let t = _mm256_mullo_epi16(v, w);
            let product = _mm256_madd_epi16(v, t);
            sum = _mm256_add_epi32(sum, product);
        }

        horizontal_sum_i32(sum) / MAX as i32
    }
}

#[cfg(target_feature = "neon")]
mod neon {
    use super::super::MID as LAYER_1_SIZE;
    use super::{Align, MAX};
    use std::{arch::aarch64::*, mem::size_of_val};

    #[inline]
    unsafe fn load_i16s<const VEC_SIZE: usize>(
        acc: &Align<[i16; VEC_SIZE]>,
        start_idx: usize,
    ) -> int16x8_t {
        vld1q_s16(acc.0.as_ptr().add(start_idx).cast())
    }

    #[inline]
    unsafe fn horizontal_sum_i32(sum: int32x4_t) -> i32 {
        vaddlvq_s32(sum) as i32
    }

    #[inline]
    unsafe fn madd(a: int16x8_t, b: int16x8_t) -> int32x4_t {
        let a_lo = vget_low_s16(a);
        let b_lo = vget_low_s16(b);
        let a_hi = vget_high_s16(a);
        let b_hi = vget_high_s16(b);
        let product_lo = vmull_s16(a_lo, b_lo);
        let product_hi = vmull_s16(a_hi, b_hi);
        vaddq_s32(product_lo, product_hi)
    }

    /// Execute an activation on the partial activations,
    /// and accumulate the result into a sum.
    pub unsafe fn flatten(
        us: &Align<[i16; LAYER_1_SIZE]>,
        them: &Align<[i16; LAYER_1_SIZE]>,
        weights: &Align<[i16; LAYER_1_SIZE * 2]>,
    ) -> i32 {
        const CHUNK: usize = 8;

        let mut sum = vld1q_dup_s32(&0);
        let min = vld1q_dup_s16(&0);
        let max = vld1q_dup_s16(&MAX);

        // the following code uses a trick devised by the author of the Lizard chess engine.
        // we're implementing the function f(x) = clamp(x, 0, MAX)^2 * w,
        // and we do this in the following manner:
        // 1. load the input, x
        // 2. compute v := clamp(x, 0, MAX)
        // 3. load the weight, w
        // 4. compute t := v * w via truncating 16-bit multiply.
        //    this step relies on our invariant that v * w fits in i16.
        // 5. compute product := v * t via horizontally accumulating
        //    expand-to-i32 multiply.
        // 6. add product to the running sum.
        // at this point we've computed clamp(x, 0, MAX)^2 * w
        // by doing (clamp(x, 0, MAX) * w) * clamp(x, 0, MAX).
        // the clever part is step #4, which the compiler cannot know to do.

        // accumulate the first half of the weights
        for i in 0..LAYER_1_SIZE / CHUNK {
            let x = load_i16s(us, i * CHUNK);
            let v = vminq_s16(vmaxq_s16(x, min), max);
            let w = load_i16s(weights, i * CHUNK);
            let t = vmulq_s16(v, w);
            let product = madd(v, t);
            sum = vaddq_s32(sum, product);
        }

        // accumulate the second half of the weights
        for i in 0..LAYER_1_SIZE / CHUNK {
            let x = load_i16s(them, i * CHUNK);
            let v = vminq_s16(vmaxq_s16(x, min), max);
            let w = load_i16s(weights, LAYER_1_SIZE + i * CHUNK);
            let t = vmulq_s16(v, w);
            let product = madd(v, t);
            sum = vaddq_s32(sum, product);
        }

        horizontal_sum_i32(sum) / MAX as i32
    }
}
