use std::{ops::Range, sync::Arc};

use cfg_if::cfg_if;

const UNITS: i16 = 400_i16;
const FT_SCALE: i16 = 255;
const SCALE: i16 = 64;
const MIN: i16 = 0;
const MAX: i16 = FT_SCALE;
const SHIFT: i16 = 8;

#[derive(Debug, Copy, Clone)]
#[repr(align(64))]
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

    fn update_chunk<const SIGN: i16>(&self, feature_indices: &[usize], reg: &mut [i16], chunk: Range<usize>) {
        for &index in feature_indices {
            for (out, &weight) in reg.iter_mut().zip(&self.weights.0[index][chunk.clone()]) {
                *out += weight * SIGN;
            }
        }
    }

    pub fn get(&self) -> &[i16; OUTPUT] {
        &self.out.0
    }
}

#[derive(Debug, Clone)]
pub struct Dense<const INPUT: usize, const OUTPUT: usize> {
    weights: Arc<Align<[[i8; INPUT]; OUTPUT]>>,
    bias: Align<[i32; OUTPUT]>,
}

impl<const INPUT: usize, const OUTPUT: usize> Dense<INPUT, OUTPUT> {
    pub fn new(weights: Arc<Align<[[i8; INPUT]; OUTPUT]>>, bias: Align<[i32; OUTPUT]>) -> Self {
        Self { weights, bias }
    }

    #[inline]
    pub fn feed_forward(&self, inputs: &[u8; INPUT], bucket: usize) -> i32 {
        let mut out = self.bias.0[bucket];
        cfg_if! {
            if #[cfg(target_feature = "avx2")] {
                use std::arch::x86_64;
                const CHUNKS_8: usize = 256 / 8;
                const CHUNKS_16: usize = 256 / 16;
                const CHUNKS_32: usize = 256 / 32;

                let ones = unsafe { x86_64::_mm256_load_si256([1_i16; CHUNKS_16].as_ptr() as *const _) };
                let mut store = [0; CHUNKS_32];
                let mut accumulate = unsafe { x86_64::_mm256_load_si256(Align([0; CHUNKS_32]).0.as_ptr() as *const _) };
                for (inputs, weights) in inputs.chunks(CHUNKS_8).zip(self.weights.0[bucket].chunks(CHUNKS_8)) {
                    unsafe {
                        let inputs = x86_64::_mm256_load_si256(inputs.as_ptr() as *const _);
                        let weights = x86_64::_mm256_load_si256(weights.as_ptr() as *const _);
                        let result = x86_64::_mm256_maddubs_epi16(inputs, weights);
                        let result = x86_64::_mm256_madd_epi16(result, ones);
                        accumulate = x86_64::_mm256_add_epi32(accumulate, result);
                    }
                }
                unsafe { x86_64::_mm256_store_si256(store.as_mut_ptr() as *mut _, accumulate) };
                out += store.iter().sum::<i32>();
            } else {
                for (&input, &weight) in inputs.iter().zip(self.weights.0[bucket].iter()) {
                    out += weight as i32 * input as i32;
                }
            }
        }
        out
    }
}

#[inline]
pub fn scale_network_output(x: i32) -> i16 {
    (x as i32 * UNITS as i32 / (FT_SCALE as i32 * SCALE as i32)) as i16
}

#[inline]
pub fn sq_clipped_relu<const N: usize>(array: [i16; N], out: &mut [u8]) {
    for (&x, clipped) in array.iter().zip(out.iter_mut()) {
        let tmp = x.max(MIN).min(MAX) as u16;
        *clipped = ((tmp * tmp) >> SHIFT) as u8;
    }
}
