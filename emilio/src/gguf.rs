//! Minimal GGUF parser — reads metadata + tensor info + dequantizes weights.
//!
//! Supports the subset needed for Qwen2.5-0.5B-Instruct Q8_0:
//!   - GGUF v3 format
//!   - Metadata types: uint8, int8, uint16, int16, uint32, int32, float32,
//!     bool, string, uint64, int64, float64, array
//!   - Tensor types: F32, F16, Q8_0
//!
//! Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};

// ─── GGUF constants ─────────────────────────────────────────────────────────

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" as LE u32: [0x47,0x47,0x55,0x46]

/// Tensor data types (ggml_type enum).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u32)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    IQ2XXS = 16,
    IQ2XS = 17,
    IQ3XXS = 18,
    IQ1S = 19,
    IQ4NL = 20,
    IQ3S = 21,
    IQ2S = 22,
    IQ4XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1M = 29,
    BF16 = 30,
    Unknown = 0xFFFF,
}

impl From<u32> for GGMLType {
    fn from(v: u32) -> Self {
        match v {
            0 => GGMLType::F32,
            1 => GGMLType::F16,
            2 => GGMLType::Q4_0,
            3 => GGMLType::Q4_1,
            6 => GGMLType::Q5_0,
            7 => GGMLType::Q5_1,
            8 => GGMLType::Q8_0,
            9 => GGMLType::Q8_1,
            10 => GGMLType::Q2K,
            11 => GGMLType::Q3K,
            12 => GGMLType::Q4K,
            13 => GGMLType::Q5K,
            14 => GGMLType::Q6K,
            15 => GGMLType::Q8K,
            24 => GGMLType::I8,
            25 => GGMLType::I16,
            26 => GGMLType::I32,
            27 => GGMLType::I64,
            28 => GGMLType::F64,
            30 => GGMLType::BF16,
            _ => GGMLType::Unknown,
        }
    }
}

/// Metadata value types.
#[derive(Debug, Clone)]
pub enum MetaValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    Str(String),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    Array(Vec<MetaValue>),
}

impl MetaValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            MetaValue::Uint32(v) => Some(*v),
            MetaValue::Int32(v) => Some(*v as u32),
            MetaValue::Uint64(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            MetaValue::Float32(v) => Some(*v),
            MetaValue::Float64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_string_array(&self) -> Option<Vec<&str>> {
        match self {
            MetaValue::Array(arr) => {
                let mut result = Vec::new();
                for item in arr {
                    if let MetaValue::Str(s) = item {
                        result.push(s.as_str());
                    } else {
                        return None;
                    }
                }
                Some(result)
            }
            _ => None,
        }
    }
}

impl std::fmt::Display for MetaValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetaValue::Uint8(v) => write!(f, "{v}"),
            MetaValue::Int8(v) => write!(f, "{v}"),
            MetaValue::Uint16(v) => write!(f, "{v}"),
            MetaValue::Int16(v) => write!(f, "{v}"),
            MetaValue::Uint32(v) => write!(f, "{v}"),
            MetaValue::Int32(v) => write!(f, "{v}"),
            MetaValue::Float32(v) => write!(f, "{v}"),
            MetaValue::Bool(v) => write!(f, "{v}"),
            MetaValue::Str(v) => write!(f, "\"{v}\""),
            MetaValue::Uint64(v) => write!(f, "{v}"),
            MetaValue::Int64(v) => write!(f, "{v}"),
            MetaValue::Float64(v) => write!(f, "{v}"),
            MetaValue::Array(arr) => write!(f, "[{} items]", arr.len()),
        }
    }
}

/// Tensor info from the GGUF header.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: GGMLType,
    pub offset: u64,
}

impl TensorInfo {
    /// Total number of elements.
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product::<u64>().max(1)
    }

    /// Size in bytes of this tensor's data.
    pub fn byte_size(&self) -> u64 {
        let n = self.n_elements();
        match self.dtype {
            GGMLType::F32 => n * 4,
            GGMLType::F16 => n * 2,
            GGMLType::Q8_0 => {
                // Q8_0: blocks of 32 values, each block = 2 bytes (f16 scale) + 32 bytes (int8)
                let n_blocks = n.div_ceil(32);
                n_blocks * 34
            }
            GGMLType::Q4_0 => {
                let n_blocks = n.div_ceil(32);
                n_blocks * 18 // 2 bytes f16 + 16 bytes (32 * 4-bit)
            }
            GGMLType::BF16 => n * 2,
            _ => 0, // unsupported
        }
    }
}

/// Parsed GGUF file.
pub struct GGUFFile {
    pub version: u32,
    pub metadata: HashMap<String, MetaValue>,
    pub tensors: Vec<TensorInfo>,
    /// Offset in the file where tensor data starts.
    pub data_offset: u64,
    /// Path to the file (for lazy loading).
    pub path: String,
}

// ─── Reader helpers ─────────────────────────────────────────────────────────

fn read_u8(r: &mut impl Read) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_i8(r: &mut impl Read) -> io::Result<i8> {
    Ok(read_u8(r)? as i8)
}

fn read_u16(r: &mut impl Read) -> io::Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_i16(r: &mut impl Read) -> io::Result<i16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(i16::from_le_bytes(buf))
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32(r: &mut impl Read) -> io::Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_i64(r: &mut impl Read) -> io::Result<i64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(i64::from_le_bytes(buf))
}

fn read_f32(r: &mut impl Read) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64(r: &mut impl Read) -> io::Result<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_f16_as_f32(r: &mut impl Read) -> io::Result<f32> {
    let bits = read_u16(r)?;
    Ok(f16_to_f32(bits))
}

/// Convert IEEE 754 half-precision to single-precision.
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // Denormalized
        let mut e = exp;
        let mut m = mant;
        while (m & 0x400) == 0 {
            m <<= 1;
            e = e.wrapping_sub(1);
        }
        m &= 0x3ff;
        let bits = (sign << 31) | ((e + 127 - 15 + 1) << 23) | (m << 13);
        return f32::from_bits(bits);
    }
    if exp == 0x1f {
        let bits = (sign << 31) | (0xff << 23) | (mant << 13);
        return f32::from_bits(bits);
    }
    let bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    f32::from_bits(bits)
}

fn read_string(r: &mut impl Read) -> io::Result<String> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn read_meta_value(r: &mut impl Read, type_id: u32) -> io::Result<MetaValue> {
    match type_id {
        0 => Ok(MetaValue::Uint8(read_u8(r)?)),
        1 => Ok(MetaValue::Int8(read_i8(r)?)),
        2 => Ok(MetaValue::Uint16(read_u16(r)?)),
        3 => Ok(MetaValue::Int16(read_i16(r)?)),
        4 => Ok(MetaValue::Uint32(read_u32(r)?)),
        5 => Ok(MetaValue::Int32(read_i32(r)?)),
        6 => Ok(MetaValue::Float32(read_f32(r)?)),
        7 => Ok(MetaValue::Bool(read_u8(r)? != 0)),
        8 => Ok(MetaValue::Str(read_string(r)?)),
        9 => {
            // Array
            let elem_type = read_u32(r)?;
            let len = read_u64(r)? as usize;
            let mut arr = Vec::with_capacity(len.min(1_000_000));
            for _ in 0..len {
                arr.push(read_meta_value(r, elem_type)?);
            }
            Ok(MetaValue::Array(arr))
        }
        10 => Ok(MetaValue::Uint64(read_u64(r)?)),
        11 => Ok(MetaValue::Int64(read_i64(r)?)),
        12 => Ok(MetaValue::Float64(read_f64(r)?)),
        _ => Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("unknown metadata type: {type_id}"))),
    }
}

// ─── GGUF parser ────────────────────────────────────────────────────────────

impl GGUFFile {
    /// Parse a GGUF file from disk.  Reads only the header — tensor data
    /// is loaded on demand via `load_tensor_f64`.
    pub fn parse(path: &str) -> io::Result<Self> {
        let f = File::open(path)?;
        let mut r = BufReader::new(f);

        // Magic
        let magic = read_u32(&mut r)?;
        if magic != GGUF_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("bad GGUF magic: 0x{magic:08x}, expected 0x{GGUF_MAGIC:08x}")));
        }

        // Version
        let version = read_u32(&mut r)?;
        if !(2..=3).contains(&version) {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("unsupported GGUF version: {version}")));
        }

        // Counts
        let n_tensors = read_u64(&mut r)? as usize;
        let n_meta = read_u64(&mut r)? as usize;

        // Metadata
        let mut metadata = HashMap::new();
        for _ in 0..n_meta {
            let key = read_string(&mut r)?;
            let val_type = read_u32(&mut r)?;
            let val = read_meta_value(&mut r, val_type)?;
            metadata.insert(key, val);
        }

        // Tensor info
        let mut tensors = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = read_string(&mut r)?;
            let n_dims = read_u32(&mut r)? as usize;
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(read_u64(&mut r)?);
            }
            let dtype = GGMLType::from(read_u32(&mut r)?);
            let offset = read_u64(&mut r)?;
            tensors.push(TensorInfo { name, dims, dtype, offset });
        }

        // Data offset — aligned to 32 bytes
        let header_end = r.stream_position()?;
        let alignment = metadata.get("general.alignment")
            .and_then(|v| v.as_u32())
            .unwrap_or(32) as u64;
        let data_offset = header_end.div_ceil(alignment) * alignment;

        Ok(GGUFFile {
            version,
            metadata,
            tensors,
            data_offset,
            path: path.to_string(),
        })
    }

    /// Get a metadata value by key.
    pub fn meta(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key)
    }

    /// Get a metadata value as u32.
    pub fn meta_u32(&self, key: &str) -> Option<u32> {
        self.metadata.get(key).and_then(|v| v.as_u32())
    }

    /// Get a metadata value as f32.
    pub fn meta_f32(&self, key: &str) -> Option<f32> {
        self.metadata.get(key).and_then(|v| v.as_f32())
    }

    /// Get a metadata value as string.
    pub fn meta_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(|v| v.as_str())
    }

    /// Find a tensor by name.
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Load tensor data and dequantize to f64.
    pub fn load_tensor_f64(&self, tensor: &TensorInfo) -> io::Result<Vec<f64>> {
        let mut f = File::open(&self.path)?;
        let abs_offset = self.data_offset + tensor.offset;
        f.seek(SeekFrom::Start(abs_offset))?;
        let mut r = BufReader::new(f);

        let n = tensor.n_elements() as usize;

        match tensor.dtype {
            GGMLType::F32 => {
                let mut data = Vec::with_capacity(n);
                for _ in 0..n {
                    data.push(read_f32(&mut r)? as f64);
                }
                Ok(data)
            }
            GGMLType::F16 => {
                let mut data = Vec::with_capacity(n);
                for _ in 0..n {
                    data.push(read_f16_as_f32(&mut r)? as f64);
                }
                Ok(data)
            }
            GGMLType::Q8_0 => {
                dequant_q8_0(&mut r, n)
            }
            GGMLType::BF16 => {
                let mut data = Vec::with_capacity(n);
                for _ in 0..n {
                    let bits = read_u16(&mut r)?;
                    let f = f32::from_bits((bits as u32) << 16);
                    data.push(f as f64);
                }
                Ok(data)
            }
            _ => Err(io::Error::new(io::ErrorKind::Unsupported,
                format!("dequantization not implemented for {:?}", tensor.dtype))),
        }
    }

    /// Print a summary of the model.
    pub fn print_summary(&self) {
        println!("GGUF v{}", self.version);
        println!("Tensors: {}", self.tensors.len());
        println!("Metadata keys: {}", self.metadata.len());
        println!();

        // Architecture info
        let arch = self.meta_str("general.architecture").unwrap_or("unknown");
        let name = self.meta_str("general.name").unwrap_or("unknown");
        println!("Model: {name}");
        println!("Architecture: {arch}");

        // Key dimensions
        let keys = [
            "general.architecture",
            &format!("{arch}.embedding_length"),
            &format!("{arch}.block_count"),
            &format!("{arch}.attention.head_count"),
            &format!("{arch}.attention.head_count_kv"),
            &format!("{arch}.feed_forward_length"),
            &format!("{arch}.context_length"),
            &format!("{arch}.attention.layer_norm_rms_epsilon"),
            &format!("{arch}.rope.freq_base"),
            &format!("{arch}.vocab_size"),
        ];

        println!();
        for key in &keys {
            if let Some(val) = self.metadata.get(*key) {
                println!("  {key}: {val}");
            }
        }

        // Tensor summary by type
        println!();
        let mut type_counts: HashMap<String, (usize, u64)> = HashMap::new();
        for t in &self.tensors {
            let entry = type_counts.entry(format!("{:?}", t.dtype)).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += t.byte_size();
        }
        println!("Tensor types:");
        for (dtype, (count, bytes)) in &type_counts {
            println!("  {dtype}: {count} tensors, {:.1} MB", *bytes as f64 / 1e6);
        }
    }
}

// ─── Q8_0 dequantization ────────────────────────────────────────────────────
//
// Q8_0 format: blocks of 32 values
//   - 2 bytes: f16 scale factor (delta)
//   - 32 bytes: 32 × int8 quantized values
//
// Dequantized value = int8_val * delta

fn dequant_q8_0(r: &mut impl Read, n_elements: usize) -> io::Result<Vec<f64>> {
    let n_blocks = n_elements.div_ceil(32);
    let mut result = Vec::with_capacity(n_blocks * 32);

    for _ in 0..n_blocks {
        let delta = read_f16_as_f32(r)? as f64;
        let mut quants = [0i8; 32];
        let mut buf = [0u8; 32];
        r.read_exact(&mut buf)?;
        for (i, &b) in buf.iter().enumerate() {
            quants[i] = b as i8;
        }
        for &q in &quants {
            result.push(q as f64 * delta);
        }
    }

    result.truncate(n_elements);
    Ok(result)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_to_f32() {
        // 1.0 in f16 = 0x3C00
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-6);
        // 0.0
        assert_eq!(f16_to_f32(0x0000), 0.0);
        // -1.0 in f16 = 0xBC00
        assert!((f16_to_f32(0xBC00) - (-1.0)).abs() < 1e-6);
        // 0.5 in f16 = 0x3800
        assert!((f16_to_f32(0x3800) - 0.5).abs() < 1e-6);
    }
}
