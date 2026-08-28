use super::{CompilerOptions, GridPolicy, KernelArtifact, KernelBinaryFormat, LaunchRequirements};
use severian_fusion::{
    AliasKind, Dimension, DimensionExpression, ElementKind, FusionGraph, FusionRegion, GpuTarget,
    KernelSpecialization, Mutation, NodeKind, OperandRole, Rank, StorageLayout, Stride,
};
use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 8] = b"SEVGPU\0\x01";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CacheKey([u64; 4]);

impl CacheKey {
    pub fn for_kernel(
        graph: &FusionGraph,
        region: &FusionRegion,
        specialization: &KernelSpecialization,
        options: &CompilerOptions,
        donor_revision: &str,
    ) -> Self {
        let mut hash = StableHash::new();
        hash.bytes(b"severian-gpu-kernel-cache-v1");
        encode_graph(&mut hash, graph, region);
        encode_specialization(&mut hash, specialization);
        // Element representation is also encoded node-by-node in graph data;
        // this explicit stream makes that cache-key contract unmistakable.
        hash.u64(graph.nodes().len() as u64);
        for node in graph.nodes() {
            hash.u8(element_kind(node.shape.element_kind));
            hash.u16(node.shape.element_bits);
        }
        hash.string(&options.architecture);
        hash.string(donor_revision);
        hash.u8(target(options.target));
        hash.u32(options.num_warps);
        hash.u32(options.warp_size);
        hash.u32(options.num_ctas);
        hash.u32(options.num_stages);
        hash.u8(format(options.emit));
        hash.u8(u8::from(options.debug));
        Self(hash.finish())
    }

    pub fn to_hex(self) -> String {
        let mut output = String::with_capacity(64);
        for lane in self.0 {
            use std::fmt::Write as _;
            let _ = write!(output, "{lane:016x}");
        }
        output
    }

    fn bytes(self) -> [u8; 32] {
        let mut bytes = [0; 32];
        for (index, lane) in self.0.into_iter().enumerate() {
            bytes[index * 8..index * 8 + 8].copy_from_slice(&lane.to_le_bytes());
        }
        bytes
    }
}

impl fmt::Display for CacheKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.to_hex())
    }
}

#[derive(Debug)]
pub enum CacheError {
    Io(std::io::Error),
    Corrupt(String),
}

impl fmt::Display for CacheError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => error.fmt(formatter),
            Self::Corrupt(error) => write!(formatter, "corrupt compiled GPU kernel: {error}"),
        }
    }
}

impl std::error::Error for CacheError {}

impl From<std::io::Error> for CacheError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

#[derive(Debug, Default)]
pub struct KernelCache {
    directory: Option<PathBuf>,
    memory: BTreeMap<CacheKey, KernelArtifact>,
}

impl KernelCache {
    pub fn memory() -> Self {
        Self::default()
    }

    pub fn persistent(directory: impl Into<PathBuf>) -> Self {
        Self {
            directory: Some(directory.into()),
            memory: BTreeMap::new(),
        }
    }

    pub fn directory(&self) -> Option<&Path> {
        self.directory.as_deref()
    }

    pub fn get(&mut self, key: &CacheKey) -> Result<Option<KernelArtifact>, CacheError> {
        if let Some(artifact) = self.memory.get(key) {
            return Ok(Some(artifact.clone()));
        }
        let Some(directory) = &self.directory else {
            return Ok(None);
        };
        let path = directory.join(format!("{key}.sevkernel"));
        let bytes = match fs::read(path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let artifact = decode(*key, &bytes)?;
        self.memory.insert(*key, artifact.clone());
        Ok(Some(artifact))
    }

    pub fn insert(&mut self, key: CacheKey, artifact: KernelArtifact) -> Result<(), CacheError> {
        if let Some(directory) = &self.directory {
            fs::create_dir_all(directory)?;
            let path = directory.join(format!("{key}.sevkernel"));
            let temporary = directory.join(format!(".{key}.{}.tmp", std::process::id()));
            fs::write(&temporary, encode(key, &artifact))?;
            if let Err(error) = fs::rename(&temporary, &path) {
                let _ = fs::remove_file(&temporary);
                return Err(error.into());
            }
        }
        self.memory.insert(key, artifact);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.memory.len()
    }

    pub fn is_empty(&self) -> bool {
        self.memory.is_empty()
    }
}

fn encode_graph(hash: &mut StableHash, graph: &FusionGraph, region: &FusionRegion) {
    hash.u64(graph.nodes().len() as u64);
    for node in graph.nodes() {
        hash.u32(node.id.0);
        hash.u8(node_kind(node.kind));
        hash.string(&node.operation);
        hash.i64s(&node.attributes);
        hash.u32s(&node.inputs.iter().map(|input| input.0).collect::<Vec<_>>());
        hash.u64(node.operand_roles.len() as u64);
        for role in &node.operand_roles {
            hash.u8(match role {
                OperandRole::Data => 0,
                OperandRole::RuntimeShape => 1,
                OperandRole::RuntimeStrides => 2,
            });
        }
        encode_rank(hash, &node.shape.rank);
        hash.u64(node.shape.dimension_expressions.len() as u64);
        for expression in &node.shape.dimension_expressions {
            encode_dimension_expression(hash, expression);
        }
        hash.u8(element_kind(node.shape.element_kind));
        hash.u16(node.shape.element_bits);
        encode_layout(hash, &node.layout);
        hash.u64(node.bytes_read);
        hash.u64(node.bytes_written);
        hash.u64(node.flops);
        hash.u64(node.shared_memory_bytes);
        hash.u16(node.unnested_reductions);
        hash.u8(u8::from(node.has_side_effects));
        hash.u64(node.aliases.len() as u64);
        for alias in &node.aliases {
            hash.u16(alias.input_index);
            hash.u8(match alias.kind {
                AliasKind::View => 0,
                AliasKind::InPlace => 1,
            });
        }
        match node.mutation {
            Mutation::None => hash.u8(0),
            Mutation::WritesInput(input) => {
                hash.u8(1);
                hash.u16(input);
            }
        }
        match &node.matmul {
            None => hash.u8(0),
            Some(matmul) => {
                hash.u8(1);
                encode_rank(hash, &matmul.lhs_shape);
                encode_rank(hash, &matmul.rhs_shape);
                encode_rank(hash, &matmul.result_shape);
                hash.u64(matmul.batch_dimensions.len() as u64);
                for dimension in &matmul.batch_dimensions {
                    hash.u32(dimension.result);
                    hash.option_u32(dimension.lhs);
                    hash.option_u32(dimension.rhs);
                }
                hash.u64(matmul.contraction_dimensions.len() as u64);
                for dimension in &matmul.contraction_dimensions {
                    hash.u32(dimension.lhs);
                    hash.u32(dimension.rhs);
                }
            }
        }
    }
    hash.u32(region.id.0);
    hash.u32s(&region.nodes.iter().map(|node| node.0).collect::<Vec<_>>());
    hash.u32s(&region.inputs.iter().map(|node| node.0).collect::<Vec<_>>());
    hash.u32s(&region.outputs.iter().map(|node| node.0).collect::<Vec<_>>());
}

fn encode_dimension_expression(hash: &mut StableHash, expression: &DimensionExpression) {
    match expression {
        DimensionExpression::Constant(value) => {
            hash.u8(0);
            hash.u64(*value);
        }
        DimensionExpression::Symbol(symbol) => {
            hash.u8(1);
            hash.u64(*symbol);
        }
        DimensionExpression::Dynamic => hash.u8(2),
        DimensionExpression::Add(left, right) => {
            hash.u8(3);
            encode_dimension_expression(hash, left);
            encode_dimension_expression(hash, right);
        }
        DimensionExpression::Multiply(left, right) => {
            hash.u8(4);
            encode_dimension_expression(hash, left);
            encode_dimension_expression(hash, right);
        }
        DimensionExpression::DivideExact(left, right) => {
            hash.u8(5);
            encode_dimension_expression(hash, left);
            encode_dimension_expression(hash, right);
        }
    }
}

fn encode_specialization(hash: &mut StableHash, specialization: &KernelSpecialization) {
    hash.u8(target(specialization.target));
    let mut shapes = specialization.shapes.iter().collect::<Vec<_>>();
    shapes.sort_by_key(|shape| shape.node);
    hash.u64(shapes.len() as u64);
    for shape in shapes {
        hash.u32(shape.node.0);
        hash.u64s(&shape.dimensions);
    }
    let mut strides = specialization.strides.iter().collect::<Vec<_>>();
    strides.sort_by_key(|strides| strides.node);
    hash.u64(strides.len() as u64);
    for strides in strides {
        hash.u32(strides.node.0);
        hash.i64s(&strides.strides);
        hash.i64(strides.offset);
    }
}

fn encode_rank(hash: &mut StableHash, rank: &Rank) {
    match rank {
        Rank::Unranked => hash.u8(0),
        Rank::Ranked(dimensions) => {
            hash.u8(1);
            hash.u64(dimensions.len() as u64);
            for dimension in dimensions {
                match dimension {
                    Dimension::Dynamic => hash.u8(0),
                    Dimension::Known(value) => {
                        hash.u8(1);
                        hash.u64(*value);
                    }
                }
            }
        }
    }
}

fn encode_layout(hash: &mut StableHash, layout: &StorageLayout) {
    match layout {
        StorageLayout::Runtime => hash.u8(0),
        StorageLayout::Dense { minor_to_major } => {
            hash.u8(1);
            hash.u32s(minor_to_major);
        }
        StorageLayout::Strided { strides, offset } => {
            hash.u8(2);
            hash.u64(strides.len() as u64);
            for stride in strides {
                encode_stride(hash, *stride);
            }
            encode_stride(hash, *offset);
        }
    }
}

fn encode_stride(hash: &mut StableHash, stride: Stride) {
    match stride {
        Stride::Dynamic => hash.u8(0),
        Stride::Known(value) => {
            hash.u8(1);
            hash.i64(value);
        }
    }
}

fn node_kind(kind: NodeKind) -> u8 {
    match kind {
        NodeKind::Parameter => 0,
        NodeKind::Constant => 1,
        NodeKind::Elementwise => 2,
        NodeKind::Reduction => 3,
        NodeKind::Contraction => 4,
        NodeKind::Reshape => 5,
        NodeKind::Permute => 6,
        NodeKind::Slice => 7,
        NodeKind::Broadcast => 8,
        NodeKind::Gather => 9,
        NodeKind::Scatter => 10,
        NodeKind::Concatenate => 11,
        NodeKind::Convert => 12,
        NodeKind::StorageView => 13,
    }
}

fn element_kind(kind: ElementKind) -> u8 {
    match kind {
        ElementKind::SignedInteger => 0,
        ElementKind::UnsignedInteger => 1,
        ElementKind::IeeeFloat => 2,
        ElementKind::BrainFloat => 3,
        ElementKind::Float8E4M3Fn => 4,
        ElementKind::Float8E5M2 => 5,
        ElementKind::Boolean => 6,
        ElementKind::Opaque => 7,
    }
}

fn target(target: GpuTarget) -> u8 {
    match target {
        GpuTarget::Amd => 0,
        GpuTarget::Nvidia => 1,
    }
}

fn format(format: KernelBinaryFormat) -> u8 {
    match format {
        KernelBinaryFormat::LlvmIr => 0,
        KernelBinaryFormat::AmdGcN => 1,
        KernelBinaryFormat::Hsaco => 2,
        KernelBinaryFormat::Ptx => 3,
        KernelBinaryFormat::Cubin => 4,
    }
}

fn decode_format(value: u8) -> Result<KernelBinaryFormat, CacheError> {
    match value {
        0 => Ok(KernelBinaryFormat::LlvmIr),
        1 => Ok(KernelBinaryFormat::AmdGcN),
        2 => Ok(KernelBinaryFormat::Hsaco),
        3 => Ok(KernelBinaryFormat::Ptx),
        4 => Ok(KernelBinaryFormat::Cubin),
        _ => Err(CacheError::Corrupt("unknown binary format".into())),
    }
}

fn encode(key: CacheKey, artifact: &KernelArtifact) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&key.bytes());
    bytes.push(format(artifact.format));
    put_bytes(&mut bytes, artifact.entry_point.as_bytes());
    put_bytes(&mut bytes, &artifact.code);
    match artifact.launch.grid {
        GridPolicy::Fixed(grid) => {
            bytes.push(0);
            for value in grid {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        GridPolicy::Linear {
            output,
            elements_per_program,
        } => {
            bytes.push(1);
            bytes.extend_from_slice(&output.0.to_le_bytes());
            bytes.extend_from_slice(&elements_per_program.to_le_bytes());
        }
    }
    for value in artifact.launch.block {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes.extend_from_slice(&artifact.launch.num_warps.to_le_bytes());
    bytes.extend_from_slice(&artifact.launch.warp_size.to_le_bytes());
    bytes.extend_from_slice(&artifact.launch.num_ctas.to_le_bytes());
    bytes.extend_from_slice(&artifact.launch.shared_memory_bytes.to_le_bytes());
    bytes
}

fn decode(key: CacheKey, bytes: &[u8]) -> Result<KernelArtifact, CacheError> {
    let mut cursor = Cursor::new(bytes);
    if cursor.take(MAGIC.len())? != MAGIC {
        return Err(CacheError::Corrupt("bad magic/version".into()));
    }
    if cursor.take(32)? != key.bytes() {
        return Err(CacheError::Corrupt("cache key mismatch".into()));
    }
    let format = decode_format(cursor.u8()?)?;
    let entry_point = String::from_utf8(cursor.bytes()?.to_vec())
        .map_err(|_| CacheError::Corrupt("entry point is not UTF-8".into()))?;
    let code = cursor.bytes()?.to_vec();
    let grid = match cursor.u8()? {
        0 => GridPolicy::Fixed([cursor.u64()?, cursor.u64()?, cursor.u64()?]),
        1 => GridPolicy::Linear {
            output: severian_fusion::NodeId(cursor.u32()?),
            elements_per_program: cursor.u64()?,
        },
        _ => return Err(CacheError::Corrupt("unknown grid policy".into())),
    };
    let block = [cursor.u32()?, cursor.u32()?, cursor.u32()?];
    let num_warps = cursor.u32()?;
    let warp_size = cursor.u32()?;
    let num_ctas = cursor.u32()?;
    let shared_memory_bytes = cursor.u64()?;
    if !cursor.remaining().is_empty() {
        return Err(CacheError::Corrupt("trailing bytes".into()));
    }
    Ok(KernelArtifact {
        format,
        entry_point,
        code,
        launch: LaunchRequirements {
            grid,
            block,
            num_warps,
            warp_size,
            num_ctas,
            shared_memory_bytes,
        },
    })
}

fn put_bytes(output: &mut Vec<u8>, value: &[u8]) {
    output.extend_from_slice(&(value.len() as u64).to_le_bytes());
    output.extend_from_slice(value);
}

struct Cursor<'a> {
    remaining: &'a [u8],
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { remaining: bytes }
    }

    fn remaining(&self) -> &'a [u8] {
        self.remaining
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], CacheError> {
        if count > self.remaining.len() {
            return Err(CacheError::Corrupt("truncated artifact".into()));
        }
        let (value, remaining) = self.remaining.split_at(count);
        self.remaining = remaining;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, CacheError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, CacheError> {
        let bytes = self.take(4)?.try_into().expect("four bytes");
        Ok(u32::from_le_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64, CacheError> {
        let bytes = self.take(8)?.try_into().expect("eight bytes");
        Ok(u64::from_le_bytes(bytes))
    }

    fn bytes(&mut self) -> Result<&'a [u8], CacheError> {
        let length = usize::try_from(self.u64()?)
            .map_err(|_| CacheError::Corrupt("length does not fit usize".into()))?;
        self.take(length)
    }
}

struct StableHash {
    lanes: [u64; 4],
}

impl StableHash {
    fn new() -> Self {
        Self {
            lanes: [
                0xcbf29ce484222325,
                0x84222325cbf29ce4,
                0x9e3779b97f4a7c15,
                0x517cc1b727220a95,
            ],
        }
    }

    fn finish(self) -> [u64; 4] {
        self.lanes
    }

    fn bytes(&mut self, bytes: &[u8]) {
        self.raw(&(bytes.len() as u64).to_le_bytes());
        self.raw(bytes);
    }

    fn raw(&mut self, bytes: &[u8]) {
        for byte in bytes {
            for (index, lane) in self.lanes.iter_mut().enumerate() {
                *lane ^= u64::from(*byte).wrapping_add((index as u64) << 8);
                *lane = lane.wrapping_mul(0x100000001b3 + index as u64 * 2);
                *lane ^= *lane >> (29 + index);
            }
        }
    }

    fn string(&mut self, value: &str) {
        self.bytes(value.as_bytes());
    }

    fn u8(&mut self, value: u8) {
        self.raw(&[value]);
    }

    fn u16(&mut self, value: u16) {
        self.raw(&value.to_le_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.raw(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.raw(&value.to_le_bytes());
    }

    fn i64(&mut self, value: i64) {
        self.raw(&value.to_le_bytes());
    }

    fn u32s(&mut self, values: &[u32]) {
        self.u64(values.len() as u64);
        for value in values {
            self.u32(*value);
        }
    }

    fn u64s(&mut self, values: &[u64]) {
        self.u64(values.len() as u64);
        for value in values {
            self.u64(*value);
        }
    }

    fn i64s(&mut self, values: &[i64]) {
        self.u64(values.len() as u64);
        for value in values {
            self.i64(*value);
        }
    }

    fn option_u32(&mut self, value: Option<u32>) {
        match value {
            None => self.u8(0),
            Some(value) => {
                self.u8(1);
                self.u32(value);
            }
        }
    }
}
