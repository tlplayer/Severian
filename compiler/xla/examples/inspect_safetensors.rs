use severian_xla::SafeTensorStore;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: inspect_safetensors MODEL_DIRECTORY")?;
    let store = SafeTensorStore::open(&directory)?;
    let validation = store.validate_all()?;
    let embedding = store.get("model.embed_tokens.weight")?;
    let first_query = store.get("model.layers.0.self_attn.q_proj.weight")?;
    let last_query = store.get("model.layers.35.self_attn.q_proj.weight")?;
    println!(
        "tensors={} shards={} payload_bytes={} bf16_payload_bytes={}",
        validation.tensors,
        validation.shards,
        validation.payload_bytes,
        validation.bf16_payload_bytes,
    );
    println!("embedding={:?} {:?}", embedding.entry().dtype, embedding.entry().shape);
    println!("layer0_q={:?} {:?}", first_query.entry().dtype, first_query.entry().shape);
    println!("layer35_q={:?} {:?}", last_query.entry().dtype, last_query.entry().shape);
    Ok(())
}
