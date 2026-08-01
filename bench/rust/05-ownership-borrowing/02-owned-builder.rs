struct Buffer { bytes: Vec<u8> }
impl Buffer { fn push(&mut self, byte: u8) { self.bytes.push(byte); } }
fn freeze(buffer: Buffer) -> Buffer { buffer }
fn main() { let mut buffer = Buffer { bytes: vec![] }; buffer.push(65); println!("{:?}", freeze(buffer).bytes); }
