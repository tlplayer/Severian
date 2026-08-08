use std::{
    io,
    thread::{self, JoinHandle},
};

pub struct RuntimeThread {
    name: String,
    handle: Option<JoinHandle<()>>,
}

impl RuntimeThread {
    pub fn spawn(name: impl Into<String>, body: impl FnOnce() + Send + 'static) -> io::Result<Self> {
        let name = name.into();
        let handle = thread::Builder::new().name(name.clone()).spawn(body)?;
        Ok(Self {
            name,
            handle: Some(handle),
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn join(mut self) -> thread::Result<()> {
        self.handle.take().unwrap().join()
    }
}
