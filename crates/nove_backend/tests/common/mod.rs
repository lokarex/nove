use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

pub struct TestDir {
    root: PathBuf,
    path: PathBuf,
}

impl TestDir {
    pub fn new() -> io::Result<Self> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target");
        fs::create_dir_all(&root)?;
        let root = root.canonicalize()?;
        loop {
            let id = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let path = root.join(format!(
                "backend-contract-tests-{}-{id}",
                std::process::id()
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self { root, path }),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
    }

    pub fn file(&self, name: &str) -> PathBuf {
        let mut components = Path::new(name).components();
        assert!(matches!(components.next(), Some(Component::Normal(_))));
        assert!(components.next().is_none());
        self.path.join(name)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        if self.path.parent() == Some(self.root.as_path())
            && self.path.canonicalize().ok().as_deref() == Some(self.path.as_path())
        {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}
