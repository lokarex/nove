use crate::device::Device;

pub trait Model {
    fn to_device(&mut self, device: &Device);
}
