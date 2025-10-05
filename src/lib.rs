pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}

pub struct Foo {
    bar: bool,
}

impl Default for Foo {
    fn default() -> Self {
        Self { bar: true }
    }
}

fn main() {
    let a = Foo::default();
    println!("{}", a.bar);
}