use std::io;
mod lexer;
// add   = mul | add, ( "+" | "-" ), mul;
// mul   = unary | mul, ( "*" | "/"), unary;
// unary = ( "+" | "-" )? atom;
// atom  = unum | "(" add ")";
// unum  = digit , { digit };
// digit = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9";

fn prompt(s: &str) -> io::Result<()> {
    use std::io::{stdout, Write};
    let stdout = stdout();
    let mut stdout = stdout.lock();
    stdout.write(s.as_bytes())?;
    stdout.flush()
}

fn main() {
    use std::io::{stdin, BufRead, BufReader};

    let stdin = stdin();
    let stdin = stdin.lock();
    let stdin = BufReader::new(stdin);
    let mut lines = stdin.lines();

    loop {
        prompt("> ").unwrap();
        if let Some(Ok(line)) = lines.next() {
            let token = lexer::run(&line);
            println!("{:?}", token);
        } else {
            break;
        }
    }
}
